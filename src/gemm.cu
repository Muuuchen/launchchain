#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <torch/torch.h>

#include <cstdio>
#include <iostream>

#include "gemm.cuh"
using namespace cute;
void gemm_wrapper(int M, int N, int K, cutlass::float_e4m3_t const *ptrA,
                  cutlass::float_e5m2_t const *ptrB, cutlass::float_e4m3_t *ptrC,
                  cutlass::float_e4m3_t const *ptrD, float overlap_ratio, float prefetch_ratio) {
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // A matrix configuration
    using ElementA = cutlass::float_e4m3_t;     // Element type for A matrix operand
    using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
    constexpr int AlignmentA =
        128 /
        cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A matrix
                                                // in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = cutlass::float_e5m2_t;        // Element type for B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
    constexpr int AlignmentB =
        128 /
        cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B matrix
                                                // in units of elements (up to 16 bytes)

    // C matrix configuration
    using ElementC = cutlass::float_e4m3_t;        // Element type for C and D matrix operands
    using LayoutC = cutlass::layout::ColumnMajor;  // Layout type for C and D matrix operands
    constexpr int AlignmentC =
        128 /
        cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C matrix
                                                // in units of elements (up to 16 bytes)

    // D matrix configuration
    using ElementD = ElementC;
    using LayoutD = LayoutC;
    constexpr int AlignmentD = AlignmentC;

    // Core kernel configurations
    using ElementAccumulator = float;                      // Element type for internal accumulation
    using ElementCompute = float;                          // Element type for epilogue computation
    using ArchTag = cutlass::arch::Sm90;                   // Tag indicating the minimum SM that
                                                           // supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
    using TileShape = Shape<_64, _64, _128>;               // Threadblock-level tile size
    // Cluster_N > 1 is not supported yet.
    using ClusterShape = Shape<_1, _1, _1>;  // Shape of the threadblocks in a cluster
    using KernelSchedule =
        cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                             CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm;

    // Extract information from Gemm kernel.
    using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
    using ElementScalar = typename EpilogueOutputOp::ElementScalar;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    /// Initialization
    StrideA stride_A;
    StrideB stride_B;
    StrideC stride_C;
    StrideD stride_D;

    // options l means batch size
    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto a_coord = cutlass::make_Coord(M * 1, K);
    auto c_coord = cutlass::make_Coord(M * 1, N);
    auto b_coord = cutlass::make_Coord(K, N * 1);

    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       {M, N, K, 1},
                                       {ptrA, stride_A, ptrB, stride_B},
                                       {{},  // epilogue.thread
                                        ptrC,
                                        stride_C,
                                        ptrD,
                                        stride_D}};

    auto &fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = 1.00f;
    fusion_args.beta = 0.00f;
    //   fusion_args.alpha_ptr = scalar_alpha.device_data();
    //   fusion_args.beta_ptr = scalar_beta.device_data();
    ///////////////////////////////////
    arguments.mainloop.overlap_ratio = overlap_ratio;
    arguments.mainloop.prefetch_ratio = prefetch_ratio;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    CUTLASS_CHECK(gemm.can_implement(arguments));

    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run(nullptr, nullptr, /* launch_with_pdl = */ true));
}

void gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D,
                 float overlap_ratio, float prefetch_ratio) {
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];
    const int K = A.sizes()[1];
    cutlass::float_e4m3_t const *ptrA = reinterpret_cast<cutlass::float_e4m3_t *>(A.data_ptr());
    cutlass::float_e5m2_t const *ptrB = reinterpret_cast<cutlass::float_e5m2_t *>(B.data_ptr());
    cutlass::float_e4m3_t *ptrC = reinterpret_cast<cutlass::float_e4m3_t *>(C.data_ptr());
    cutlass::float_e4m3_t const *ptrD = reinterpret_cast<cutlass::float_e4m3_t *>(D.data_ptr());
    gemm_wrapper(M, N, K, ptrA, ptrB, ptrC, ptrD, overlap_ratio, prefetch_ratio);
}

void gemm_with_prefetch_type_check(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                                   torch::Tensor D, float overlap_ratio, float prefetch_ratio) {
    if (A.dtype() != torch::kFloat8_e4m3fn || B.dtype() != torch::kFloat8_e5m2 ||
        C.dtype() != torch::kFloat8_e4m3fn || D.dtype() != torch::kFloat8_e4m3fn) {
        throw std::runtime_error("Unsupported data type for A");
    } else {
        gemm_unpack(A, B, C, D, overlap_ratio, prefetch_ratio);
    }
}
