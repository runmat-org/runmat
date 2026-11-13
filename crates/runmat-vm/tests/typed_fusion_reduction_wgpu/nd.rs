use super::*;
use runmat_accelerate_api::{AccelProvider, ProviderPrecision};

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_vm_rank_three_third_axis_folded_min_full_call() {
    runmat_gc::gc_test_context(|| {
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for rank-three extrema qualification");
        assert_eq!(provider.precision(), ProviderPrecision::F32);
        let bytecode = test_helpers::compile_source(
            "x = gpuArray(reshape(single(1:16384), 4, 64, 64)); y = min(x + x, [], 3); z = gather(y);",
        )
        .expect("compile authentic third-axis folded minimum");
        assert_compiled_folded_graph(&bytecode, 1);
        assert_actual_folded_full_call_shape(
            &bytecode,
            &[4, 64, 1],
            &typed_reduction_graph::expected_third_axis_min(2.0),
        );
    });
}
