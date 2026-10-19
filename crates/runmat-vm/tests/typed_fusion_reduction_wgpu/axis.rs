use super::*;

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_vm_folded_min_second_axis_full_call_numerics_and_receipt() {
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for second-axis folded minimum");
        let bytecode = test_helpers::compile_source(
            "x = gpuArray(reshape(single(1:16384), 128, 128)); y = min(x + x, [], 2); z = gather(y);",
        )
        .expect("compile authentic second-axis folded minimum");
        assert_compiled_folded_graph(&bytecode, 1);
        assert_actual_folded_full_call_shape(&bytecode, &[128, 1], &expected_row_min(2.0));
    });
}
