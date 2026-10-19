use super::*;
use runmat_accelerate_api::{AccelProvider, ProviderPrecision};

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_vm_folded_min_include_omit_nan_and_signed_zero_full_call() {
    runmat_gc::gc_test_context(|| {
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for NaN extrema qualification");
        assert_eq!(provider.precision(), ProviderPrecision::F32);
        for omit in [false, true] {
            block_on(special_cases::assert_borrowed_nan_min(provider, omit));
        }
        block_on(special_cases::assert_empty_min_ordinary_fallback());
    });
}
