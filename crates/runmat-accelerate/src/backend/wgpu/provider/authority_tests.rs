use super::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::WgpuAccessError;

#[test]
#[ignore = "requires two actual WGPU devices; run explicitly"]
fn wrong_access_cannot_enter_canonical_emission_and_raw_export_stays_sticky() {
    let first = WgpuProvider::new(WgpuProviderOptions::default()).expect("first WGPU device");
    let second = WgpuProvider::new(WgpuProviderOptions::default()).expect("second WGPU device");
    let entered = std::cell::Cell::new(false);
    assert!(matches!(
        first
            .access
            .with_profiled_shared(&second.emission_authority, |_| {
                entered.set(true);
            }),
        Err(WgpuAccessError::WrongDevice)
    ));
    assert!(!entered.get(), "wrong authority must reject before effects");

    assert!(!first.access.has_raw_export());
    first
        .access
        .with_shared(|emission| {
            let _raw = emission.device();
            assert!(first.access.has_raw_export());
        })
        .expect("raw lexical emission");
    assert!(
        first.access.has_raw_export(),
        "raw export survives later scopes"
    );
    assert!(
        !second.access.has_raw_export(),
        "other access stays independent"
    );
}

#[test]
#[ignore = "requires actual WGPU device; run explicitly"]
fn mapped_initializer_records_unmeasured_effect_before_device_call() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("WGPU device");
    assert!(!provider.access.has_unmeasured_implicit_transfer());
    provider
        .access
        .with_profiled_shared(&provider.emission_authority, |emission| {
            let _buffer = emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                label: Some("runmat-provenance-test"),
                contents: &[1, 2, 3, 4],
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            assert!(provider.access.has_unmeasured_implicit_transfer());
        })
        .expect("canonical emission");
    assert!(provider.access.has_unmeasured_implicit_transfer());
    assert!(!provider.access.has_raw_export());
}
