use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView};

#[test]
fn exact_integer_scalar_upload_preserves_64_bit_class_and_admission_rules() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let shape = [1usize, 1usize];
    let signed = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[0]),
            shape: &shape,
        })
        .expect("upload int64 prototype");
    let unsigned = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[0]),
            shape: &shape,
        })
        .expect("upload uint64 prototype");
    let signed_scalar = upload_exact_integer_scalar_like(provider, &signed, -7.0)
        .expect("integer scalar upload operation")
        .expect("representable int64 scalar");
    let unsigned_scalar = upload_exact_integer_scalar_like(provider, &unsigned, 7.0)
        .expect("integer scalar upload operation")
        .expect("representable uint64 scalar");
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&signed_scalar),
        Some(IntegerElementType::I64)
    );
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&unsigned_scalar),
        Some(IntegerElementType::U64)
    );
    assert_eq!(
        block_on(provider.download_integer(&signed_scalar))
            .expect("download int64")
            .data,
        HostIntegerDataOwned::I64(vec![-7])
    );
    assert_eq!(
        block_on(provider.download_integer(&unsigned_scalar))
            .expect("download uint64")
            .data,
        HostIntegerDataOwned::U64(vec![7])
    );
    assert!(upload_exact_integer_scalar_like(provider, &signed, 1.5)
        .expect("nonintegral scalar admission")
        .is_none());
    assert!(upload_exact_integer_scalar_like(provider, &unsigned, -1.0)
        .expect("negative unsigned scalar admission")
        .is_none());
    let handles = [signed, unsigned, signed_scalar, unsigned_scalar];
    let leases: Vec<_> = handles
        .iter()
        .map(|h| h.allocation_lease().downgrade())
        .collect();
    for handle in handles {
        handle.release().expect("release integer scalar handle");
    }
    for lease in leases {
        assert!(lease.acquire().is_none());
    }
}
