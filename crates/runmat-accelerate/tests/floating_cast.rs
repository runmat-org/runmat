use futures::executor::block_on;
#[path = "support/retirement.rs"]
mod retirement;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, NumericElementType,
};

#[test]
fn inprocess_single_typed_f32_is_shape_preserving_and_independently_owned() {
    let provider = InProcessProvider::new();
    for shape in [vec![0, 2, 3], vec![1, 1, 2]] {
        let data = vec![1.25_f32; shape.iter().product()];
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&data),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let output = block_on(provider.unary_single(&input)).unwrap();
        assert_eq!(output.shape(), shape);
        assert_ne!(input.view_identity(), output.view_identity());
        assert!(input.shares_allocation_with(&output));
        assert_eq!(
            output.descriptor().element_type,
            Some(NumericElementType::F32)
        );
        let weak = input.allocation_lease().downgrade();
        assert_eq!(
            input.release().unwrap(),
            runmat_accelerate_api::allocation::AllocationReleaseOutcome::ReferenceReleased
        );
        assert!(weak.acquire().is_some());
        assert_eq!(
            block_on(provider.download_numeric(&output)).unwrap().data,
            HostNumericDataOwned::F32(data)
        );
        let second = block_on(provider.unary_single(&output)).unwrap();
        assert!(second.shares_allocation_with(&output));
        assert_ne!(second.view_identity(), output.view_identity());
        second.release().unwrap();
        assert!(block_on(provider.download_numeric(&output)).is_ok());
        retirement::release_all([output]);
        assert!(weak.acquire().is_none());
    }
}

#[test]
fn inprocess_single_f64_conversion_rounds_into_genuine_f32_storage() {
    let provider = InProcessProvider::new();
    let data = [16_777_217.0, -2.5];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&data),
            shape: &[1, 1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let output = block_on(provider.unary_single(&input)).unwrap();
    assert_eq!(output.shape(), [1, 1, 2]);
    assert!(!output.shares_allocation_with(&input));
    assert_eq!(
        block_on(provider.download_numeric(&output)).unwrap().data,
        HostNumericDataOwned::F32(vec![16_777_216.0, -2.5])
    );
    retirement::release_all([output]);
    assert_eq!(
        block_on(provider.download_numeric(&input)).unwrap().data,
        HostNumericDataOwned::F64(data.to_vec())
    );
    retirement::release_all([input]);
}
