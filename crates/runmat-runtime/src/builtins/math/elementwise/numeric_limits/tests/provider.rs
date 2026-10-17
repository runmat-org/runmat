use futures::executor::block_on;
use runmat_accelerate_api::{HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView};
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::{gpu_helpers, test_support};

use super::{eps, flintmax, intmax, realmax};

#[test]
fn unsupported_resident_spacing_never_downloads_explicit_input() {
    test_support::with_f32_test_provider(|provider| {
        let input = Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        provider.reset_telemetry();
        assert!(eps(vec![Value::GpuTensor(handle.clone())]).is_err());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        handle.allocation_lease().validate_live().unwrap();
        handle.release().unwrap();
    });
}

#[test]
fn integer_like_preserves_class_and_wide_value() {
    test_support::with_test_provider(|provider| {
        let shape = [1usize, 1usize];
        let prototype = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[9_007_199_254_740_993]),
                shape: &shape,
            })
            .expect("integer prototype upload");

        let output = intmax(vec![
            Value::from("like"),
            Value::GpuTensor(prototype.clone()),
        ])
        .expect("gpu intmax like");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident integer output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let downloaded = block_on(provider.download_integer(&output)).expect("download");
        assert_eq!(downloaded.data, HostIntegerDataOwned::U64(vec![u64::MAX]));
        assert_eq!(downloaded.shape, vec![1, 1]);
        prototype.release().expect("release prototype");
        output.release().expect("release output");
    });
}

#[test]
fn integer_like_class_is_immutable_and_contradictory_wire_class_rejects() {
    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[1]),
                shape: &[1, 1],
            })
            .expect("integer prototype upload");
        let original = prototype.wire_descriptor();
        let acquired = provider
            .acquire_live(&original)
            .expect("exact native acquisition");
        assert!(acquired.shares_allocation_with(&prototype));
        drop(acquired);
        let mut contradictory = original.clone();
        contradictory.physical = runmat_accelerate_api::GpuPhysicalDescriptor::new(
            runmat_accelerate_api::NumericElementType::F64,
            original.physical.storage(),
            original.physical.element_capacity(),
            original.physical.payload_bytes(),
            original.physical.requested_bytes(),
        )
        .unwrap();
        provider.reset_telemetry();
        let error = provider
            .acquire_live(&contradictory)
            .expect_err("changed native class");
        assert_eq!(
            error,
            runmat_accelerate_api::allocation::AllocationAcquisitionError::PhysicalMismatch
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(prototype.wire_descriptor(), original);
        let output = intmax(vec![
            Value::from("like"),
            Value::GpuTensor(prototype.clone()),
        ])
        .expect("original native prototype remains valid");
        drop(output);
        let weak = prototype.downgrade();
        prototype.release().expect("release prototype");
        assert!(weak.acquire().is_err());
    });
}

#[test]
fn floating_like_preserves_single_representation_and_intent() {
    test_support::with_f32_test_provider(|provider| {
        let prototype = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).expect("prototype");
        let handle = gpu_helpers::upload_tensor(provider, &prototype)
            .expect("upload")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit prototype view");

        let output = realmax(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
            .expect("resident realmax like");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident single output")
        };
        assert_eq!(output.shape(), &[1, 1]);
        assert_eq!(output.device_id(), handle.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            runmat_accelerate_api::GpuTensorStorage::Real
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(output.clone()),
        ))
        .expect("gather");
        let Value::Tensor(gathered) = gathered else {
            panic!("expected gathered single tensor")
        };
        assert_eq!(gathered.as_f32_slice(), Some([f32::MAX].as_slice()));
        handle.release().expect("release handle");
        output.release().expect("release output");
    });

    test_support::with_f32_test_provider(|provider| {
        let prototype = ComplexTensor::from_f32(vec![(1.0, -2.0)], vec![1, 1]).expect("prototype");
        let handle = gpu_helpers::upload_complex_tensor(provider, &prototype).expect("upload");
        let output = flintmax(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
            .expect("resident complex flintmax like");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident complex single output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(output.clone()),
        ))
        .expect("gather");
        let Value::ComplexTensor(gathered) = gathered else {
            panic!("expected gathered complex single tensor")
        };
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            gathered.as_f32_slice(),
            Some([runmat_value::ComplexElement(2f32.powi(24), 0.0)].as_slice())
        );
        handle.release().expect("release handle");
        output.release().expect("release output");
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn integer_like_preserves_wgpu_class_and_wide_value() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let shape = [1usize, 1usize];
    let prototype = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[9_007_199_254_740_993]),
            shape: &shape,
        })
        .expect("WGPU integer prototype upload");

    let output = super::intmin(vec![
        Value::from("like"),
        Value::GpuTensor(prototype.clone()),
    ])
    .expect("WGPU intmin like");
    let Value::GpuTensor(output) = output else {
        panic!("expected resident integer output")
    };
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::I64)
    );
    let downloaded = block_on(provider.download_integer(&output)).expect("download");
    assert_eq!(downloaded.data, HostIntegerDataOwned::I64(vec![i64::MIN]));
    prototype.release().expect("release prototype");
    output.release().expect("release output");
}

#[test]
#[cfg(feature = "wgpu")]
fn floating_like_preserves_wgpu_single_class_and_value() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let prototype = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("prototype");
    let handle = gpu_helpers::upload_tensor(provider, &prototype).expect("upload");

    let output = super::realmin(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
        .expect("WGPU realmin like");
    let Value::GpuTensor(output) = output else {
        panic!("expected resident single output")
    };
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
    let gathered = block_on(provider.download(&output)).expect("download");
    assert_eq!(gathered.data, vec![f64::from(f32::MIN_POSITIVE)]);
    assert_eq!(gathered.shape, vec![1, 1]);
    handle.release().expect("release handle");
    output.release().expect("release output");
}
