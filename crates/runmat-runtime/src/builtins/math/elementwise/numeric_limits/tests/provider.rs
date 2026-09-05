use futures::executor::block_on;
use runmat_accelerate_api::{HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView};
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::{gpu_helpers, test_support};

use super::{flintmax, intmax, realmax};

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
        provider.free(&prototype).ok();
        provider.free(&output).ok();
    });
}

#[test]
fn integer_like_rejects_contradictory_class_metadata() {
    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[1]),
                shape: &[1, 1],
            })
            .expect("integer prototype upload");
        runmat_accelerate_api::set_handle_class_identity(&prototype, "double");
        let error = intmax(vec![
            Value::from("like"),
            Value::GpuTensor(prototype.clone()),
        ])
        .expect_err("contradictory resident prototype must reject");
        assert!(error.message().contains("contradictory class metadata"));
        provider.free(&prototype).ok();
    });
}

#[test]
fn floating_like_preserves_single_representation_and_intent() {
    test_support::with_f32_test_provider(|provider| {
        let prototype = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).expect("prototype");
        let mut handle = gpu_helpers::upload_tensor(provider, &prototype).expect("upload");
        runmat_accelerate_api::set_handle_provenance(
            &mut handle,
            runmat_accelerate_api::GpuHandleProvenance::Explicit,
        );

        let output = realmax(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
            .expect("resident realmax like");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident single output")
        };
        assert_eq!(output.shape, vec![1, 1]);
        assert_eq!(output.device_id, handle.device_id);
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
        provider.free(&handle).ok();
        provider.free(&output).ok();
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
        provider.free(&handle).ok();
        provider.free(&output).ok();
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
    provider.free(&prototype).ok();
    provider.free(&output).ok();
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
    provider.free(&handle).ok();
    provider.free(&output).ok();
}
