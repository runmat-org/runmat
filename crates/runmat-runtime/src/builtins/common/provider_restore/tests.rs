use super::{precision::align_floating_value_precision, upload::upload_value_protected};
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorHandle, GpuTensorStorage, NumericElementType,
};
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

struct F32OnlyProvider {
    upload_calls: AtomicUsize,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn precision_alignment_preserves_low_rank_real_and_complex_shapes() {
    let owner = TestAllocationOwner::new();
    for (element_type, dtype) in [
        (NumericElementType::F32, NumericDType::F32),
        (NumericElementType::F64, NumericDType::F64),
    ] {
        let prototype = owner
            .allocate_zeroed(vec![1, 1], element_type, GpuTensorStorage::Real)
            .expect("floating-point prototype");

        let real = Value::Tensor(Tensor::new(vec![1.25], Vec::new()).expect("rank-zero real"));
        let Value::Tensor(real) =
            align_floating_value_precision(real, &prototype.handle, "restore-test")
                .expect("real precision alignment")
        else {
            panic!("real result expected")
        };
        assert!(real.shape().is_empty());
        assert_eq!(real.numeric_dtype(), dtype);
        assert_eq!(real.materialize_f64(), vec![1.25]);

        let complex = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.25, 2.5), (3.75, 4.5)], vec![2]).expect("rank-one complex"),
        );
        let Value::ComplexTensor(complex) =
            align_floating_value_precision(complex, &prototype.handle, "restore-test")
                .expect("complex precision alignment")
        else {
            panic!("complex result expected")
        };
        assert_eq!(complex.shape(), &[2]);
        assert_eq!(complex.numeric_dtype(), dtype);
        assert_eq!(complex.materialize_f64(), vec![(1.25, 2.5), (3.75, 4.5)]);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn provider_restore_preserves_low_rank_shape_and_exact_owner() {
    test_support::with_test_provider(|provider| {
        for value in [
            Value::Tensor(Tensor::new(vec![1.25], Vec::new()).expect("rank-zero real")),
            Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2])
                    .expect("rank-one complex"),
            ),
        ] {
            let expected_shape = match &value {
                Value::Tensor(tensor) => tensor.shape(),
                Value::ComplexTensor(tensor) => tensor.shape(),
                _ => unreachable!("fixture is numeric"),
            }
            .to_vec();
            let Value::GpuTensor(handle) =
                upload_value_protected(provider, value, "restore-test", &[])
                    .expect("provider restore")
            else {
                panic!("resident output expected")
            };
            assert_eq!(handle.shape(), expected_shape);
            assert!(
                crate::builtins::common::gpu_helpers::exact_provider_for_handle(&handle)
                    .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(
                        owner, provider
                    ))
            );
        }
    });
}

impl runmat_accelerate_api::AccelProvider for F32OnlyProvider {
    fn upload(
        &self,
        _host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.upload_calls.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::anyhow!("unexpected upload"))
    }

    fn download<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow::anyhow!("download unsupported")) })
    }

    fn device_info(&self) -> String {
        "f32-only-test".to_string()
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F32
    }
}

#[test]
fn preserves_non_f32_exact_double_without_metadata_relabeling() {
    test_support::with_test_provider(|provider| {
        let value = 1.0000000000000002_f64;
        let output = upload_value_protected(provider, Value::Num(value), "restore-test", &[])
            .expect("double restore");
        let Value::GpuTensor(handle) = output else {
            panic!("expected resident output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&handle),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
        let downloaded = block_on(provider.download(&handle)).expect("download");
        assert_eq!(downloaded.data, vec![value]);
    });
}

#[test]
fn rejects_double_when_f32_only_provider_declines_typed_upload() {
    let provider = F32OnlyProvider {
        upload_calls: AtomicUsize::new(0),
    };
    let error = upload_value_protected(
        &provider,
        Value::Num(1.0000000000000002),
        "restore-test",
        &[],
    )
    .expect_err("f32-only provider cannot supply a double result");
    assert!(error.message().contains("failed to restore result"));
    assert_eq!(provider.upload_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn accepts_single_when_provider_supports_typed_upload() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::from_f32(vec![1.0000001], vec![1, 1]).expect("single");
        let output = upload_value_protected(provider, Value::Tensor(tensor), "restore-test", &[])
            .expect("typed single result");
        let Value::GpuTensor(handle) = output else {
            panic!("expected resident single output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&handle),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
    });
}
