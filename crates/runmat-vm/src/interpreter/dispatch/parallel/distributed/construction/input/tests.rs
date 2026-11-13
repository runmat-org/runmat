use super::*;
use futures::executor::block_on;
use once_cell::sync::Lazy;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{GpuHandleProvenance, ThreadProviderGuard};
use runmat_value::{NumericStorage, Tensor};

static PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);

#[test]
fn automatic_constructor_payloads_gather_exactly_but_explicit_intent_is_preserved() {
    runmat_gc::gc_test_context(|| {
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        for storage in [
            NumericStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            NumericStorage::U32(vec![1, u32::MAX]),
            NumericStorage::F32(vec![1.25, -2.5]),
        ] {
            let tensor = Tensor::from_numeric_storage(storage, vec![1, 2]).unwrap();
            let handle =
                runmat_runtime::builtins::common::gpu_helpers::upload_tensor(&*PROVIDER, &tensor)
                    .unwrap()
                    .with_provenance(GpuHandleProvenance::Automatic)
                    .unwrap();
            // This directly supplies automatic residency: disabling automatic
            // placement cannot accidentally make the boundary witness pass.
            let (host, _roots) = block_on(prepare(Value::GpuTensor(handle.clone()))).unwrap();
            let Value::Tensor(actual) = host else {
                panic!("expected host tensor");
            };
            assert_eq!(actual.numeric_dtype(), tensor.numeric_dtype());
            assert_eq!(actual.shape(), tensor.shape());
            for index in 0..2 {
                assert_eq!(
                    actual.numeric_value_at(index),
                    tensor.numeric_value_at(index)
                );
            }
            let explicit = handle
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            let error = block_on(prepare(Value::GpuTensor(explicit))).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:DistributedExplicitDeviceIntent")
            );
        }
        let (host, _roots) = block_on(prepare(Value::Num(7.0))).unwrap();
        assert!(matches!(host, Value::Num(7.0)));
    });
}
