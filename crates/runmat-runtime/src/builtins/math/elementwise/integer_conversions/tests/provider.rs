use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
    HostTensorView, IntegerElementType,
};
use runmat_types::IntegerClass;
use runmat_value::{Tensor, Value};

use crate::builtins::common::test_support;

#[test]
fn singleton_resident_inference_matches_actual_opaque_result() {
    use crate::value_fact::value_fact;
    use runmat_types::{infer_numeric_conversion, NumericClass};
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[7.0],
                shape: &[1, 1],
            })
            .unwrap();
        for (target, name) in [
            (NumericClass::Int8, "int8"),
            (NumericClass::UInt64, "uint64"),
        ] {
            let source = Value::GpuTensor(input.clone());
            let inferred = infer_numeric_conversion(&value_fact(&source), target).fact;
            let output = crate::dispatcher::call_builtin(name, &[source]).unwrap();
            let actual = value_fact(output.value());
            assert_eq!(inferred.kind, actual.kind);
            assert_eq!(inferred.shape, actual.shape);
            assert_eq!(inferred.storage, actual.storage);
            assert_eq!(inferred.residency, actual.residency);
            output.with_owned(|value| {
                let Value::GpuTensor(output) = value else {
                    panic!("resident output")
                };
                let weak = output.downgrade();
                output.release().expect("release conversion result");
                assert!(weak.acquire().is_err());
            });
        }
        let weak = input.downgrade();
        let original = block_on(provider.download(&input)).expect("caller survives conversions");
        assert_eq!(original.data, vec![7.0]);
        input.release().expect("release caller");
        assert!(weak.acquire().is_err());
    });
}

#[test]
fn every_identity_returns_typed_resident_storage() {
    test_support::with_test_provider(|provider| {
        for (name, element_type) in [
            ("int8", IntegerElementType::I8),
            ("int16", IntegerElementType::I16),
            ("int32", IntegerElementType::I32),
            ("int64", IntegerElementType::I64),
            ("uint8", IntegerElementType::U8),
            ("uint16", IntegerElementType::U16),
            ("uint32", IntegerElementType::U32),
            ("uint64", IntegerElementType::U64),
        ] {
            let source = Tensor::new(vec![-1.0, 4.4], vec![1, 2]).expect("source");
            let materialized = source.materialize_f64();
            let input = provider
                .upload(&HostTensorView {
                    data: &materialized,
                    shape: source.shape(),
                })
                .expect("upload");
            let output = crate::dispatcher::call_builtin(name, &[Value::GpuTensor(input)])
                .expect("resident conversion");
            let Value::GpuTensor(output) = output.value() else {
                panic!("{name} must remain resident");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(output),
                Some(element_type)
            );
            assert_eq!(output.device_id(), provider.device_id());
        }
    });
}

#[test]
fn resident_conversion_uses_the_input_owner_and_preserves_exact_wide_values() {
    let _guard = test_support::accel_test_lock();
    let owner: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let current: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    unsafe {
        runmat_accelerate_api::register_provider(owner);
        runmat_accelerate_api::register_provider(current);
    }
    let _current = runmat_accelerate_api::ThreadProviderGuard::set(Some(current));
    let input = owner
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[0, 1_u64 << 63, u64::MAX]),
            shape: &[1, 3],
        })
        .expect("upload");

    let output = crate::dispatcher::call_builtin("int64", &[Value::GpuTensor(input)])
        .expect("int64 conversion");
    let Value::GpuTensor(output) = output.value() else {
        panic!("conversion must remain resident");
    };
    assert_eq!(output.device_id(), owner.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_integer_class(output),
        Some(IntegerClass::Int64)
    );
    assert_eq!(
        block_on(owner.download_integer(output))
            .expect("download")
            .data,
        HostIntegerDataOwned::I64(vec![0, i64::MAX, i64::MAX])
    );
}
