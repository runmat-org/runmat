use super::*;
use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor};

fn classify_with_flag(flag: Value) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(isstrprop_builtin(
        None,
        Value::String("A".into()),
        Value::String("alpha".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("ForceCellOutput".into()),
            flag,
        ]),
    ))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn double_scalar_and_one_element_dense_carriers_join_catalog_and_runtime() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let entry = runmat_builtins::builtin_catalog_entry_by_name("isstrprop")
        .expect("canonical isstrprop catalog entry");
    for (shape, storage, literal, flag) in [
        (
            ShapeFact::Scalar,
            StorageFact::Scalar,
            LiteralValue::Number(1.0),
            Value::Num(1.0),
        ),
        (
            ShapeFact::from(vec![Some(1), Some(1)]),
            StorageFact::Dense,
            LiteralValue::Matrix(vec![vec![LiteralValue::Number(1.0)]]),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        ),
    ] {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    shape,
                    storage,
                ),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::String("A".into()),
                LiteralValue::String("alpha".into()),
                LiteralValue::String("ForceCellOutput".into()),
                literal,
            ]),
            outputs: OutputSelection::default(),
        };
        let inferred = runmat_builtins::infer_catalog_call(entry, &request);
        assert!(inferred.diagnostics.is_empty(), "{inferred:?}");
        assert!(matches!(&inferred.outputs[0].kind, ValueKindFact::Cell(_)));
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
        assert!(matches!(
            classify_with_flag(flag).expect("admitted Double flag"),
            Value::Cell(cell)
                if cell.shape == vec![1, 1] && cell.data() == vec![Value::Bool(true)]
        ));
    }
    for flag in [
        Value::Num(0.0),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![-0.0], vec![1, 1]).unwrap()),
    ] {
        assert_eq!(classify_with_flag(flag).unwrap(), Value::Bool(true));
    }
    assert!(matches!(
        classify_with_flag(Value::LogicalArray(
            LogicalArray::new(vec![1], vec![1, 1]).unwrap()
        ))
        .unwrap(),
        Value::Cell(_)
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn nonbinary_wrong_class_empty_and_nonscalar_flags_reject_without_conversion() {
    use std::error::Error;

    for flag in [
        Value::Num(2.0),
        Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![f64::INFINITY], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap()),
        Value::Int(IntValue::U8(1)),
        Value::Complex(1.0, 0.0),
        Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
    ] {
        let error = classify_with_flag(flag).expect_err("invalid flag must be rejected");
        assert!(error.message().contains("ForceCellOutput must be"));
        assert!(Error::source(&error).is_none());
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn resident_flag_rejects_without_download_or_borrowed_lease_loss() {
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    test_support::with_test_provider(|provider| {
        let flag = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&[1]),
                shape: &[1, 1],
            })
            .expect("resident flag fixture");
        let weak = flag.allocation_lease().downgrade();
        let descriptor = flag.wire_descriptor();
        let before = provider.telemetry_snapshot().download_bytes;
        let error = classify_with_flag(Value::GpuTensor(flag.clone()))
            .expect_err("resident option is not a host Double flag");
        assert!(error.message().contains("ForceCellOutput must be"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, before);
        assert!(weak.acquire().is_some());
        assert_eq!(flag.wire_descriptor(), descriptor);
        drop(flag);
        assert!(weak.acquire().is_none());
    });
}
