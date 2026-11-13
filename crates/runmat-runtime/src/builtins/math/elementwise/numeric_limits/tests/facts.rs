use super::*;
use crate::value_fact::value_fact;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
};
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor};

fn check(name: &str, call: fn(Vec<Value>) -> BuiltinResult<Value>, prototype: Value) -> Value {
    let arguments = vec![Value::from("like"), prototype];
    let inferred = runmat_builtins::infer_catalog_call(
        runmat_builtins::builtin_catalog_entry_by_name(name).unwrap(),
        &CallRequest {
            arguments: arguments.iter().map(value_fact).collect(),
            literals: LiteralContext::new(vec![
                LiteralValue::Keyword("like".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{name}: {:?}",
        inferred.diagnostics
    );
    let output = call(arguments).unwrap();
    let actual = value_fact(&output);
    assert_eq!(inferred.outputs[0].kind, actual.kind, "{name}");
    assert_eq!(inferred.outputs[0].storage, actual.storage, "{name}");
    assert_eq!(
        inferred.outputs[0].shape.element_count(),
        actual.shape.element_count(),
        "{name}"
    );
    output
}

#[test]
fn numeric_limit_static_representation_matches_actual_host_carriers() {
    check(
        "realmax",
        realmax,
        Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
    );
    check(
        "eps",
        eps,
        Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
    );
    check(
        "realmin",
        realmin,
        Value::ComplexTensor(ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap()),
    );
    check(
        "intmax",
        intmax,
        Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![0]),
                    IntegerStorage::I16(vec![1]),
                )
                .unwrap(),
                vec![1, 1],
            )
            .unwrap(),
        ),
    );
    check(
        "flintmax",
        flintmax,
        Value::SparseTensor(
            runmat_value::SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap(),
        ),
    );
}

#[test]
fn numeric_limit_static_storage_matches_actual_resident_scalar_carrier() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        for (name, call, input) in [
            (
                "realmax",
                realmax as fn(Vec<Value>) -> BuiltinResult<Value>,
                Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap(),
            ),
            (
                "intmax",
                intmax,
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
            ),
        ] {
            let source =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &input).unwrap();
            let output = check(name, call, Value::GpuTensor(source.clone()));
            let Value::GpuTensor(output) = output else {
                panic!("resident scalar result");
            };
            let source_weak = source.downgrade();
            let output_weak = output.downgrade();
            source.release().expect("release prototype");
            output.release().expect("release output");
            assert!(source_weak.acquire().is_err());
            assert!(output_weak.acquire().is_err());
        }
    });
}
