use runmat_runtime::{call_builtin, value_fact::value_fact};
use runmat_types::{
    infer_binary, infer_concatenate, infer_reduction, infer_reshape, infer_unary, DimensionFact,
    FactInference, OperatorKind,
};
use runmat_value::{Tensor, Value};

fn assert_runtime_parity(runtime: &Value, inference: FactInference) {
    assert!(
        inference.diagnostics.is_empty(),
        "unexpected static diagnostics: {:?}",
        inference.diagnostics
    );
    let runtime = value_fact(runtime);
    assert_eq!(inference.fact.kind, runtime.kind);
    assert_eq!(inference.fact.shape, runtime.shape);
    assert_eq!(inference.fact.storage, runtime.storage);
}

#[test]
fn canonical_operator_rules_match_executable_runtime_shapes_and_kinds() {
    let left = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let right = Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap());
    let runtime = call_builtin("times", &[left.clone(), right.clone()]).unwrap();
    assert_runtime_parity(
        &runtime,
        infer_binary(
            OperatorKind::ElementwiseMultiply,
            &value_fact(&left),
            &value_fact(&right),
        ),
    );

    let left = Value::Tensor(Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap());
    let right = Value::Tensor(Tensor::new(vec![2.0; 12], vec![3, 4]).unwrap());
    let runtime = call_builtin("mtimes", &[left.clone(), right.clone()]).unwrap();
    assert_runtime_parity(
        &runtime,
        infer_binary(
            OperatorKind::MatrixMultiply,
            &value_fact(&left),
            &value_fact(&right),
        ),
    );

    let runtime = call_builtin("transpose", std::slice::from_ref(&left)).unwrap();
    assert_runtime_parity(
        &runtime,
        infer_unary(OperatorKind::Transpose, &value_fact(&left)),
    );
}

#[test]
fn canonical_transform_aggregate_and_reduction_rules_match_runtime() {
    let source = Value::Tensor(Tensor::new((1..=12).map(f64::from).collect(), vec![2, 6]).unwrap());
    let runtime = call_builtin(
        "reshape",
        &[source.clone(), Value::Num(4.0), Value::Num(3.0)],
    )
    .unwrap();
    assert_runtime_parity(
        &runtime,
        infer_reshape(
            &value_fact(&source),
            vec![DimensionFact::Known(4), DimensionFact::Known(3)],
        ),
    );

    let runtime = call_builtin("sum", std::slice::from_ref(&source)).unwrap();
    assert_runtime_parity(&runtime, infer_reduction(&value_fact(&source), None));

    let left = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let right = Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]).unwrap());
    let runtime = call_builtin("horzcat", &[left.clone(), right.clone()]).unwrap();
    assert_runtime_parity(
        &runtime,
        infer_concatenate(2, &[value_fact(&left), value_fact(&right)]),
    );
}
