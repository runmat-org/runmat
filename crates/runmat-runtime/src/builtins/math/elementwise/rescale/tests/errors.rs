use runmat_value::{IntValue, Value};

use super::super::rescale_builtin;
use super::support::tensor;

#[tokio::test]
async fn rejects_invalid_argument_grammar_and_intervals() {
    let source = tensor(vec![1.0, 2.0], vec![1, 2]);
    let unknown = rescale_builtin(source.clone(), vec![Value::from("Range"), Value::Num(1.0)])
        .await
        .expect_err("unknown option");
    assert_eq!(unknown.identifier(), Some("RunMat:rescale:InvalidArgument"));

    let interval = rescale_builtin(source, vec![Value::Num(1.0), Value::Num(1.0)])
        .await
        .expect_err("invalid interval");
    assert_eq!(
        interval.identifier(),
        Some("RunMat:rescale:InvalidArgument")
    );
}

#[tokio::test]
async fn rejects_mismatched_bounds_and_complex_input() {
    let mismatch = rescale_builtin(
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        vec![tensor(vec![0.0; 3], vec![1, 3]), Value::Num(1.0)],
    )
    .await
    .expect_err("size mismatch");
    assert_eq!(mismatch.identifier(), Some("RunMat:rescale:SizeMismatch"));

    let complex = rescale_builtin(Value::Complex(1.0, 2.0), vec![])
        .await
        .expect_err("complex");
    assert_eq!(complex.identifier(), Some("RunMat:rescale:InvalidInput"));
}

#[tokio::test]
async fn rejects_integer_values_that_binary64_cannot_represent_exactly() {
    let error = rescale_builtin(Value::Int(IntValue::U64(u64::MAX)), vec![])
        .await
        .expect_err("inexact integer boundary");
    assert_eq!(error.identifier(), Some("RunMat:rescale:InvalidInput"));
}
