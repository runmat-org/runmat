#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

use runmat_builtins::Value;

fn logical_truth(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Num(value) => *value != 0.0,
        other => panic!("expected logical value, got {other:?}"),
    }
}

fn sparse_scalar(value: &Value) -> f64 {
    match value {
        Value::SparseTensor(sparse) if sparse.shape() == vec![1, 1] => {
            sparse.get(0, 0).unwrap_or(0.0)
        }
        other => panic!("expected sparse scalar value, got {other:?}"),
    }
}

#[test]
fn logical_operators_and_short_circuit() {
    let vars =
        execute_source("a = 0 && (1/0); b = 1 || (1/0); c = 0 & 5; d = 0 | 5; e = ~0; f = ~5;")
            .unwrap();
    assert!(!logical_truth(&vars[0]));
    assert!(logical_truth(&vars[1]));
    assert!(!logical_truth(&vars[2]));
    assert!(logical_truth(&vars[3]));
    assert!(logical_truth(&vars[4]));
    assert!(!logical_truth(&vars[5]));
}

#[test]
fn short_circuit_or_accepts_boolean_lhs_without_numeric_coercion() {
    let vars = execute_source(
        "tau = []; flight_duration = 10; guard = isempty(tau) || tau(end) < flight_duration;",
    )
    .unwrap();
    assert!(logical_truth(&vars[2]));
}

#[test]
fn issparse_reports_sparse_storage_through_vm_dispatch() {
    let vars = execute_source(
        "s = sparse([1 2], [1 2], [10 20], 2, 2); a = issparse(s); b = issparse([10 0; 0 20]); c = issparse(42);",
    )
    .unwrap();
    assert!(logical_truth(&vars[1]));
    assert!(!logical_truth(&vars[2]));
    assert!(!logical_truth(&vars[3]));
}

#[test]
fn full_densifies_sparse_storage_through_vm_dispatch() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 2], [10 30 20], 3, 2); a = full(s); b = full([1 0; 0 2]); c = issparse(a);",
    )
    .unwrap();
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data == vec![10.0, 0.0, 30.0, 0.0, 20.0, 0.0]
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.data == vec![1.0, 0.0, 0.0, 2.0]
    ));
    assert!(!logical_truth(&vars[3]));
}

#[test]
fn sparse_indexing_reads_stored_unstored_and_column_major_values() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 3], [10 30 23], 3, 3); a = s(1,1); b = s(2,1); c = s(8); d = s(end,end); tf = [issparse(a), issparse(b), issparse(c), issparse(d)]; e = s([1],[1]);",
    )
    .unwrap();
    assert_eq!(sparse_scalar(&vars[1]), 10.0);
    assert_eq!(sparse_scalar(&vars[2]), 0.0);
    assert_eq!(sparse_scalar(&vars[3]), 23.0);
    assert_eq!(sparse_scalar(&vars[4]), 0.0);
    assert!(matches!(
        &vars[5],
        Value::LogicalArray(logical)
            if logical.shape == vec![1, 4] && logical.data == vec![1, 1, 1, 1]
    ));
    assert_eq!(sparse_scalar(&vars[6]), 10.0);
}

#[test]
fn sparse_slice_indexing_preserves_sparse_outputs() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 3], [10 30 23], 3, 3); c = full(s(:,1)); r = full(s(2,:)); sub = s([1 2], [1 3]); d = full(sub); tf = issparse(sub); lin = full(s(:)); lin_tf = issparse(s(:)); pick = full(s([1 8])); pick_tf = issparse(s([1 8])); rev = full(s(3:-1:1,1)); full_range = full(s(1:end)); full_range_tf = issparse(s(1:end));",
    )
    .unwrap();
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor) if tensor.shape == vec![3, 1] && tensor.data == vec![10.0, 0.0, 30.0]
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor) if tensor.shape == vec![1, 3] && tensor.data == vec![0.0, 0.0, 23.0]
    ));
    assert!(matches!(
        &vars[3],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![2, 2]
                && sparse.get(0, 0) == Some(10.0)
                && sparse.get(1, 0).unwrap_or(0.0) == 0.0
                && sparse.get(0, 1).unwrap_or(0.0) == 0.0
                && sparse.get(1, 1) == Some(23.0)
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.data == vec![10.0, 0.0, 0.0, 23.0]
    ));
    assert!(logical_truth(&vars[5]));
    assert!(matches!(
        &vars[6],
        Value::Tensor(tensor)
            if tensor.shape == vec![9, 1]
                && tensor.data == vec![10.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 23.0, 0.0]
    ));
    assert!(logical_truth(&vars[7]));
    assert!(matches!(
        &vars[8],
        Value::Tensor(tensor) if tensor.shape == vec![1, 2] && tensor.data == vec![10.0, 23.0]
    ));
    assert!(logical_truth(&vars[9]));
    assert!(matches!(
        &vars[10],
        Value::Tensor(tensor) if tensor.shape == vec![3, 1] && tensor.data == vec![30.0, 0.0, 10.0]
    ));
    assert!(matches!(
        &vars[11],
        Value::Tensor(tensor)
            if tensor.shape == vec![9, 1]
                && tensor.data == vec![10.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 23.0, 0.0]
    ));
    assert!(logical_truth(&vars[12]));
}

#[test]
fn sparse_assignment_reports_stable_unsupported_identifier() {
    let err = execute_source("s = sparse([1], [1], [5], 1, 1); s(1,1) = 6;").unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:SparseAssignmentUnsupported"));

    let slice_err = execute_source("s = sparse([1], [1], [5], 2, 2); s(:,1) = 0;").unwrap_err();
    assert_eq!(
        slice_err.identifier(),
        Some("RunMat:SparseAssignmentUnsupported")
    );

    let range_err = execute_source("s = sparse([1], [1], [5], 2, 2); s(1:2,1) = 0;").unwrap_err();
    assert_eq!(
        range_err.identifier(),
        Some("RunMat:SparseAssignmentUnsupported")
    );

    let invalid_slice_err =
        execute_source("s = sparse([1], [1], [5], 2, 2); s([0]) = 0;").unwrap_err();
    assert_eq!(
        invalid_slice_err.identifier(),
        Some("RunMat:SparseAssignmentUnsupported")
    );
}
