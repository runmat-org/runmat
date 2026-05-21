#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::execute_semantic_source;

#[test]
fn linear_index_and_end_keyword() {
    // v = A(end) and A(1) linear indexing
    let vars = execute_semantic_source("A=[1,2;3,4]; v1=A(1); v2=A(end);").unwrap();
    if let Value::Num(v1) = &vars[1] {
        assert_eq!(*v1, 1.0);
    } else {
        panic!("v1");
    }
    if let Value::Num(v2) = &vars[2] {
        assert_eq!(*v2, 4.0);
    } else {
        panic!("v2");
    }
}

#[test]
fn logical_mask_indexing() {
    // A(logical([1 0 1 0])) over linearized A
    let vars = execute_semantic_source("A=[1,2;3,4]; idx=logical([1,0,1,0]); v=A(idx);").unwrap();
    // MATLAB uses column-major linearization: A(:) = [1;3;2;4], mask [1 0 1 0] selects [1,2]
    if let Value::Tensor(v) = &vars[2] {
        // MATLAB logical linear indexing returns a column vector.
        assert_eq!(v.shape, vec![2, 1]);
        assert_eq!(v.data, vec![1.0, 2.0]);
    } else {
        panic!("v");
    }
}

#[test]
fn logical_mask_linear_indexing_row_vector_returns_column() {
    let vars =
        execute_semantic_source("A=[10,20,30,40]; idx=logical([1,0,1,0]); v=A(idx);").unwrap();
    if let Value::Tensor(v) = &vars[2] {
        assert_eq!(v.shape, vec![2, 1]);
        assert_eq!(v.data, vec![10.0, 30.0]);
    } else {
        panic!("v");
    }
}

#[test]
fn logical_mask_linear_indexing_matrix_mask_returns_column() {
    let vars = execute_semantic_source("A=[1,2;3,4]; idx=logical([1,0;0,1]); v=A(idx);").unwrap();
    // mask selects A(1,1) and A(2,2) in column-major linear order
    if let Value::Tensor(v) = &vars[2] {
        assert_eq!(v.shape, vec![2, 1]);
        assert_eq!(v.data, vec![1.0, 4.0]);
    } else {
        panic!("v");
    }
}

#[test]
fn host_linear_indexing_accepts_gpu_backed_range_selector() {
    let vars =
        execute_semantic_source("a=length(1:10); k=floor(a/2)+1; x=(1:10)'; y=x(1:k);").unwrap();
    if let Value::Tensor(v) = &vars[3] {
        assert_eq!(v.shape, vec![1, 6]);
        assert_eq!(v.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    } else {
        panic!("y");
    }
}

#[test]
fn cell_paren_scalar_index_returns_cell_container() {
    let vars = execute_semantic_source("C={10,20,30}; D=C(2); x=D{1};").unwrap();
    let mut saw_container = false;
    let mut saw_unwrapped = false;
    for value in vars {
        match value {
            Value::Cell(cell) if cell.data.len() == 1 => {
                if matches!((*cell.data[0]).clone(), Value::Num(n) if (n - 20.0).abs() < 1e-9) {
                    saw_container = true;
                }
            }
            Value::Num(n) if (n - 20.0).abs() < 1e-9 => saw_unwrapped = true,
            _ => {}
        }
    }
    assert!(
        saw_container,
        "expected C(2) to produce a 1x1 cell container"
    );
    assert!(
        saw_unwrapped,
        "expected D{{1}} to unwrap to numeric payload"
    );
}

#[test]
fn cell_paren_range_index_returns_cell_subset() {
    let vars = execute_semantic_source("C={1,2,3,4}; D=C(2:3); x=D{1}; y=D{2};").unwrap();
    let mut saw_subset = false;
    let mut saw_x = false;
    let mut saw_y = false;
    for value in vars {
        match value {
            Value::Cell(cell) if cell.data.len() == 2 => {
                let first = (*cell.data[0]).clone();
                let second = (*cell.data[1]).clone();
                if matches!(first, Value::Num(n) if (n - 2.0).abs() < 1e-9)
                    && matches!(second, Value::Num(n) if (n - 3.0).abs() < 1e-9)
                {
                    saw_subset = true;
                }
            }
            Value::Num(n) if (n - 2.0).abs() < 1e-9 => saw_x = true,
            Value::Num(n) if (n - 3.0).abs() < 1e-9 => saw_y = true,
            _ => {}
        }
    }
    assert!(
        saw_subset,
        "expected C(2:3) to return a two-element cell subset"
    );
    assert!(
        saw_x && saw_y,
        "expected D{{1}} and D{{2}} values to be preserved"
    );
}

#[test]
fn range_start_selector_rejects_non_numeric_value() {
    let err = execute_semantic_source("A=1:5; s='x'; y=A(s:end);")
        .expect_err("non-numeric range start selector should fail");
    assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
}

#[test]
fn range_step_selector_rejects_non_numeric_value() {
    let err = execute_semantic_source("A=1:10; st='x'; y=A(1:st:end);")
        .expect_err("non-numeric range step selector should fail");
    assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
}

#[test]
fn scalar_store_index_rejects_negative_index() {
    let err = execute_semantic_source("A=[1,2,3]; A(-1)=0;")
        .expect_err("negative scalar store index should fail");
    assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
}

#[test]
fn scalar_store_index_rejects_zero_index() {
    let err = execute_semantic_source("A=[1,2,3]; A(0)=0;")
        .expect_err("zero scalar store index should fail");
    assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
}

#[test]
fn cell_brace_scalar_tensor_index_reads_value() {
    let vars = execute_semantic_source("C={10,20,30}; k=[2]; v=C{k};")
        .expect("scalar tensor cell index should execute");
    if let Value::Num(v) = &vars[2] {
        assert_eq!(*v, 20.0);
    } else {
        panic!("expected numeric cell value from scalar tensor index");
    }
}

#[test]
fn fractional_range_start_index_rejects_non_integer_selector() {
    let err = execute_semantic_source("A=[10,20,30,40,50]; y=A(1.5:4);")
        .expect_err("fractional range start should fail");
    assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
}

#[test]
fn fractional_range_step_index_rejects_non_integer_selector() {
    let err = execute_semantic_source("A=[10,20,30,40,50]; y=A(1:1.5:5);")
        .expect_err("fractional range step should fail");
    assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
}

#[test]
fn positive_range_index_rejects_upper_out_of_bounds_element() {
    let err = execute_semantic_source("A=[10,20,30,40,50]; y=A(4:6);")
        .expect_err("range containing upper out-of-bounds index should fail");
    assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
}

#[test]
fn positive_range_index_rejects_lower_out_of_bounds_element() {
    let err = execute_semantic_source("A=[10,20,30,40,50]; y=A(0:2);")
        .expect_err("range containing lower out-of-bounds index should fail");
    assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
}
