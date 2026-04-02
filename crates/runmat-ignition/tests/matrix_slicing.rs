mod test_helpers;

use runmat_builtins::Value;
use runmat_parser::parse;
use test_helpers::execute;
use test_helpers::lower;

#[test]
fn basic_matrix_and_slices() {
    // A(:)
    let ast = parse("A=[1,2;3,4]; v=A(:)").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(v) = &vars[1] {
        assert_eq!(v.rows(), 4);
        assert_eq!(v.cols(), 1);
        assert_eq!(v.data, vec![1.0, 3.0, 2.0, 4.0]);
    } else {
        panic!("Expected vector from A(:)");
    }

    // A(:,2)
    let ast = parse("A=[1,2;3,4]; c=A(:,2)").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(c) = &vars[1] {
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 1);
        assert_eq!(c.data, vec![2.0, 4.0]);
    } else {
        panic!("Expected column slice A(:,2)");
    }

    // A(2,:)
    let ast = parse("A=[1,2;3,4]; r=A(2,:)").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(r) = &vars[1] {
        assert_eq!(r.rows(), 1);
        assert_eq!(r.cols(), 2);
        assert_eq!(r.data, vec![3.0, 4.0]);
    } else {
        panic!("Expected row slice A(2,:)");
    }

    // A(:,:)
    let ast = parse("A=[1,2;3,4]; B=A(:,:)").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(b) = &vars[1] {
        assert_eq!(b.rows(), 2);
        assert_eq!(b.cols(), 2);
        // Column-major storage
        assert_eq!(b.data, vec![1.0, 3.0, 2.0, 4.0]);
    } else {
        panic!("Expected full slice A(:,:)");
    }
}

#[test]
fn empty_slice_from_two_arg_colon() {
    let program = parse("A = [10 20 30]; B = A(1:0); sz = size(B);").unwrap();
    let hir = lower(&program).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(b) = &vars[1] {
        assert_eq!(b.rows(), 1);
        assert_eq!(b.cols(), 0);
        assert!(b.data.is_empty());
    } else {
        panic!("Expected tensor for empty slice result");
    }
    if let Value::Tensor(sz) = &vars[2] {
        assert_eq!(sz.rows(), 1);
        assert_eq!(sz.cols(), 2);
        assert_eq!(sz.data, vec![1.0, 0.0]);
    } else {
        panic!("Expected size vector for empty slice");
    }
}

#[test]
fn empty_slice_rows_and_columns() {
    let program = parse(
        "
        M = reshape(1:12, 3, 4);
        R = M(1:0, :);
        C = M(:, 1:0);
        ",
    )
    .unwrap();
    let hir = lower(&program).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(r) = &vars[1] {
        assert_eq!(r.rows(), 0);
        assert_eq!(r.cols(), 4);
        assert!(r.data.is_empty());
    } else {
        panic!("Expected tensor for empty row slice");
    }
    if let Value::Tensor(c) = &vars[2] {
        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 0);
        assert!(c.data.is_empty());
    } else {
        panic!("Expected tensor for empty column slice");
    }
}

#[test]
fn range_to_plain_end_column_slice() {
    let program = parse(
        "
        M = reshape(1:12, 3, 4);
        C = M(:, 4:end);
        ",
    )
    .unwrap();
    let hir = lower(&program).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(c) = &vars[1] {
        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 1);
        assert_eq!(c.data, vec![10.0, 11.0, 12.0]);
    } else {
        panic!("Expected tensor for M(:, 4:end)");
    }
}

#[test]
fn mixed_end_bounded_and_plain_ranges_gather() {
    let program = parse(
        "
        M = reshape(1:20, 4, 5);
        S = M(2:end-1, 3:4);
        ",
    )
    .unwrap();
    let hir = lower(&program).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(s) = &vars[1] {
        assert_eq!(s.rows(), 2);
        assert_eq!(s.cols(), 2);
        assert_eq!(s.data, vec![10.0, 11.0, 14.0, 15.0]);
    } else {
        panic!("Expected tensor for M(2:end-1, 3:4)");
    }
}

#[test]
fn mixed_end_minus_and_plain_end_ranges_gather() {
    let program = parse(
        "
        M = reshape(1:20, 4, 5);
        S = M(2:end-1, 4:end);
        ",
    )
    .unwrap();
    let hir = lower(&program).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(s) = &vars[1] {
        assert_eq!(s.rows(), 2);
        assert_eq!(s.cols(), 2);
        assert_eq!(s.data, vec![14.0, 15.0, 18.0, 19.0]);
    } else {
        panic!("Expected tensor for M(2:end-1, 4:end)");
    }
}

#[test]
fn linear_index_preserves_numeric_index_shape() {
    let program = parse(
        "
        A = reshape(1:9, 3, 3);
        rowIdx = [1 3 5];
        colIdx = [1; 3; 5];
        R = A(rowIdx);
        C = A(colIdx);
        ",
    )
    .unwrap();
    let hir = lower(&program).unwrap();
    let vars = execute(&hir).unwrap();

    if let Value::Tensor(r) = &vars[3] {
        assert_eq!(r.rows(), 1);
        assert_eq!(r.cols(), 3);
        assert_eq!(r.data, vec![1.0, 3.0, 5.0]);
    } else {
        panic!("Expected tensor for row-shaped linear index");
    }

    if let Value::Tensor(c) = &vars[4] {
        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 1);
        assert_eq!(c.data, vec![1.0, 3.0, 5.0]);
    } else {
        panic!("Expected tensor for column-shaped linear index");
    }
}
