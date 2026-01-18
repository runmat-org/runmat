mod test_helpers;

use runmat_hir::lower;
use runmat_parser::parse;
use test_helpers::execute;

#[test]
fn reshape_and_index_3d_element() {
    // Build a 2x3x2 tensor of values 1..12 and test a specific element
    let ast = parse("A = reshape([1 2 3 4 5 6 7 8 9 10 11 12], 2, 3, 2); x = A(1,2,2);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // A(1,2,2) with column-major reshape(2,3,2) is 9 in MATLAB semantics
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn mixed_selectors_basic_2d_range() {
    let ast = parse("A=[1 2 3; 4 5 6; 7 8 9]; sub = A(1:2, 2);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Should select first two rows of column 2: [2;5]
    let sub = vars
        .iter()
        .rev()
        .find(|v| matches!(v, runmat_builtins::Value::Tensor(_)))
        .unwrap();
    if let runmat_builtins::Value::Tensor(t) = sub {
        assert_eq!(t.data, vec![2.0, 5.0]);
    }
}

#[test]
fn logical_mask_rows_select() {
    let ast = parse("A=[1 2; 3 4; 5 6]; sel = A([1 0 1], :);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Expect rows 1 and 3 selected. In column-major storage, resulting 2x2 has data [1 5 2 6].
    let sel = vars
        .iter()
        .rev()
        .find(|v| matches!(v, runmat_builtins::Value::Tensor(_)))
        .unwrap();
    if let runmat_builtins::Value::Tensor(t) = sel {
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![1.0, 5.0, 2.0, 6.0]);
    }
}

#[test]
fn slice_assignment_column_and_row() {
    let ast = parse("A=[1 2 3; 4 5 6]; A(:,2) = [8;9]; A(1,:) = [7 7 7];").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Final A should be [7 7 7; 4 9 6] -> column-major data [7 4 7 9 7 6]
    let a = vars
        .iter()
        .rev()
        .find(|v| matches!(v, runmat_builtins::Value::Tensor(_)))
        .unwrap();
    if let runmat_builtins::Value::Tensor(t) = a {
        assert_eq!(t.data, vec![7.0, 4.0, 7.0, 9.0, 7.0, 6.0]);
    }
}

#[test]
fn slice_assignment_3d_entire_slice() {
    let ast = parse("A = reshape([1 2 3 4 5 6 7 8 9 10 11 12], 2, 3, 2); Z = reshape([0 0 0 0 0 0], 2, 3); A(:, :, 1) = Z;").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // After assignment, first 2x3 slice zeros; the second slice remains [7..12]
    // Pick the 3D tensor (A), not the 2D Z
    let a = vars
        .iter()
        .filter_map(|v| {
            if let runmat_builtins::Value::Tensor(t) = v {
                Some(t)
            } else {
                None
            }
        })
        .find(|t| t.shape.len() == 3)
        .unwrap();
    {
        assert_eq!(a.shape, vec![2, 3, 2]);
        // First slice (k=1) occupies every other element interleaved due to column-major strides across dims.
        // Check elements of second slice by recomputing indices rather than slicing contiguous range.
        let second_slice_vals = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut gathered = Vec::new();
        // For dim order [rows, cols, pages], column-major linear index = r-1 + (c-1)*rows + (p-1)*rows*cols
        let rows = 2usize;
        let cols = 3usize;
        let p = 2usize;
        for c in 1..=cols {
            for r in 1..=rows {
                let idx = (r - 1) + (c - 1) * rows + (p - 1) * rows * cols;
                gathered.push(a.data[idx]);
            }
        }
        assert_eq!(gathered, second_slice_vals);
    }
}
