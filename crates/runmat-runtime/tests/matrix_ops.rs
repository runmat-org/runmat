#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use runmat_builtins::{builtin_functions, Tensor as Matrix, Value};
use runmat_runtime::{call_builtin, comparison::*, indexing::*, matrix::*};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_matrix_arithmetic() {
    let a = Matrix::new_2d(vec![1.0, 3.0, 2.0, 4.0], 2, 2).unwrap();
    let b = Matrix::new_2d(vec![2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();

    // Test addition
    let c = matrix_add(&a, &b).unwrap();
    assert_eq!(c.data, vec![3.0, 4.0, 3.0, 6.0]);

    // Test subtraction
    let d = matrix_sub(&a, &b).unwrap();
    assert_eq!(d.data, vec![-1.0, 2.0, 1.0, 2.0]);

    // Test matrix multiplication
    let e = matrix_mul(&a, &b).unwrap();
    // [1 2] * [2 1] = [4 5]
    // [3 4]   [1 2]   [10 11]
    assert_eq!(e.data, vec![4.0, 10.0, 5.0, 11.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_scalar_operations() {
    let a = Matrix::new_2d(vec![1.0, 3.0, 2.0, 4.0], 2, 2).unwrap();

    // Test scalar multiplication
    let b = matrix_scalar_mul(&a, 2.0);
    assert_eq!(b.data, vec![2.0, 6.0, 4.0, 8.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_matrix_transpose() {
    // Build A as 3x2 with column-major data for [[1,2],[3,4],[5,6]]
    let a = Matrix::new_2d(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 3, 2).unwrap();
    let b = match call_builtin("transpose", &[Value::Tensor(a)]).unwrap() {
        Value::Tensor(t) => t,
        other => panic!("expected tensor transpose, got {other:?}"),
    };

    assert_eq!(b.rows(), 2);
    assert_eq!(b.cols(), 3);
    // Transpose should yield [[1,3,5],[2,4,6]] with col-major data [1,2,3,4,5,6]
    assert_eq!(b.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_matrix_eye() {
    let eye3 = matrix_eye(3);
    assert_eq!(eye3.rows(), 3);
    assert_eq!(eye3.cols(), 3);
    assert_eq!(eye3.data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_comparison_operations() {
    let a = Matrix::new_2d(vec![1.0, 3.0, 2.0, 4.0], 2, 2).unwrap();
    let b = Matrix::new_2d(vec![2.0, 2.0, 2.0, 2.0], 2, 2).unwrap();

    // Test greater than
    let gt_result = matrix_gt(&a, &b).unwrap();
    assert_eq!(gt_result.data, vec![0.0, 1.0, 0.0, 1.0]);

    // Test less than
    let lt_result = matrix_lt(&a, &b).unwrap();
    assert_eq!(lt_result.data, vec![1.0, 0.0, 0.0, 0.0]);

    // Test equality
    let eq_result = matrix_eq(&a, &b).unwrap();
    assert_eq!(eq_result.data, vec![0.0, 0.0, 1.0, 0.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_matrix_indexing() {
    let mut matrix = Matrix::new_2d(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3).unwrap();

    // Test getting element (1-based indexing)
    assert_eq!(matrix_get_element(&matrix, 1, 2).unwrap(), 2.0);
    assert_eq!(matrix_get_element(&matrix, 2, 3).unwrap(), 6.0);

    // Test setting element
    matrix_set_element(&mut matrix, 1, 1, 10.0).unwrap();
    assert_eq!(matrix.get2(0, 0).unwrap(), 10.0);

    // Test 0-based indexing error
    assert!(matrix_get_element(&matrix, 0, 1).is_err());
    assert!(matrix_set_element(&mut matrix, 1, 0, 5.0).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_row_column_access() {
    let matrix = Matrix::new_2d(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3).unwrap();

    // Test getting row
    let row1 = matrix_get_row(&matrix, 1).unwrap();
    assert_eq!(row1.data, vec![1.0, 2.0, 3.0]);
    assert_eq!(row1.rows(), 1);
    assert_eq!(row1.cols(), 3);

    let row2 = matrix_get_row(&matrix, 2).unwrap();
    assert_eq!(row2.data, vec![4.0, 5.0, 6.0]);

    // Test getting column
    let col1 = matrix_get_col(&matrix, 1).unwrap();
    assert_eq!(col1.data, vec![1.0, 4.0]);
    assert_eq!(col1.rows(), 2);
    assert_eq!(col1.cols(), 1);

    let col3 = matrix_get_col(&matrix, 3).unwrap();
    assert_eq!(col3.data, vec![3.0, 6.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_builtin_functions() {
    // Test that our new built-in functions are registered
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();

    assert!(names.contains(&"matrix_zeros"));
    assert!(names.contains(&"matrix_ones"));
    assert!(names.contains(&"matrix_eye"));
    assert!(names.contains(&"transpose"));
    assert!(names.contains(&"gt"));
    assert!(names.contains(&"lt"));
    assert!(names.contains(&"eq"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_builtin_dispatch() {
    // Test zeros function
    let result = call_builtin(
        "matrix_zeros",
        &[
            Value::Int(runmat_builtins::IntValue::I32(2)),
            Value::Int(runmat_builtins::IntValue::I32(3)),
        ],
    )
    .unwrap();
    if let Value::Tensor(m) = result {
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data, vec![0.0; 6]);
    } else {
        panic!("Expected matrix result");
    }

    // Test eye function
    let result = call_builtin(
        "matrix_eye",
        &[Value::Int(runmat_builtins::IntValue::I32(2))],
    )
    .unwrap();
    if let Value::Tensor(m) = result {
        assert_eq!(m.data, vec![1.0, 0.0, 0.0, 1.0]);
    } else {
        panic!("Expected matrix result");
    }

    // Test comparison
    let result = call_builtin("gt", &[Value::Num(3.0), Value::Num(2.0)]).unwrap();
    assert_eq!(result, Value::Num(1.0));

    let result = call_builtin("lt", &[Value::Num(1.0), Value::Num(2.0)]).unwrap();
    assert_eq!(result, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn test_matrix_dimension_errors() {
    let a = Matrix::new_2d(vec![1.0, 2.0], 1, 2).unwrap();
    let b = Matrix::new_2d(vec![1.0, 2.0, 3.0], 1, 3).unwrap();

    // Different dimensions should error
    assert!(matrix_add(&a, &b).is_err());
    assert!(matrix_sub(&a, &b).is_err());
    assert!(matrix_gt(&a, &b).is_err());

    // Test matrix multiplication
    let c = Matrix::new_2d(vec![1.0, 3.0, 2.0, 4.0], 2, 2).unwrap();

    // 1x2 * 2x2 = 1x2 (valid)
    let result = matrix_mul(&a, &c).unwrap();
    assert_eq!(result.rows(), 1);
    assert_eq!(result.cols(), 2);

    // 2x2 * 1x2 is invalid (inner dimensions don't match)
    assert!(matrix_mul(&c, &a).is_err());
}
