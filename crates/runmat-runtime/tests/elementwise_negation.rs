use runmat_builtins::{Matrix, Value};
use runmat_runtime::elementwise_neg;

#[test]
fn test_scalar_negation() {
    // Test numeric scalar
    let num = Value::Num(42.0);
    let result = elementwise_neg(&num).unwrap();
    assert_eq!(result, Value::Num(-42.0));
    
    // Test integer scalar
    let int = Value::Int(5);
    let result = elementwise_neg(&int).unwrap();
    assert_eq!(result, Value::Int(-5));
    
    // Test boolean scalar 
    let bool_val = Value::Bool(true);
    let result = elementwise_neg(&bool_val).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[test]
fn test_matrix_negation() {
    // Test 2x2 matrix
    let matrix = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let value = Value::Matrix(matrix);
    let result = elementwise_neg(&value).unwrap();
    
    if let Value::Matrix(result_matrix) = result {
        assert_eq!(result_matrix.data, vec![-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(result_matrix.rows, 2);
        assert_eq!(result_matrix.cols, 2);
    } else {
        panic!("Expected matrix result");
    }
}

#[test]
fn test_vector_negation() {
    // Test row vector
    let vector = Matrix::new(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let value = Value::Matrix(vector);
    let result = elementwise_neg(&value).unwrap();
    
    if let Value::Matrix(result_matrix) = result {
        assert_eq!(result_matrix.data, vec![-1.0, -2.0, -3.0]);
        assert_eq!(result_matrix.rows, 1);
        assert_eq!(result_matrix.cols, 3);
    } else {
        panic!("Expected matrix result");
    }
}

#[test]
fn test_zero_negation() {
    // Test that -0 = 0
    let zero = Value::Num(0.0);
    let result = elementwise_neg(&zero).unwrap();
    assert_eq!(result, Value::Num(-0.0));
}

#[test]
fn test_unsupported_type() {
    // Test string negation should fail
    let string_val = Value::String("test".to_string());
    let result = elementwise_neg(&string_val);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Negation not supported"));
}