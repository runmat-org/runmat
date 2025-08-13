use runmat_builtins::{Tensor, Value};
use runmat_runtime::{elementwise_pow, power};

#[test]
fn test_scalar_power() {
    // Test numeric scalar
    let result = power(&Value::Num(2.0), &Value::Num(3.0)).unwrap();
    assert_eq!(result, Value::Num(8.0));

    // Test integer scalar
    let result = power(&Value::Int(3), &Value::Int(2)).unwrap();
    assert_eq!(result, Value::Num(9.0));

    // Test mixed types
    let result = power(&Value::Num(2.5), &Value::Int(2)).unwrap();
    assert_eq!(result, Value::Num(6.25));
}

#[test]
fn test_matrix_power() {
    // Test A^0 = I (identity)
    let matrix = Tensor::new_2d(vec![2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
    let result = power(&Value::Tensor(matrix), &Value::Int(0)).unwrap();

    if let Value::Tensor(result_matrix) = result {
        assert_eq!(result_matrix.data, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(result_matrix.rows(), 2);
        assert_eq!(result_matrix.cols(), 2);
    } else {
        panic!("Expected matrix result");
    }

    // Test A^1 = A
    let matrix = Tensor::new_2d(vec![2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
    let expected = matrix.clone();
    let result = power(&Value::Tensor(matrix), &Value::Int(1)).unwrap();

    if let Value::Tensor(result_matrix) = result {
        assert_eq!(result_matrix.data, expected.data);
        assert_eq!(result_matrix.rows(), expected.rows());
        assert_eq!(result_matrix.cols(), expected.cols());
    } else {
        panic!("Expected matrix result");
    }

    // Test A^2 = A * A
    // A = [1 3; 2 4] in column-major
    let matrix = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let result = power(&Value::Tensor(matrix), &Value::Int(2)).unwrap();

    if let Value::Tensor(result_matrix) = result {
        // [1 3; 2 4]^2 = [7 15; 10 22] => column-major [7,10,15,22]
        assert_eq!(result_matrix.data, vec![7.0, 10.0, 15.0, 22.0]);
    } else {
        panic!("Expected matrix result");
    }
}

#[test]
fn test_matrix_power_float_integer() {
    // Test A^2.0 (float that's an integer)
    let matrix = Tensor::new_2d(vec![2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();
    let result = power(&Value::Tensor(matrix), &Value::Num(2.0)).unwrap();

    if let Value::Tensor(result_matrix) = result {
        // [2,1;1,2]^2 = [5,4;4,5] -> data [5,4,4,5]
        assert_eq!(result_matrix.data, vec![5.0, 4.0, 4.0, 5.0]);
    } else {
        panic!("Expected matrix result");
    }
}

#[test]
fn test_matrix_power_non_integer_fails() {
    let matrix = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let result = power(&Value::Tensor(matrix), &Value::Num(2.5));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("integer exponent"));
}

#[test]
fn test_matrix_power_non_square_fails() {
    let matrix = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap(); // 2x3 matrix
    let result = power(&Value::Tensor(matrix), &Value::Int(2));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("square"));
}

#[test]
fn test_matrix_power_negative_fails() {
    let matrix = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let result = power(&Value::Tensor(matrix), &Value::Int(-1));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Negative"));
}

#[test]
fn test_elementwise_vs_matrix_power() {
    // A = [2 4; 3 5] column-major
    let matrix = Tensor::new_2d(vec![2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();

    // Matrix power: A^2 = A * A
    let matrix_power_result = power(&Value::Tensor(matrix.clone()), &Value::Int(2)).unwrap();

    // Element-wise power: A.^2 = [a_ij^2]
    let elementwise_result = elementwise_pow(&Value::Tensor(matrix), &Value::Int(2)).unwrap();

    // Results should be different
    if let (Value::Tensor(m1), Value::Tensor(m2)) = (matrix_power_result, elementwise_result) {
        // Matrix: [2 4; 3 5]^2 = [16 28; 21 37] -> column-major [16,21,28,37]
        assert_eq!(m1.data, vec![16.0, 21.0, 28.0, 37.0]);

        // Element-wise: [2,3;4,5].^2 = [4,9;16,25]
        assert_eq!(m2.data, vec![4.0, 9.0, 16.0, 25.0]);
    } else {
        panic!("Expected matrix results");
    }
}

#[test]
fn test_unsupported_power_combinations() {
    let matrix = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let string_val = Value::String("test".to_string());

    // String^matrix should fail
    let result = power(&string_val, &Value::Tensor(matrix.clone()));
    assert!(result.is_err());

    // Matrix^matrix should fail (not supported for regular power)
    let result = power(&Value::Tensor(matrix.clone()), &Value::Tensor(matrix));
    assert!(result.is_err());
}
