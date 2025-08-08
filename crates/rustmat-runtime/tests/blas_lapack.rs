#![cfg(feature = "blas-lapack")]

use rustmat_builtins::{Matrix, Value};
use rustmat_runtime::{blas::*, call_builtin, lapack::*, matrix_transpose};

#[test]
fn test_blas_matrix_multiplication() {
    let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = Matrix::new(vec![2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();

    let result = blas_matrix_mul(&a, &b).unwrap();

    // [1 2] * [2 1] = [4 5]
    // [3 4]   [1 2]   [10 11]
    assert_eq!(result.data, vec![4.0, 5.0, 10.0, 11.0]);
    assert_eq!(result.rows, 2);
    assert_eq!(result.cols, 2);
}

#[test]
fn test_blas_matrix_vector_multiplication() {
    let matrix = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let vector = vec![2.0, 3.0];

    let result = blas_matrix_vector_mul(&matrix, &vector).unwrap();

    // [1 2] * [2] = [8]
    // [3 4]   [3]   [18]
    assert_eq!(result, vec![8.0, 18.0]);
}

#[test]
fn test_blas_dot_product() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    let result = blas_dot_product(&a, &b).unwrap();

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(result, 32.0);
}

#[test]
fn test_blas_vector_norm() {
    let vector = vec![3.0, 4.0]; // 3-4-5 triangle
    let norm = blas_vector_norm(&vector);
    assert!((norm - 5.0).abs() < 1e-10);
}

#[test]
fn test_blas_vector_operations() {
    let mut x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];
    let mut result = y.clone();

    // Test scaling
    blas_scale_vector(&mut x, 2.0);
    assert_eq!(x, vec![2.0, 4.0, 6.0]);

    // Test AXPY: result = 2.0 * x + result = 2*[2,4,6] + [4,5,6] = [8,13,18]
    blas_vector_add(2.0, &x, &mut result).unwrap();
    assert_eq!(result, vec![8.0, 13.0, 18.0]);
}

#[test]
fn test_lapack_linear_solve() {
    // Solve the system:
    // 2x + y = 5
    // x + 3y = 6
    // Solution: x = 1.8, y = 1.4

    let a = Matrix::new(vec![2.0, 1.0, 1.0, 3.0], 2, 2).unwrap();
    let b = vec![5.0, 6.0];

    let solution = lapack_solve_linear_system(&a, &b).unwrap();

    assert!((solution[0] - 1.8).abs() < 1e-10);
    assert!((solution[1] - 1.4).abs() < 1e-10);
}

#[test]
fn test_lapack_lu_decomposition() {
    let matrix = Matrix::new(vec![4.0, 3.0, 2.0, 1.0], 2, 2).unwrap();

    let lu = lapack_lu_decomposition(&matrix).unwrap();

    // Verify dimensions
    assert_eq!(lu.lu_matrix.rows, 2);
    assert_eq!(lu.lu_matrix.cols, 2);
    assert_eq!(lu.pivots.len(), 2);
}

#[test]
fn test_lapack_qr_decomposition() {
    let matrix = Matrix::new(vec![1.0, 1.0, 1.0, 2.0], 2, 2).unwrap();

    let qr = lapack_qr_decomposition(&matrix).unwrap();

    // Verify dimensions
    assert_eq!(qr.q.rows, 2);
    assert_eq!(qr.q.cols, 2);
    assert_eq!(qr.r.rows, 2);
    assert_eq!(qr.r.cols, 2);

    // Q should be orthogonal (Q^T * Q = I)
    let qt_q = blas_matrix_mul(&matrix_transpose(&qr.q), &qr.q).unwrap();

    // Check if Q^T * Q is approximately identity
    let tolerance = 1e-10;
    assert!((qt_q.data[0] - 1.0).abs() < tolerance); // [0,0] should be 1
    assert!((qt_q.data[1]).abs() < tolerance); // [0,1] should be 0
    assert!((qt_q.data[2]).abs() < tolerance); // [1,0] should be 0
    assert!((qt_q.data[3] - 1.0).abs() < tolerance); // [1,1] should be 1
}

#[test]
fn test_lapack_eigenvalues() {
    // Test with a symmetric matrix: [[2, 1], [1, 2]]
    // Eigenvalues should be 1 and 3
    let matrix = Matrix::new(vec![2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();

    let eig = lapack_eigenvalues(&matrix, false).unwrap();

    assert_eq!(eig.eigenvalues.len(), 2);

    // Sort eigenvalues for comparison
    let mut eigenvals = eig.eigenvalues;
    eigenvals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert!((eigenvals[0] - 1.0).abs() < 1e-10);
    assert!((eigenvals[1] - 3.0).abs() < 1e-10);
}

#[test]
fn test_lapack_determinant() {
    // Test determinant of [[2, 1], [1, 2]]
    // det = 2*2 - 1*1 = 3
    let matrix = Matrix::new(vec![2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();

    let det = lapack_determinant(&matrix).unwrap();

    assert!((det - 3.0).abs() < 1e-10);
}

#[test]
fn test_lapack_matrix_inverse() {
    // Test inverse of [[2, 1], [1, 2]]
    // Inverse should be [[2/3, -1/3], [-1/3, 2/3]]
    let matrix = Matrix::new(vec![2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();

    let inv = lapack_matrix_inverse(&matrix).unwrap();

    let expected = [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];

    for (i, &expected_val) in expected.iter().enumerate() {
        assert!((inv.data[i] - expected_val).abs() < 1e-10);
    }

    // Verify A * A^(-1) = I
    let identity = blas_matrix_mul(&matrix, &inv).unwrap();

    let tolerance = 1e-10;
    assert!((identity.data[0] - 1.0).abs() < tolerance);
    assert!((identity.data[1]).abs() < tolerance);
    assert!((identity.data[2]).abs() < tolerance);
    assert!((identity.data[3] - 1.0).abs() < tolerance);
}

#[test]
fn test_builtin_blas_functions() {
    // Test BLAS matrix multiplication builtin
    let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = Matrix::new(vec![2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();

    let result = call_builtin("blas_matmul", &[Value::Matrix(a), Value::Matrix(b)]).unwrap();

    if let Value::Matrix(m) = result {
        assert_eq!(m.data, vec![4.0, 5.0, 10.0, 11.0]);
    } else {
        panic!("Expected matrix result");
    }

    // Test dot product builtin
    let vec_a = vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)];
    let vec_b = vec![Value::Num(4.0), Value::Num(5.0), Value::Num(6.0)];

    let dot_result = call_builtin("dot", &[Value::Cell(vec_a), Value::Cell(vec_b)]).unwrap();

    if let Value::Num(dot) = dot_result {
        assert_eq!(dot, 32.0);
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_builtin_lapack_functions() {
    // Test linear solve builtin
    let a = Matrix::new(vec![2.0, 1.0, 1.0, 3.0], 2, 2).unwrap();
    let b_vec = vec![Value::Num(5.0), Value::Num(6.0)];

    let solution = call_builtin("solve", &[Value::Matrix(a), Value::Cell(b_vec)]).unwrap();

    if let Value::Cell(sol) = solution {
        assert_eq!(sol.len(), 2);
        if let (Value::Num(x), Value::Num(y)) = (&sol[0], &sol[1]) {
            assert!((x - 1.8).abs() < 1e-10);
            assert!((y - 1.4).abs() < 1e-10);
        } else {
            panic!("Expected numeric solution");
        }
    } else {
        panic!("Expected vector result");
    }

    // Test determinant builtin
    let matrix = Matrix::new(vec![2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();
    let det_result = call_builtin("det", &[Value::Matrix(matrix)]).unwrap();

    if let Value::Num(det) = det_result {
        assert!((det - 3.0).abs() < 1e-10);
    } else {
        panic!("Expected numeric result");
    }
}

#[test]
fn test_error_handling() {
    // Test dimension mismatch errors
    let a = Matrix::new(vec![1.0, 2.0], 1, 2).unwrap();
    let b = Matrix::new(vec![1.0, 2.0, 3.0], 1, 3).unwrap();

    assert!(blas_matrix_mul(&a, &b).is_err());

    // Test non-square matrix errors
    let non_square = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    assert!(lapack_determinant(&non_square).is_err());
    assert!(lapack_matrix_inverse(&non_square).is_err());

    // Test vector length mismatch
    let vec_a = vec![1.0, 2.0];
    let vec_b = vec![1.0, 2.0, 3.0];
    assert!(blas_dot_product(&vec_a, &vec_b).is_err());
}
