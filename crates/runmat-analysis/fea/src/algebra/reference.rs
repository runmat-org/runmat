use super::RealSparseMatrix;

pub fn spmv(matrix: &RealSparseMatrix, x: &[f64]) -> Result<Vec<f64>, &'static str> {
    if x.len() != matrix.columns() {
        return Err("sparse matrix/vector dimension mismatch");
    }
    let mut result = vec![0.0; matrix.rows()];
    for (row, result_value) in result.iter_mut().enumerate() {
        *result_value = matrix
            .pattern()
            .row_range(row)
            .map(|entry| matrix.values()[entry] * x[matrix.pattern().column(entry)])
            .sum();
    }
    Ok(result)
}

pub fn residual(
    matrix: &RealSparseMatrix,
    x: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if rhs.len() != matrix.rows() {
        return Err("sparse residual dimension mismatch");
    }
    Ok(rhs
        .iter()
        .zip(spmv(matrix, x)?)
        .map(|(right, applied)| right - applied)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseMatrix};

    #[test]
    fn identity_application_and_residual_are_deterministic() {
        let identity = RealSparseMatrix::diagonal(
            vec![1.0, 1.0, 1.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let x = vec![3.0, -2.0, 7.0];
        assert_eq!(spmv(&identity, &x).unwrap(), x);
        assert_eq!(residual(&identity, &x, &x).unwrap(), vec![0.0; 3]);
    }
}
