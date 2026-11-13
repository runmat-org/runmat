//! Intrinsic compiled CSC validation; numerical/domain policy stays with caller.
use super::*;
pub(crate) async fn validate_csc(
    matrix: &CscMatrix<f64>,
    cooperation: &mut ScalarCooperation,
    invalid: impl Fn(&str) -> RuntimeError,
) -> BuiltinResult<()> {
    if matrix.n.checked_add(1) != Some(matrix.colptr.len())
        || matrix.rowval.len() != matrix.nzval.len()
        || matrix.colptr.first() != Some(&0)
        || matrix.colptr.last() != Some(&matrix.nzval.len())
    {
        return Err(invalid("compiled CSC layout is inconsistent"));
    }
    for column in 0..matrix.n {
        let first = matrix.colptr[column];
        let end = matrix.colptr[column + 1];
        if first > end || end > matrix.nzval.len() {
            return Err(invalid("compiled CSC pointers are inconsistent"));
        }
        let mut previous = None;
        for index in first..end {
            let row = matrix.rowval[index];
            if row >= matrix.m
                || previous.is_some_and(|old| old >= row)
                || !matrix.nzval[index].is_finite()
            {
                return Err(invalid("compiled rows require ordered finite CSC entries"));
            }
            previous = Some(row);
            if index % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
