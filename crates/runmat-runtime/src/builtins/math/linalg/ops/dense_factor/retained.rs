//! Immutable canonical dense factors; factor and solve scratch are invocation-local.
mod factor;
mod solve;
mod storage;
use super::*;
pub(in crate::builtins::math::linalg) use factor::{factor_dense, DenseFactorOptions};
use runmat_value::{
    NativeDenseFactorMatrix, NumericFactorCandidate, NumericFactorData, NumericFactorKind,
    NumericFactorValidation,
};
#[cfg(test)]
pub(in crate::builtins::math::linalg) use solve::solve_dense;
pub(in crate::builtins::math::linalg) use solve::solve_dense_indexed;
pub(in crate::builtins::math::linalg) use storage::DenseScalar;

pub(super) async fn validate(
    candidate: NumericFactorCandidate,
    work: &mut work::Work<'_>,
) -> Result<NumericFactorData, DenseFactorError> {
    let mut proof = NumericFactorValidation::new(candidate)
        .map_err(|s| DenseFactorError::InvalidData(s.into()))?;
    loop {
        let done = proof
            .advance(std::num::NonZeroUsize::new(1).expect("one"))
            .map_err(DenseFactorError::InvalidData)?;
        work.tick().await?;
        if done {
            break;
        }
    }
    work.flush().await?;
    proof
        .finish()
        .map_err(|s| DenseFactorError::InvalidData(s.into()))
}
