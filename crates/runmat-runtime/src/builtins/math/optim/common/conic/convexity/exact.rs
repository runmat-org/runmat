//! One exact sparse Schur fallback for supplied and original compiled Hessians.
//! Arithmetic limbs use the caller's original cooperation, not library operators.
use super::super::super::{model_policy::Policy, proof_rational::Rational};
use super::*;
mod restriction;
pub(in crate::builtins::math::optim) use restriction::certify_restricted;
pub(in crate::builtins::math::optim) struct OriginalHessian {
    rows: Vec<BTreeMap<usize, Rational>>,
}
impl OriginalHessian {
    pub(in crate::builtins::math::optim) async fn new(
        dimension: usize,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut rows =
            runmat_value::exact_capacity_vec(dimension).map_err(|e| policy.storage(e))?;
        for _ in 0..dimension {
            rows.push(BTreeMap::new());
            cooperation.completed(1).await?;
        }
        Ok(Self { rows })
    }
    /// Adds an exact coefficient of the symmetric matrix's upper triangle.
    pub(in crate::builtins::math::optim) async fn add(
        &mut self,
        row: usize,
        column: usize,
        value: Rational,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        cooperation.completed(0).await?;
        if row > column || column >= self.rows.len() {
            return Err(policy.invalid("exact Hessian coordinate exceeds its upper triangle"));
        }
        let value = match self.rows[row].remove(&column) {
            Some(previous) => previous.add(&value, policy, cooperation).await?,
            None => value,
        };
        if !value.is_zero() {
            self.rows[row].insert(column, value);
        }
        cooperation.completed(1).await
    }
}
// The ordinary fallback previously had no caller model-error descriptor. Keep
// its owner and absence of a numerical-invalid identifier; model consumers pass
// their own policy to the same Schur arithmetic instead.
const INVALID: runmat_builtins::BuiltinErrorDescriptor = runmat_builtins::BuiltinErrorDescriptor {
    code: "InvalidConvexityEvidence",
    identifier: None,
    when: "exact Hessian certificate storage is inconsistent",
    message: "exact Hessian certificate storage is inconsistent",
};
pub(super) async fn certify(
    p: &Matrix,
    owner: runmat_builtins::BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let policy = Policy {
        identity: owner,
        invalid: &INVALID,
    };
    let mut original = OriginalHessian::new(p.n, policy, cooperation).await?;
    for col in 0..p.n {
        for entry in p.colptr[col]..p.colptr[col + 1] {
            original
                .add(
                    p.rowval[entry],
                    col,
                    Rational::from_float(p.nzval[entry])
                        .ok_or_else(|| policy.invalid("normalized Hessian is not finite"))?,
                    policy,
                    cooperation,
                )
                .await?;
        }
        cooperation.completed(1).await?;
    }
    certify_original(original, policy, cooperation).await
}
pub(in crate::builtins::math::optim) async fn certify_original(
    mut original: OriginalHessian,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for pivot in 0..original.rows.len() {
        let mut row = std::mem::take(&mut original.rows[pivot]);
        let diagonal = row.remove(&pivot).unwrap_or_else(Rational::zero);
        if diagonal.is_negative() {
            return Ok(false);
        }
        if diagonal.is_zero() {
            // add() and every update omit exact zeros.
            if !row.is_empty() {
                return Ok(false);
            }
            cooperation.completed(1).await?;
            continue;
        }
        let mut active =
            runmat_value::exact_capacity_vec(row.len()).map_err(|e| policy.storage(e))?;
        for entry in row {
            active.push(entry);
            cooperation.completed(1).await?;
        }
        for (i, (row, a)) in active.iter().enumerate() {
            let quotient = a.div(&diagonal, policy, cooperation).await?;
            for (column, b) in &active[i..] {
                let delta = quotient.mul(b, policy, cooperation).await?;
                let previous = original.rows[*row]
                    .remove(column)
                    .unwrap_or_else(Rational::zero);
                let value = previous.sub(&delta, policy, cooperation).await?;
                if !value.is_zero() {
                    original.rows[*row].insert(*column, value);
                }
                cooperation.completed(1).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(true)
}
