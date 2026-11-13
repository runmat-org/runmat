//! Exact supplied-H restriction to the original equality null space.
use super::*;
use crate::builtins::math::optim::common::{
    conic::convexity::{certify_restricted, OriginalHessian},
    model_policy::Policy,
    proof_rational::Rational,
};
use std::collections::BTreeMap;
pub(super) const POLICY: Policy = Policy {
    identity: runmat_builtins::QUADPROG_CATALOG_ENTRY.identity,
    invalid: &runmat_builtins::QUADPROG_ERROR_INVALID_INPUT,
};
pub(super) async fn certify(model: &Model, c: &mut ScalarCooperation) -> BuiltinResult<bool> {
    if model.eq == 0 {
        return psd::certify(&model.p, c).await;
    }
    let mut original = OriginalHessian::new(model.f.len(), POLICY, c).await?;
    let mut equations = capacity(model.eq)?;
    for _ in 0..model.eq {
        equations.push(BTreeMap::new());
        c.completed(1).await?;
    }
    for column in 0..model.f.len() {
        for entry in model.p.colptr[column]..model.p.colptr[column + 1] {
            original
                .add(
                    model.p.rowval[entry],
                    column,
                    Rational::from_float(model.p.nzval[entry])
                        .ok_or_else(|| invalid("nonfinite original Hessian"))?,
                    POLICY,
                    c,
                )
                .await?;
        }
        for entry in model.a.colptr[column]..model.a.colptr[column + 1] {
            let row = model.a.rowval[entry];
            if row < model.eq && model.a.nzval[entry] != 0. {
                equations[row].insert(
                    column,
                    Rational::from_float(model.a.nzval[entry])
                        .ok_or_else(|| invalid("nonfinite original equality"))?,
                );
            }
            c.completed(1).await?;
        }
    }
    certify_restricted(original, equations, POLICY, c).await
}
