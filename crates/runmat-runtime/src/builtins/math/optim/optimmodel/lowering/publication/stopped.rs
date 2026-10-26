//! Original evaluated-only LSQ values; no feasible-point or certificate upgrade.
use super::*;
pub(super) async fn publish(
    mut solved: run::AlgebraicStopped,
    requested: Option<usize>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    let count = requested.unwrap_or(1);
    if count > 5 {
        return Err(policy.invalid("solve returns at most five outputs"));
    }
    let flag = solved.outcome.flag(policy)?;
    let mut roots = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    if count > 0 {
        roots.push(RootedValue::in_optional_context(
            solved.candidate.solution().value().clone(),
            cooperation.runtime(),
        )?);
    }
    if count > 1 {
        roots.push(match solved.candidate.objective() {
            Some(value) => {
                RootedValue::in_optional_context(Value::Num(value), cooperation.runtime())?
            }
            None => empty(policy, cooperation)?,
        });
    }
    if count > 2 {
        roots.push(super::super::super::exitflag::publish(
            flag,
            policy,
            cooperation.runtime(),
        )?);
    }
    if count > 3 {
        roots.push(information::publish(&mut solved.outcome, policy, cooperation).await?);
    }
    if count > 4 {
        roots.push(
            multipliers::project(
                &solved.program.layout,
                &solved.formulation,
                &solved.rows,
                None,
                policy,
                cooperation,
            )
            .await?,
        );
    }
    finish(roots, requested, policy, cooperation).await
}
