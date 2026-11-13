//! Original affine equation differences become residual rows, never hard rows.
use super::*;
use crate::builtins::math::optim::lsqlin::compiled;
pub(super) async fn assemble(
    program: &Program,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<compiled::Problem> {
    let objective::ModelObjective::Equations(evidence) = &program.objective else {
        return Err(policy.invalid("equation assembly requires original residual evidence"));
    };
    let mut entries = Vec::new();
    let mut d = runmat_value::exact_capacity_vec(evidence.rows).map_err(|e| policy.storage(e))?;
    for entry in &evidence.formulation.entries {
        let residual = program
            .array(entry.left, policy)?
            .clone()
            .add(
                program.array(entry.right, policy)?.clone(),
                runmat_types::OperatorKind::Subtract,
                entry.shape.clone(),
                program.layout.columns,
                policy,
                c,
            )
            .await?
            .map_err(|stop| stop.error(policy))?;
        let start = d.len();
        for row in 0..residual.count(policy)? {
            if !residual.assigned.at(row, policy)? {
                return Err(Stop::Unassigned.error(policy));
            }
            d.push(-residual.offset.at(row, policy)?);
            c.completed(1).await?;
        }
        entries
            .try_reserve(residual.coefficients.nonzeros())
            .map_err(|e| policy.reserve(e))?;
        for term in residual.coefficients.entries() {
            entries.push(terms::Term {
                row: start
                    .checked_add(term.row)
                    .ok_or_else(|| policy.invalid("equation residual row overflow"))?,
                ..term
            });
            c.completed(1).await?;
        }
    }
    if d.len() != evidence.rows {
        return Err(policy.invalid("equation assembly changed residual geometry"));
    }
    let matrix =
        super::super::assembly::matrix(entries, d.len(), program.layout.columns, policy, c)
            .await?
            .map_err(|stop| stop.error(policy))?;
    let empty = super::super::assembly::matrix(Vec::new(), 0, program.layout.columns, policy, c)
        .await?
        .map_err(|stop| stop.error(policy))?;
    let equality = super::super::assembly::matrix(Vec::new(), 0, program.layout.columns, policy, c)
        .await?
        .map_err(|stop| stop.error(policy))?;
    let bounds = layout::continuous_bound_values(&program.layout, policy, c).await?;
    Ok(compiled::Problem {
        c: matrix,
        d,
        weights: compiled::Weights::Uniform,
        a: empty,
        b: Vec::new(),
        aeq: equality,
        beq: Vec::new(),
        lb: bounds.lower,
        ub: bounds.upper,
    })
}
