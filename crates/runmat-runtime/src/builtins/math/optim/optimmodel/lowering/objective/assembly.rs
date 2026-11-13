//! Final residual CSC assembly, never normal equations or square-root weights.
use super::*;
use crate::builtins::math::optim::lsqlin::compiled as lsq;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn assemble(
    program: &Program,
    assembled: super::super::assembly::Assembled,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<lsq::Problem> {
    if !matches!(assembled.cost, super::super::assembly::Cost::Residual) {
        return Err(policy.invalid("residual solve received a linear-cost assembly"));
    }
    let objective = program
        .objective
        .squares()
        .ok_or_else(|| policy.invalid("original residual objective is unavailable"))?;
    let (mut entries, mut d, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    for term in &objective.terms {
        let array = program.array(term.residual, policy)?;
        let count = array.count(policy)?;
        let start = d.len();
        d.try_reserve(count).map_err(|e| policy.reserve(e))?;
        weights.try_reserve(count).map_err(|e| policy.reserve(e))?;
        for row in 0..count {
            if !array.assigned.at(row, policy)? {
                return Ok(Err(Stop::Unassigned));
            }
            d.push(-array.offset.at(row, policy)?);
            weights.push(term.weight.proposal);
            c.completed(1).await?;
        }
        entries
            .try_reserve(array.coefficients.nonzeros())
            .map_err(|e| policy.reserve(e))?;
        for entry in array.coefficients.entries() {
            entries.push(terms::Term {
                row: entry
                    .row
                    .checked_add(start)
                    .ok_or_else(|| policy.invalid("residual row overflow"))?,
                ..entry
            });
            c.completed(1).await?;
        }
    }
    let matrix =
        match super::super::assembly::matrix(entries, d.len(), program.layout.columns, policy, c)
            .await?
        {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
    Ok(Ok(lsq::Problem {
        c: matrix,
        d,
        weights: lsq::Weights::PerRow(weights),
        a: assembled.a_ineq,
        b: assembled.b_ineq,
        aeq: assembled.a_eq,
        beq: assembled.b_eq,
        lb: assembled.lb,
        ub: assembled.ub,
    }))
}
