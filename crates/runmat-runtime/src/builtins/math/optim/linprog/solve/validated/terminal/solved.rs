//! Original selected status arithmetic and callback ordering.
use super::*;
use crate::builtins::math::optim::linprog::compiled;
pub(super) async fn run<V: compiled::Validation>(
    prepared: Prepared<'_, V>,
) -> BuiltinResult<LinprogOutcome> {
    let Prepared {
        model,
        mut point,
        cooperation,
        controls,
        products,
        validation,
    } = prepared;
    let mut z = std::mem::take(&mut point.z);
    for chunk in z.chunks_mut(1024) {
        for value in chunk.iter_mut() {
            *value *= model.objective_scale;
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    let checked = certificates::solution(&model, &point.x, &z, cooperation).await?;
    let domain_valid = if checked.as_ref().is_some_and(|checked| checked.valid) {
        validation
            .solution(&point.x, &model.roles, &z, cooperation)
            .await?
    } else {
        false
    };
    if let Some(checked) = checked.filter(|checked| checked.valid && domain_valid) {
        let lambda = if products == compiled::Products::WithMultipliers {
            Some(model::Multipliers::from_dual(&model, &z, cooperation).await?)
        } else {
            None
        };
        let x = std::mem::take(&mut point.x);
        drop(point);
        Ok(LinprogOutcome {
            x: Some(x),
            fval: Some(checked.point.objective),
            lambda,
            exitflag: 1,
            iterations: controls.iterations(),
            constrviolation: checked.point.violation,
            primalfeasibility: checked.point.normalized,
            dualfeasibility: checked.dual_residual,
            dualitygap: checked.gap,
            firstorderopt: checked.raw_stationarity,
            message:
                "Optimal solution found; original primal, dual and complementarity checks passed."
                    .into(),
        })
    } else {
        super::super::incumbent(
            &model,
            std::mem::take(&mut point.x),
            -4,
            controls.iterations(),
            "Backend result did not satisfy original primal/dual optimality checks.",
            cooperation,
            validation,
        )
        .await
    }
}
