//! Validate a generation before any fitness/constraint callback can run.
use super::super::evaluation::generation::{Evaluated, Violation};
use super::super::{invalid, linear::Geometry, population::Generation, ScalarCooperation, POLICY};
use crate::BuiltinResult;

pub(super) async fn admit<G: Geometry>(
    generation: &Generation,
    domain: Option<&mut G>,
    tolerance: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let Some(domain) = domain else {
        return POLICY.capacity(0, cooperation.runtime());
    };
    let population = generation
        .numeric()
        .ok_or_else(|| invalid("linear admission requires a numeric generation"))?;
    let mut maxima = POLICY.capacity(population.members, cooperation.runtime())?;
    for row in 0..population.members {
        let point = super::super::population::coordinates(population, row, cooperation).await?;
        let membership = domain.classify(&point, tolerance, cooperation).await?;
        if !membership.feasible {
            return Err(invalid(
                "population violates its original linear, bound or integer domain",
            ));
        }
        maxima.push(membership.violation);
        cooperation.completed(1).await?;
    }
    Ok(maxima)
}
pub(super) async fn combine(
    evaluated: &mut Evaluated,
    maxima: Vec<f64>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if maxima.is_empty() {
        return Ok(());
    }
    if maxima.len() != evaluated.violations.len() {
        return Err(invalid("linear observation extent changed"));
    }
    for (violation, linear) in evaluated.violations.iter_mut().zip(maxima) {
        if let Violation::Finite {
            linear: recorded, ..
        } = violation
        {
            *recorded = linear;
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
