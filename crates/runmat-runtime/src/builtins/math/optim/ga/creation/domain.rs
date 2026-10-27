//! One original geometry acquisition; ordinary and borrowed creation share
//! the same domain-admission phase without another model or sparse copy.
use super::super::common::solver_timing::SolverTiming;
use super::super::{inputs::Problem, linear, options, ScalarCooperation};
use crate::BuiltinResult;

pub(super) async fn acquire(
    problem: &mut Problem,
    options: &options::Options,
    timing: &SolverTiming,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<linear::Domain> {
    let data = problem
        .linear
        .take()
        .ok_or_else(|| super::super::invalid("creation geometry was already consumed"))?;
    let input = data.into_repair(cooperation).await?;
    let model = super::super::super::intlinprog::feasibility::Domain::new(
        input,
        timing.clone(),
        options.maximum,
        super::super::numeric::linear_tolerance(options.constraint_tolerance),
        cooperation,
    )
    .await?;
    let domain = linear::Domain::new(model, cooperation).await?;
    Ok(domain)
}

pub(super) async fn prepare<F: super::factory::Factory>(
    factory: F,
    draft: &mut super::Draft,
    problem: &mut Problem,
    options: &options::Options,
    timing: &SolverTiming,
    admission: &mut super::Admission,
    changed: &mut super::ChangedRows,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<F::Geometry> {
    let mut domain = factory.build(problem, options, timing, cooperation).await?;
    if !matches!(*admission, super::Admission::TimeLimit) {
        *admission = super::admit(
            draft,
            &mut domain,
            timing,
            options.maximum,
            changed,
            cooperation,
        )
        .await?;
    }
    Ok(domain)
}
