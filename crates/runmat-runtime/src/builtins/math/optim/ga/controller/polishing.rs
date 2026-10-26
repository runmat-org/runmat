//! A local solver supplies a candidate and its own evaluation count. Only
//! original-domain validation can replace GA's independently retained best.
use super::super::common::callback_source::Callable;
use super::super::linear::Geometry;
use super::super::{
    assessment::{Assessment, Mode},
    common::population::{Fitness, Measurement},
    evaluation::{
        fitness::{Origin, RawFitness},
        generation::{Evaluated, Violation},
        nonlinear::Nonlinear,
    },
    invalid,
    population::Generation,
    ScalarCooperation, POLICY,
};
use super::search::Search;
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticPopulation;
use runmat_value::Value;

pub(super) struct Report {
    pub flag: f64,
    pub evaluations: usize,
    pub accepted: bool,
}
pub(super) async fn run_with<G: Geometry, I: super::super::callbacks::Invocation>(
    search: &mut Search<G>,
    invocation: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<std::ops::ControlFlow<I::Stop, Option<Report>>> {
    if search.hybrid.is_none() {
        return Ok(std::ops::ControlFlow::Continue(None));
    }
    let runtime = super::continuation::original(cooperation)?;
    super::continuation::run(&runtime, move || {
        run_selected(search, invocation, cooperation)
    })?
    .await
}
async fn run_selected<G: Geometry, I: super::super::callbacks::Invocation>(
    search: &mut Search<G>,
    invocation: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<std::ops::ControlFlow<I::Stop, Option<Report>>> {
    let Some(selected) = search.hybrid.as_ref() else {
        return Ok(std::ops::ControlFlow::Continue(None));
    };
    cooperation.completed(0).await?;
    if search.options.maximum.reached(search.timing.elapsed()) {
        return Ok(std::ops::ControlFlow::Continue(None));
    }
    let point = search
        .best
        .point
        .as_ref()
        .ok_or_else(|| invalid("hybrid invocation requires a retained best point"))?;
    let candidate = match selected
        .run_with(point, &search.problem, invocation, cooperation)
        .await?
    {
        std::ops::ControlFlow::Continue(candidate) => candidate,
        std::ops::ControlFlow::Break(stop) => return Ok(std::ops::ControlFlow::Break(stop)),
    };
    let mut report = Report {
        flag: candidate.flag,
        evaluations: candidate.evaluations,
        accepted: false,
    };
    cooperation.completed(0).await?;
    if search.options.maximum.reached(search.timing.elapsed()) {
        return Ok(std::ops::ControlFlow::Continue(Some(report)));
    }
    let population = Generation::from_result(
        candidate.point,
        1,
        search.problem.variables,
        OptimGeneticPopulation::DoubleVector,
        cooperation,
    )
    .await?;
    let linear_violation = if let Some(domain) = &mut search.domain {
        let Value::Tensor(point) = population.value() else {
            return Err(invalid("hybrid point lost its numeric storage"));
        };
        let membership = domain
            .classify(
                point.as_f64_slice().expect("admitted double point"),
                search.options.constraint_tolerance,
                cooperation,
            )
            .await?;
        if !membership.feasible {
            return Ok(std::ops::ControlFlow::Continue(Some(report)));
        }
        membership.violation
    } else {
        0.
    };
    // No hidden fitness replay: the hybrid's reported value remains distinct
    // provenance, while constraints are measured at the returned coordinate.
    let constraints = Nonlinear {
        function: search.sources.nonlinear.as_ref().map(Callable::Selected),
        vectorized: search.options.vectorized,
        parallel: search.nonlinear_parallel.as_ref(),
        counts: &mut search.nonlinear_counts,
        extents: &mut search.extents,
        mixed_integer: false,
        aliases: None,
        cooperation,
    }
    .evaluate(&population)
    .await?;
    let violation = constraints.row_violation(0, cooperation).await?;
    let mut violations = POLICY.capacity(1, cooperation.runtime())?;
    violations.push(
        violation.map_or(Violation::Undefined, |(maximum, sum)| Violation::Finite {
            maximum,
            sum,
            linear: linear_violation,
        }),
    );
    let mut values = POLICY.capacity(1, cooperation.runtime())?;
    values.push(Measurement::Real(candidate.fitness));
    let mut origins = POLICY.capacity(1, cooperation.runtime())?;
    origins.push(Origin::HybridReported);
    let evaluated = Evaluated {
        fitness: RawFitness {
            values: Fitness::from_measurements(POLICY, values, cooperation).await?,
            origins,
        },
        constraints,
        violations,
    };
    let assessment = Assessment::new(
        &evaluated,
        Mode::Raw,
        search.options.constraint_tolerance,
        cooperation,
    )
    .await?;
    report.accepted = match search
        .best
        .observe(
            &population,
            &evaluated,
            &assessment,
            search.progress.generation,
            invocation,
            cooperation,
        )
        .await?
    {
        std::ops::ControlFlow::Continue(accepted) => accepted,
        std::ops::ControlFlow::Break(stop) => return Ok(std::ops::ControlFlow::Break(stop)),
    };
    Ok(std::ops::ControlFlow::Continue(Some(report)))
}
