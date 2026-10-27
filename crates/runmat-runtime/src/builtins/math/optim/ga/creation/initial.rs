//! Creation transfers the original geometry, never a cloned solver model.
use super::super::callbacks::{take_direct, Direct, Invocation};
use super::super::{
    common, inputs::Problem, invalid, linear, options, population::Generation, ScalarCooperation,
};
use crate::BuiltinResult;
use common::solver_timing::SolverTiming;
use runmat_builtins::{OptimGeneticCreation, OptimGeneticPopulation};
use std::ops::ControlFlow;

pub(in super::super) struct Initial<G = linear::Domain> {
    pub generation: Generation,
    pub domain: Option<G>,
    pub admission: super::Admission,
    pub constraint_calls: usize,
    pub constraint_points: usize,
    pub extents: Option<(usize, usize)>,
    pub supplied: usize,
    pub invalidated_scores: Vec<usize>,
}

pub(in super::super) async fn create(
    problem: &mut Problem,
    options: &mut options::Options,
    timing: &SolverTiming,
    sources: Option<&super::super::callbacks::Sources>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Initial> {
    create_with(
        problem,
        options,
        timing,
        sources,
        super::factory::Owned,
        &mut Direct,
        cooperation,
    )
    .await
    .map(take_direct)
}

mod custom;
mod numeric;
pub(in super::super) fn create_with<'a, F, I>(
    problem: &'a mut Problem,
    options: &'a mut options::Options,
    timing: &'a SolverTiming,
    sources: Option<&'a super::super::callbacks::Sources>,
    factory: F,
    invoker: &'a mut I,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<ControlFlow<I::Stop, Initial<F::Geometry>>>> + 'a
where
    F: super::factory::Factory + 'a,
    F::Geometry: 'a,
    I: Invocation + 'a,
{
    use futures::future::Either;
    // Only immutable typed branch selection happens before first poll. Both
    // selected children begin with the same original preparation below.
    if options.population == OptimGeneticPopulation::Custom {
        Either::Left(custom::create(
            problem,
            options,
            factory,
            invoker,
            cooperation,
        ))
    } else {
        Either::Right(numeric::create(
            problem,
            options,
            timing,
            sources,
            factory,
            invoker,
            cooperation,
        ))
    }
}
fn prepare(
    problem: &Problem,
    options: &options::Options,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<(Option<runmat_value::Value>, super::ChangedRows)> {
    let changed = super::ChangedRows::new(score_prefix(problem, options)?, cooperation)?;
    let fields = common::ScalarOptionFields::from_struct_for_solver(
        Some(options.prepared.structure()),
        runmat_builtins::OptimSolver::Ga,
    );
    let initial = fields
        .exact(runmat_builtins::OptimOptionField::InitialPopulationMatrix)
        .cloned();
    Ok((initial, changed))
}

fn score_prefix(problem: &Problem, options: &options::Options) -> BuiltinResult<usize> {
    if problem.nonlinear.is_some() {
        return Ok(0);
    }
    let count = match options.field(runmat_builtins::OptimOptionField::InitialScoresMatrix) {
        None => 0,
        Some(runmat_value::Value::Num(_) | runmat_value::Value::Int(_)) => 1,
        Some(runmat_value::Value::Tensor(value)) => value.len(),
        _ => {
            return Err(invalid(
                "InitialScoresMatrix must be an admitted real vector",
            ))
        }
    };
    if count > options.layout.total() {
        return Err(invalid("InitialScoresMatrix exceeds PopulationSize"));
    }
    Ok(count)
}
