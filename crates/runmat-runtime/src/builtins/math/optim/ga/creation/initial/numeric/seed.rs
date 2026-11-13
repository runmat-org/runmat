//! Seed construction retires before nonlinear feasibility is polled.
use super::*;

pub(super) struct Seed {
    pub draft: creation::Draft,
    pub changed: creation::ChangedRows,
    pub bitstring: bool,
}

pub(super) async fn prepare<F: creation::factory::Factory, I: Invocation>(
    problem: &Problem,
    options: &mut options::Options,
    factory: &F,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, ga::control::Retained<Seed>>> {
    let (initial, changed) = super::super::prepare(problem, options, cooperation)?;
    let bitstring = options.population == OptimGeneticPopulation::BitString;
    let mut draft = creation::Draft::new(
        initial,
        options.layout.total(),
        problem.variables,
        bitstring,
        cooperation,
    )
    .await?;
    if !draft.missing().is_empty() {
        match options.creation.kind() {
            Some(OptimGeneticCreation::UniformInteger) => {
                let integers = factory.integers(problem)?;
                creation::fill_uniform_integer(&mut draft, &options.ranges, integers, cooperation)
                    .await?;
            }
            Some(OptimGeneticCreation::Sobol) => {
                creation::fill_sobol(&mut draft, &options.ranges, cooperation).await?;
            }
            Some(
                OptimGeneticCreation::Uniform
                | OptimGeneticCreation::LinearFeasible
                | OptimGeneticCreation::NonlinearFeasible,
            ) => {
                draft
                    .fill_uniform(&options.ranges, bitstring, cooperation)
                    .await?;
            }
            None => {
                options.creation.bind(cooperation)?;
                let runtime = ga::controller::continuation::original(cooperation)?;
                let original_invoker = &mut *invoker;
                let original = &mut *cooperation;
                let original_problem = problem;
                let original_options = &*options;
                let created = ga::controller::continuation::run(&runtime, move || {
                    super::super::custom::numeric(
                        draft,
                        original_problem,
                        original_options,
                        bitstring,
                        original_invoker,
                        original,
                    )
                })?
                .await?;
                draft = match created {
                    ControlFlow::Continue(created) => created,
                    ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
                };
            }
        }
    }

    Ok(ControlFlow::Continue(ga::control::Retained::new(
        Seed {
            draft,
            changed,
            bitstring,
        },
        cooperation,
        ga::control::POLICY,
    )?))
}
