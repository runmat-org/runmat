//! Initial evaluation starts after population construction has retired.
use super::*;
pub(super) async fn complete<G: super::super::super::super::linear::Geometry, S: Source>(
    created: super::super::super::super::control::Retained<Created<G>>,
    source: &mut S,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<S::Stop, Start<G>>> {
    let Created {
        mut ready,
        timing,
        sources,
        initial,
        hybrid,
        objective_parallel,
        nonlinear_parallel,
    } = created.into_inner();
    let (problem, options) = ready.parts_mut();
    let reason = match initial.admission {
        creation::Admission::Complete => None,
        creation::Admission::Infeasible => Some(Termination::NoFeasiblePoint),
        creation::Admission::TimeLimit => Some(Termination::Time),
        creation::Admission::Unverified => Some(Termination::Unverified),
    };
    if let Some(reason) = reason {
        return Ok(ControlFlow::Continue(Start {
            ready: None,
            stopped: Some(InitialStop {
                reason,
                nonlinear_calls: initial.constraint_calls,
                nonlinear_points: initial.constraint_points,
            }),
        }));
    }
    let mut domain = initial.domain;
    let mut objective_counts = Counts::default();
    let mut nonlinear_counts = Counts {
        points: initial.constraint_points,
        callbacks: initial.constraint_calls,
    };
    let mut extents = initial.extents;
    let mut flags = Flags {
        eval_elites: true,
        duplicates: options.population != OptimGeneticPopulation::Custom,
    };
    let evaluated = {
        let maxima = linear::admit(
            &initial.generation,
            domain.as_mut(),
            options.constraint_tolerance,
            cooperation,
        )
        .await?;
        let aliases = Aliases::new(&initial.generation, flags.duplicates, cooperation).await?;
        if let Some(aliases) = &aliases {
            flags.duplicates = aliases.selected().len() < aliases.positions().len();
        }
        let mut supplied = supplied::initial(
            options,
            problem.nonlinear.is_some(),
            &initial.invalidated_scores,
            cooperation,
        )
        .await?;
        source
            .seed_initial(&initial.generation, &mut supplied, cooperation)
            .await?;
        let evaluated = Evaluator {
            objective: Callable::Selected(&sources.objective),
            nonlinear: sources.nonlinear.as_ref().map(Callable::Selected),
            objective_parallel: objective_parallel.as_ref(),
            nonlinear_parallel: nonlinear_parallel.as_ref(),
            vectorized: options.vectorized,
            check_values: options.check_values,
            mixed_integer: problem.mixed_integer,
            mode: evaluation_mode(problem, options),
            tolerance: options.constraint_tolerance,
            objective_counts: &mut objective_counts,
            nonlinear_counts: &mut nonlinear_counts,
            extents: &mut extents,
            aliases: aliases.as_ref(),
            cooperation,
        }
        .evaluate_with(&initial.generation, supplied, source)
        .await?;
        let mut evaluated = match evaluated {
            ControlFlow::Continue(evaluated) => evaluated,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        linear::combine(&mut evaluated, maxima, cooperation).await?;
        evaluated
    };
    source.refresh_counts(&mut objective_counts, &mut nonlinear_counts)?;
    let merit = if augmented(problem, options) {
        Some(initial_merit(&evaluated, options, cooperation).await?)
    } else {
        None
    };
    let assessment = Assessment::new(
        &evaluated,
        assessment_mode(problem, options, merit.as_ref()),
        options.constraint_tolerance,
        cooperation,
    )
    .await?;
    let mut best = Best::new();
    if let ControlFlow::Break(stop) = best
        .observe(
            &initial.generation,
            &evaluated,
            &assessment,
            0,
            source,
            cooperation,
        )
        .await?
    {
        return Ok(ControlFlow::Break(stop));
    }
    let mut progress = Progress::new(options, problem.variables, merit.is_some())?;
    progress
        .observed(&assessment, false, true, &timing, cooperation)
        .await?;
    let mutation = reproduction::Mutation::new(options, cooperation).await?;
    let merit_row = merit
        .as_ref()
        .map(|_| super::super::super::nonlinear::Row::new(&evaluated, cooperation))
        .transpose()?;
    let action =
        (merit.is_some() && best.point.is_none()).then_some(OptimGeneticAction::InfeasiblePoint);
    let (problem, options) = ready.take();
    drop(ready);
    Ok(ControlFlow::Continue(Start {
        ready: Some(Retained::new(
            Search {
                hybrid,
                problem,
                options,
                timing,
                generation: initial.generation,
                domain,
                evaluated,
                assessment,
                merit,
                objective_counts,
                nonlinear_counts,
                extents,
                best,
                progress,
                flags,
                supplied: initial.supplied,
                sources,
                action,
                objective_parallel,
                nonlinear_parallel,
                mutation,
                merit_row,
            },
            cooperation,
            super::super::super::super::control::POLICY,
        )?),
        stopped: None,
    }))
}
