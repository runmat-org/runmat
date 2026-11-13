//! Public genetic plot callbacks return their caller's state unchanged.
use super::super::invalid;
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_builtins::{OptimGeneticObserverState as Phase, OptimGeneticPlotKind as Kind};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

async fn execute(
    kind: Kind,
    runtime: Option<&RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() != 3 {
        return Err(invalid("genetic plot callbacks require options,state,flag"));
    }
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::GA_PLOT_HELPER_DESCRIPTOR,
        &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
        kind.name(),
        runtime,
    )?;
    let _original = args.fork();
    let phase = crate::builtins::common::validation::value_text_scalar_string(&args[2])
        .and_then(|value| Phase::parse(&value))
        .ok_or_else(|| invalid("genetic plot flag must be init,iter,interrupt,done"))?;
    if !matches!(&args[1], Value::Struct(_)) {
        return Err(invalid("genetic plot state must be a scalar structure"));
    }
    super::admission::options(&args[0])?;
    let mut cooperation = super::admission::begin(kind.name(), runtime)?;
    cooperation.completed(0).await?;
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = phase;
        return Err(
            crate::build_runtime_error("genetic plots require plot-core")
                .with_identifier("RunMat:graphics:PlotCoreUnavailable")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        );
    }
    #[cfg(feature = "plot-core")]
    {
        if phase != Phase::Interrupt {
            let state = super::plot_state::State::new(&args[1])?;
            state.preflight(kind, &cooperation)?;
            if kind == Kind::Stopping {
                super::plot_state::stopping_options(&args[0], &cooperation)?;
            }
            if kind == Kind::MaxConstraint {
                let fields = super::super::common::ScalarOptionFields::from_struct_for_solver(
                    if let Value::Struct(value) = &args[0] {
                        Some(value)
                    } else {
                        None
                    },
                    runmat_builtins::OptimSolver::Ga,
                );
                let algorithm = fields
                    .exact(runmat_builtins::OptimOptionField::NonlinearConstraintAlgorithm)
                    .map(|value| {
                        super::super::options::text(value).and_then(|text| {
                            runmat_builtins::OptimGeneticConstraint::parse(&text)
                                .ok_or_else(|| invalid("unrecognized NonlinearConstraintAlgorithm"))
                        })
                    })
                    .transpose()?
                    .unwrap_or(runmat_builtins::OptimGeneticConstraint::AugmentedLagrangian);
                if algorithm != runmat_builtins::OptimGeneticConstraint::AugmentedLagrangian {
                    return Err(invalid(
                        "gaplotmaxconstr requires auglag nonlinear constraints",
                    ));
                }
            }
            let charts =
                super::plot_data::prepare(kind, &state, &args[0], &mut cooperation).await?;
            for (series, data) in charts {
                crate::builtins::plotting::state::update_genetic_plot(
                    kind, series, phase, data, runtime,
                )?;
                cooperation.completed(1).await?;
            }
        }
        if crate::output_count::current_output_count_in(runtime) == Some(0) {
            Ok(Value::output_list(Vec::new()))
        } else {
            Ok(args[1].clone())
        }
    }
}

#[runtime_builtin(
    name = "gaplotbestf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotbestf(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::BestFitness, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotbestindiv",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotbestindiv(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::BestIndividual, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotdistance",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotdistance(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Distance, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotexpectation",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotexpectation(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Expectation, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotgenealogy",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotgenealogy(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Genealogy, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotmaxconstr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotmaxconstr(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::MaxConstraint, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotrange",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotrange(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Range, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotscorediversity",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotscorediversity(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::ScoreDiversity, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotscores",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotscores(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Scores, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotselection",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotselection(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Selection, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "gaplotstopping",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::plot"
)]
async fn gaplotstopping(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Stopping, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
