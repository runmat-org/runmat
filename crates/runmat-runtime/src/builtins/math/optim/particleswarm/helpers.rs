//! Genuine public creation/plot helpers with their documented grammars.
use super::{common, creation, inputs, options::Options, ScalarCooperation};
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[runtime_builtin(
    name = "pswcreationuniform",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::particleswarm::helpers"
)]
async fn uniform(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    uniform_once(runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
async fn uniform_once(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let invalid = |message: &str| helper_invalid(Helper::Creation, message);
    crate::context::host_allocation::require_unbounded_workspace(runtime,"pswcreationuniform",
        "finite host-memory execution is unsupported; initial swarm and random workspaces have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::PSWCREATIONUNIFORM_DESCRIPTOR,
        &runmat_builtins::PSWCREATIONUNIFORM_ERROR_INVALID_ARGUMENT,
        "pswcreationuniform",
        runtime,
    )?;
    if args.len() != 1 {
        return Err(invalid("pswcreationuniform requires one problem structure"));
    }
    let _original = args.fork();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity {
            name: "pswcreationuniform",
        },
        runtime,
    );
    let mut problem = inputs::parse_creation(args, &mut cooperation).await?;
    if problem.inconsistent {
        return Err(invalid("pswcreationuniform requires consistent bounds"));
    }
    // Only actual creation controls are active here; hybrid/observer/parallel
    // carriers and supplied Fval are not read or gathered by this helper.
    let mut selected = runmat_value::UnboundedStructValueBuilder::new();
    if let Value::Struct(options) = &problem.options {
        let fields = common::ScalarOptionFields::from_struct(Some(options));
        for field in [
            runmat_builtins::OptimOptionField::SwarmSize,
            runmat_builtins::OptimOptionField::InitialSwarmSpan,
            runmat_builtins::OptimOptionField::InitialPoints,
        ] {
            if let Some(value) = fields.exact(field) {
                let value = if field == runmat_builtins::OptimOptionField::InitialPoints {
                    if let Value::Struct(points) = value {
                        let x = points
                            .fields
                            .get("X")
                            .ok_or_else(|| invalid("InitialPoints requires X"))?;
                        let mut only_x = runmat_value::UnboundedStructValueBuilder::new();
                        only_x.insert("X", x.clone());
                        Value::Struct(only_x.finish())
                    } else {
                        value.clone()
                    }
                } else {
                    value.clone()
                };
                selected.insert(field.name(), value);
            }
        }
    } else if !matches!(&problem.options,Value::Tensor(v)if v.is_empty()) {
        return Err(invalid(
            "pswcreationuniform problem options must be a structure or []",
        ));
    }
    problem.options = Value::Struct(selected.finish());
    let options = Options::parse(problem.options, problem.variables, &mut cooperation).await?;
    let positions = creation::uniform_positions(
        options.initial.as_ref(),
        options.members,
        problem.variables,
        &options.span,
        &problem.lower,
        &problem.upper,
        &mut cooperation,
    )
    .await?;
    if crate::output_count::current_output_count_in(runtime) == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else {
        Ok(Value::Tensor(positions))
    }
}
#[runtime_builtin(
    name = "pswplotbestf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::particleswarm::helpers"
)]
async fn best_plot(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    best_plot_once(runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
async fn best_plot_once(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let invalid = |message: &str| helper_invalid(Helper::Plot, message);
    crate::context::host_allocation::require_unbounded_workspace(runtime,"pswplotbestf",
        "finite host-memory execution is unsupported; scene progress storage has no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::PSWPLOTBESTF_DESCRIPTOR,
        &runmat_builtins::PSWPLOTBESTF_ERROR_INVALID_ARGUMENT,
        "pswplotbestf",
        runtime,
    )?;
    if args.len() != 2 {
        return Err(invalid("pswplotbestf requires optimValues,state"));
    }
    plot_feature()?;
    let phase = crate::builtins::common::validation::value_text_scalar_string(&args[1])
        .and_then(|s| common::output_callback::OutputState::parse(&s))
        .ok_or_else(|| invalid("plot state must be init, iter, or done"))?;
    let Value::Struct(values) = &args[0] else {
        return Err(invalid("optimValues must be a scalar structure"));
    };
    let iteration = values
        .fields
        .get("iteration")
        .ok_or_else(|| invalid("optimValues.iteration is required"))?;
    let best = values
        .fields
        .get("bestfval")
        .ok_or_else(|| invalid("optimValues.bestfval is required"))?;
    for value in [iteration, best] {
        if !matches!(
            value,
            Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
        ) {
            return Err(invalid("plot metrics must be concrete real double scalars"));
        }
        let fact = crate::value_fact::value_fact(value);
        if !matches!(fact.kind,runmat_types::ValueKindFact::Numeric(n)if n.class==runmat_types::NumericClass::Double&&n.domain==runmat_types::NumericDomain::Real)
            || fact.storage == runmat_types::StorageFact::Sparse
            || fact.shape.element_count().is_some_and(|n| n != 1)
        {
            return Err(invalid("plot metrics must be full real double scalars"));
        }
        common::preflight_host_residency("pswplotbestf", value, None, runtime)?;
    }
    let read = |value: Value| match value {
        Value::Num(v) => Ok(v),
        Value::Tensor(v) if v.len() == 1 => Ok(v.as_f64_slice().expect("double metric")[0]),
        _ => Err(invalid("plot metric is not a scalar")),
    };
    let iteration =
        read(crate::dispatcher::gather_if_needed_in_context_async(iteration, runtime).await?)?;
    let best = read(crate::dispatcher::gather_if_needed_in_context_async(best, runtime).await?)?;
    if !iteration.is_finite() || iteration < 0.0 || iteration.fract() != 0.0 || !best.is_finite() {
        return Err(invalid(
            "plot iteration/bestfval must be finite and iteration nonnegative integral",
        ));
    }
    #[cfg(feature = "plot-core")]
    crate::builtins::plotting::state::update_optimization_plot(
        runmat_builtins::OptimPlotKind::ParticleBestFunctionValue,
        phase,
        Vec::new(),
        iteration,
        best,
        runtime,
    )?;
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (phase, iteration, best);
        return Err(invalid("pswplotbestf requires plot-core"));
    }
    #[cfg(feature = "plot-core")]
    if crate::output_count::current_output_count_in(runtime) == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else {
        Ok(Value::Bool(false))
    }
}
fn plot_feature() -> BuiltinResult<()> {
    #[cfg(feature = "plot-core")]
    {
        Ok(())
    }
    #[cfg(not(feature = "plot-core"))]
    {
        Err(helper_invalid(
            Helper::Plot,
            "pswplotbestf requires plot-core",
        ))
    }
}
enum Helper {
    Creation,
    Plot,
}
fn helper_invalid(role: Helper, message: &str) -> crate::RuntimeError {
    let (name, error) = match role {
        Helper::Creation => (
            "pswcreationuniform",
            &runmat_builtins::PSWCREATIONUNIFORM_ERROR_INVALID_ARGUMENT,
        ),
        Helper::Plot => (
            "pswplotbestf",
            &runmat_builtins::PSWPLOTBESTF_ERROR_INVALID_ARGUMENT,
        ),
    };
    crate::build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(error.identifier.expect("public helper identifier"))
        .build()
}
