//! Public optimization plot callbacks; data is retained only by graphics owners.
use crate::builtins::math::optim::common::{output_callback::OutputState, ScalarCooperation};
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::OptimPlotKind;
use runmat_macros::runtime_builtin;
#[path = "optimplot/surrogate.rs"]
pub(crate) mod surrogate;
#[path = "optimplot/values.rs"]
mod values;
use runmat_value::Value;

#[runtime_builtin(
    name = "optimplotx",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::optimplot"
)]
async fn optimplotx_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(OptimPlotKind::CurrentPoint, runtime.as_ref(), args).await
}
#[runtime_builtin(
    name = "optimplotfunccount",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::optimplot"
)]
async fn optimplotfunccount_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(OptimPlotKind::FunctionCount, runtime.as_ref(), args).await
}
#[runtime_builtin(
    name = "optimplotfval",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::optimplot"
)]
async fn optimplotfval_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(OptimPlotKind::FunctionValue, runtime.as_ref(), args).await
}

#[runtime_builtin(
    name = "optimplotconstrviolation",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::optimplot"
)]
async fn optimplotconstrviolation_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(OptimPlotKind::ConstraintViolation, runtime.as_ref(), args).await
}

#[runtime_builtin(
    name = "optimplotstepsize",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::optimplot"
)]
async fn optimplotstepsize_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(OptimPlotKind::StepSize, runtime.as_ref(), args).await
}

#[runtime_builtin(
    name = "optimplotfirstorderopt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::optimplot"
)]
async fn optimplotfirstorderopt_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(OptimPlotKind::FirstOrderOptimality, runtime.as_ref(), args).await
}

#[runtime_builtin(
    name = "optimplotfvalconstr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::optimplot"
)]
async fn optimplotfvalconstr_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(
        OptimPlotKind::ConstrainedFunctionValue,
        runtime.as_ref(),
        args,
    )
    .await
}

async fn evaluate(
    kind: OptimPlotKind,
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    // This owner performs its own exact readbacks and may already have
    // committed a scene or notified a host. No error authorizes replay.
    evaluate_once(kind, runtime, args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
async fn evaluate_once(
    kind: OptimPlotKind,
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let name = kind.name();
    crate::context::host_allocation::require_unbounded_workspace(runtime,name,
        "finite host-memory execution is unsupported; progress history and graphics scene allocations have no complete resource bound")?;
    if args.len() != 3
        || crate::output_count::current_output_count_in(runtime).is_some_and(|count| count > 1)
    {
        return Err(invalid(
            kind,
            "expected (x,optimValues,state) and at most one stop output",
        ));
    }
    let phase = crate::builtins::common::validation::value_text_scalar_string(&args[2])
        .and_then(|state| OutputState::parse(&state))
        .ok_or_else(|| invalid(kind, "state must be init, iter, or done"))?;
    let Value::Struct(values) = &args[1] else {
        return Err(invalid(kind, "optimValues must be a scalar struct"));
    };
    let iteration_value = values
        .fields
        .get("iteration")
        .ok_or_else(|| invalid(kind, "optimValues.iteration is required"))?;
    let scalar_value = kind
        .value_field()
        .map(|field| {
            values
                .fields
                .get(field)
                .ok_or_else(|| invalid(kind, "the selected optimValues metric is required"))
        })
        .transpose()?;
    let violation = if kind == OptimPlotKind::ConstrainedFunctionValue {
        Some(
            values
                .fields
                .get("constrviolation")
                .ok_or_else(|| invalid(kind, "optimValues.constrviolation is required"))?,
        )
    } else {
        None
    };
    let tolerance = (kind == OptimPlotKind::ConstrainedFunctionValue)
        .then(|| values.fields.get("constraintTolerance"))
        .flatten();
    let feasible_count = (kind == OptimPlotKind::ConstrainedFunctionValue
        && scalar_value.is_some_and(empty))
    .then(|| values.fields.get("feasibleCount"))
    .flatten();
    // Admit exactly the values this chart will read before the first transfer.
    // Other optimValues fields, captures and the unused x of scalar-history
    // plots do not acquire a download or host-conversion side effect.
    for value in std::iter::once(iteration_value)
        .chain(scalar_value)
        .chain(violation)
        .chain(tolerance)
        .chain(feasible_count)
        .chain((kind == OptimPlotKind::CurrentPoint).then_some(&args[0]))
    {
        real_float(kind, value)?;
        if crate::builtins::common::residency::host_gather_intent(value)?
            == crate::builtins::common::residency::HostGatherIntent::ExplicitOrUnproven
        {
            return Err(invalid(kind,"host plot values require automatic residency; gather explicit or unproven device inputs first"));
        }
    }
    let mut cooperation =
        ScalarCooperation::for_builtin(runmat_builtins::BuiltinCatalogIdentity { name }, runtime);
    cooperation.completed(0).await?;
    let iteration = scalar(kind, &gather(iteration_value, runtime).await?)?;
    if runmat_types::decode_shape_dimension(iteration).is_err() {
        return Err(invalid(
            kind,
            "iteration must be a nonnegative representable integer",
        ));
    }
    let value = match kind {
        OptimPlotKind::CurrentPoint => None,
        OptimPlotKind::FunctionCount => {
            let value = scalar(
                kind,
                &gather(scalar_value.expect("count field"), runtime).await?,
            )?;
            if runmat_types::decode_shape_dimension(value).is_err() {
                return Err(invalid(
                    kind,
                    "funccount must be a nonnegative representable integer",
                ));
            }
            Some(value)
        }
        _ => {
            let value = gather(scalar_value.expect("function value field"), runtime).await?;
            if matches!(
                kind,
                OptimPlotKind::FunctionValue | OptimPlotKind::ConstrainedFunctionValue
            ) && empty(&value)
            {
                None
            } else {
                Some(scalar(kind, &value)?)
            }
        }
    };
    cooperation.completed(0).await?;
    let point = if kind == OptimPlotKind::CurrentPoint {
        match gather(&args[0], runtime).await? {
            Value::Num(value) => {
                if !value.is_finite() {
                    return Err(invalid(kind, "x coordinates must be finite"));
                }
                vec![value]
            }
            Value::Tensor(tensor) => {
                let mut point =
                    runmat_value::exact_capacity_vec(tensor.len()).map_err(|source| {
                        crate::build_runtime_error("optimization point allocation failed")
                            .with_builtin(name)
                            .with_source(source)
                            .build()
                    })?;
                for first in (0..tensor.len()).step_by(1024) {
                    let end = first.saturating_add(1024).min(tensor.len());
                    for index in first..end {
                        let value =
                            crate::builtins::common::tensor::tensor_value_f64(&tensor, index);
                        if !value.is_finite() {
                            return Err(invalid(kind, "x coordinates must be finite"));
                        }
                        point.push(value);
                    }
                    cooperation.completed((end - first) as u64).await?;
                }
                point
            }
            _ => {
                return Err(invalid(
                    kind,
                    "x must be a nonempty real host single/double array",
                ))
            }
        }
    } else {
        Vec::new()
    };
    let value = if kind == OptimPlotKind::CurrentPoint && !point.is_empty() {
        Some(0.0)
    } else {
        value
    };
    cooperation.completed(0).await?;
    if let Some(violation) = violation {
        let maximum = gather(violation, runtime).await?;
        let maximum = if empty(&maximum) {
            None
        } else {
            Some(scalar(kind, &maximum)?)
        };
        if maximum.is_some_and(|v| !v.is_finite() || tolerance.is_none() && v < 0.0) {
            return Err(invalid(
                kind,
                "constrviolation must be finite and nonnegative",
            ));
        }
        let explicit_tolerance = tolerance.is_some();
        let tolerance = match tolerance {
            Some(value) => scalar(kind, &gather(value, runtime).await?)?,
            None => 1e-6,
        };
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(invalid(
                kind,
                "constraintTolerance must be finite and nonnegative",
            ));
        }
        let feasible_count = match feasible_count {
            Some(value) => Some(
                runmat_types::decode_shape_dimension(scalar(kind, &gather(value, runtime).await?)?)
                    .map_err(|_| {
                        invalid(
                            kind,
                            "feasibleCount must be a nonnegative representable integer",
                        )
                    })?,
            ),
            None => None,
        };
        if let (Some(value), Some(maximum), false) = (value, maximum, explicit_tolerance) {
            super::state::update_constrained_optimization_plot(
                phase, iteration, value, maximum, runtime,
            )?;
        } else {
            super::state::update_constrained_metric(
                phase,
                iteration,
                super::state::ConstrainedPlotSample {
                    objective: value,
                    maximum,
                    tolerance,
                    feasible_count,
                },
                runtime,
            )?;
        }
    } else {
        super::state::update_optimization_plot(kind, phase, point, iteration, value, runtime)?;
    }
    cooperation.completed(0).await?;
    Ok(
        if crate::output_count::current_output_count_in(runtime) == Some(0) {
            Value::output_list(Vec::new())
        } else {
            Value::Bool(false)
        },
    )
}
fn empty(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => value.is_empty(),
        Value::GpuTensor(value) => value.shape().contains(&0),
        _ => false,
    }
}
async fn gather(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_in_context_async(value, runtime).await
}
fn real_float(kind: OptimPlotKind, value: &Value) -> BuiltinResult<()> {
    values::Owner::new(kind.name()).real(value)
}
fn scalar(kind: OptimPlotKind, value: &Value) -> BuiltinResult<f64> {
    values::Owner::new(kind.name()).scalar(value)
}
fn invalid(kind: OptimPlotKind, detail: &str) -> RuntimeError {
    values::Owner::new(kind.name()).invalid(detail)
}
#[cfg(test)]
#[path = "optimplot/tests.rs"]
mod tests;
