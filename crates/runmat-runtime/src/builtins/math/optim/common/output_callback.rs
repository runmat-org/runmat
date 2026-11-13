//! Iteration observers use the ordinary semantic callback port and original
//! options generation. They are not objective evaluations or solver steps.
use super::{optim_error, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{NumericDType, Tensor, Value};

pub(crate) fn empty(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<Value> {
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    };
    let fail = |source| {
        host_allocation_runtime_error(
            runtime,
            "RunMat:HostAllocation:OptimizerCallback",
            "empty optimizer callback default",
            source,
        )
    };
    let bytes =
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[0, 0]).map_err(fail)?;
    let receipt =
        HostAllocationReceipt::admit(runtime, HostAllocationPlan::new(0, bytes).map_err(fail)?)
            .map_err(fail)?;
    if let (Some(runtime), Some(receipt)) = (runtime, receipt.as_ref()) {
        let output = receipt
            .exact_output_tensor(runtime, NumericDType::F64, &[0, 0])
            .map_err(fail)?;
        let output = output
            .into_tensor()
            .map_err(|source| fail(HostAllocationError::Numeric(source)))?;
        receipt.checkpoint(runtime).map_err(fail)?;
        Ok(Value::Tensor(output))
    } else {
        Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .map_err(|source| optim_error("optimoptions", source.to_string()))
    }
}

pub(crate) use runmat_builtins::OptimObserverState as OutputState;

pub(crate) fn validate(value: &Value) -> BuiltinResult<()> {
    validate_for(value, CallbackForm::Handles)
}

#[derive(Clone, Copy)]
pub(crate) enum CallbackForm {
    Handles,
    HandlesOrNames,
    HandlesOrPredefined(&'static [runmat_builtins::BuiltinCatalogIdentity]),
}

pub(crate) fn milp_form(field: runmat_builtins::OptimOptionField) -> CallbackForm {
    const OUTPUT: &[runmat_builtins::BuiltinCatalogIdentity] =
        &[runmat_builtins::BuiltinCatalogIdentity {
            name: runmat_builtins::MilpObserverKind::SaveSolutions.name(),
        }];
    const PLOT: &[runmat_builtins::BuiltinCatalogIdentity] =
        &[runmat_builtins::BuiltinCatalogIdentity {
            name: runmat_builtins::MilpObserverKind::PlotBounds.name(),
        }];
    CallbackForm::HandlesOrPredefined(if field == runmat_builtins::OptimOptionField::OutputFcn {
        OUTPUT
    } else {
        PLOT
    })
}

pub(crate) fn validate_for(value: &Value, form: CallbackForm) -> BuiltinResult<()> {
    validate_for_builtin("optimoptions", value, form)
}
pub(crate) fn validate_for_builtin(
    builtin: &str,
    value: &Value,
    form: CallbackForm,
) -> BuiltinResult<()> {
    let callable = |value: &Value| {
        matches!(
            value,
            Value::ScopedFunctionHandle(_)
                | Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::Closure(_)
        ) || crate::builtins::common::validation::value_text_scalar_string(value).is_some_and(
            |name| match form {
                CallbackForm::Handles => false,
                CallbackForm::HandlesOrNames => !name.trim().is_empty(),
                CallbackForm::HandlesOrPredefined(entries) => {
                    entries.iter().any(|entry| name == entry.name)
                }
            },
        )
    };
    let valid = match value {
        Value::Tensor(array) => array.is_empty(),
        Value::Cell(array) => array.data().iter().all(callable),
        value => callable(value),
    };
    if valid {
        Ok(())
    } else {
        Err(optim_error(
            builtin,
            match form {
                CallbackForm::Handles => {
                    "OutputFcn must be a function handle, cell array of function handles, or []"
                }
                CallbackForm::HandlesOrNames => {
                    "PlotFcns must be a function handle/name, cell array of handles/names, or []"
                }
                CallbackForm::HandlesOrPredefined(_) => "observer must be a function handle, supported predefined callback name, cell array of those callbacks, or []",
            },
        ))
    }
}

pub(crate) fn callback_count(value: Option<&Value>, form: CallbackForm) -> BuiltinResult<usize> {
    let Some(value) = value else { return Ok(0) };
    validate_for(value, form)?;
    Ok(match value {
        Value::Tensor(_) => 0,
        Value::Cell(value) => value.data().len(),
        _ => 1,
    })
}

pub(crate) struct ObserverGroup<'a> {
    pub value: Option<&'a Value>,
    pub form: CallbackForm,
    pub select: Option<&'a dyn Fn(usize) -> BuiltinResult<()>>,
}

/// One original metadata owner and argument generation across every observer.
/// A plot group's synchronous selector carries exact UI target authority only.
pub(crate) async fn invoke_groups(
    builtin: &'static str,
    groups: &[ObserverGroup<'_>],
    x: Value,
    metadata: crate::execution::RootedValue,
    state: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    metadata
        .with_owned_async(|metadata| async move {
            invoke_arguments(builtin, groups, &[x, metadata, state], cooperation).await
        })
        .await
}

/// Population solvers have a documented two-argument observer grammar. Keep
/// the exact same original semantic callback/root owner, not a three-arg alias.
pub(crate) async fn invoke_population_groups(
    builtin: &'static str,
    groups: &[ObserverGroup<'_>],
    metadata: crate::execution::RootedValue,
    state: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    metadata
        .with_owned_async(|metadata| async move {
            invoke_arguments(builtin, groups, &[metadata, state], cooperation).await
        })
        .await
}
async fn invoke_arguments(
    builtin: &'static str,
    groups: &[ObserverGroup<'_>],
    args: &[Value],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for group in groups {
        if let Some(value) = group.value {
            validate_for_builtin(builtin, value, group.form)?;
        }
    }
    let mut stop = false;
    for group in groups {
        let Some(value) = group.value else { continue };
        let callbacks = match value {
            Value::Tensor(_) => continue,
            Value::Cell(value) => value.data(),
            value => std::slice::from_ref(value),
        };
        for (index, callback) in callbacks.iter().enumerate() {
            cooperation.completed(0).await?;
            if let Some(select) = group.select {
                select(index)?;
            }
            let result =
                super::call_function_in_context(cooperation.runtime(), callback, args, 1).await?;
            cooperation.completed(0).await?;
            let requested = match result.value() {
                Value::Bool(value) => *value,
                Value::LogicalArray(value) if value.data.len() == 1 => value.data[0] != 0,
                _ => {
                    return Err(optim_error(
                        builtin,
                        match group.form {
                            CallbackForm::Handles => {
                                "OutputFcn must return a logical scalar stop flag"
                            }
                            CallbackForm::HandlesOrNames => {
                                "PlotFcns must return a logical scalar stop flag"
                            }
                            CallbackForm::HandlesOrPredefined(_) => {
                                "MILP observers must return a logical scalar stop flag"
                            }
                        },
                    ))
                }
            };
            stop |= requested;
        }
    }
    Ok(stop)
}

pub(crate) struct IterationValues<'a> {
    pub x: &'a [f64],
    pub shape: &'a [usize],
    pub scalar: bool,
    pub iteration: usize,
    pub func_count: usize,
    pub fval: f64,
    pub gradient: &'a [f64],
    pub first_order_optimality: f64,
    pub step_size: f64,
    pub search_direction: &'a [f64],
    pub line_step_length: Option<f64>,
    pub directional_derivative: Option<f64>,
}

pub(crate) async fn notify(
    callback: Option<&Value>,
    state: OutputState,
    values: IterationValues<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let Some(callback) = callback else {
        return Ok(false);
    };
    validate(callback)?;
    let callbacks: &[Value] = match callback {
        Value::Tensor(_) => return Ok(false),
        Value::Cell(array) => array.data(),
        value => std::slice::from_ref(value),
    };
    if callbacks.is_empty() {
        return Ok(false);
    }
    cooperation.completed(0).await?;
    let runtime = cooperation.runtime().cloned();
    let x = vector(values.x, values.shape, values.scalar, cooperation).await?;
    let gradient = vector(
        values.gradient,
        &[values.gradient.len(), 1],
        false,
        cooperation,
    )
    .await?;
    let direction_shape = if values.search_direction.is_empty() {
        [0, 0]
    } else {
        [values.search_direction.len(), 1]
    };
    let search_direction = vector(
        values.search_direction,
        &direction_shape,
        false,
        cooperation,
    )
    .await?;
    let line_step_length = match values.line_step_length {
        Some(value) => Value::Num(value),
        None => empty(runtime.as_ref())?,
    };
    let directional_derivative = match values.directional_derivative {
        Some(value) => Value::Num(value),
        None => empty(runtime.as_ref())?,
    };
    let procedure = crate::call::argument_text::copy_str("", runtime.as_ref())
        .map(|text| Value::String(text.into()))?;
    let fields = [
        ("iteration", Value::Num(values.iteration as f64)),
        ("funccount", Value::Num(values.func_count as f64)),
        ("fval", Value::Num(values.fval)),
        ("gradient", gradient),
        ("firstorderopt", Value::Num(values.first_order_optimality)),
        ("stepsize", Value::Num(values.step_size)),
        ("searchdirection", search_direction),
        ("lssteplength", line_step_length),
        ("directionalderivative", directional_derivative),
        ("procedure", procedure),
    ];
    let metadata = crate::execution::construct_struct_from_fields_in_optional_context(
        runtime.as_ref(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    let state = crate::call::argument_text::copy_str(state.name(), runtime.as_ref())
        .map(|text| Value::String(text.into()))?;
    let groups = [ObserverGroup {
        value: Some(callback),
        form: CallbackForm::Handles,
        select: None,
    }];
    invoke_groups("optim", &groups, x, metadata, state, cooperation).await
}

/// One original exact numeric output owner for observer argument copies.
pub(crate) async fn vector(
    values: &[f64],
    shape: &[usize],
    scalar: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    if scalar {
        return match values {
            [value] => Ok(Value::Num(*value)),
            _ => Err(optim_error(
                "optim",
                "scalar callback argument requires exactly one value",
            )),
        };
    }
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    };
    let runtime = cooperation.runtime().cloned();
    let fail = |source| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:HostAllocation:OptimizerCallback",
            "optimizer callback array",
            source,
        )
    };
    let bytes =
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, shape).map_err(fail)?;
    let receipt = HostAllocationReceipt::admit(
        runtime.as_ref(),
        HostAllocationPlan::new(0, bytes).map_err(fail)?,
    )
    .map_err(fail)?;
    if let (Some(runtime), Some(receipt)) = (runtime.as_ref(), receipt.as_ref()) {
        let mut output = receipt
            .exact_output_tensor(runtime, NumericDType::F64, shape)
            .map_err(fail)?;
        for (index, chunk) in values.chunks(1024).enumerate() {
            cooperation.completed(chunk.len() as u64).await?;
            receipt.checkpoint(runtime).map_err(fail)?;
            output.as_f64_slice_mut().expect("double output")
                [index * 1024..index * 1024 + chunk.len()]
                .copy_from_slice(chunk);
        }
        let output = output
            .into_tensor()
            .map_err(|source| fail(HostAllocationError::Numeric(source)))?;
        receipt.checkpoint(runtime).map_err(fail)?;
        Ok(Value::Tensor(output))
    } else {
        let mut output = runmat_value::exact_capacity_vec(values.len())
            .map_err(|source| fail(HostAllocationError::Numeric(source)))?;
        for chunk in values.chunks(1024) {
            output.extend_from_slice(chunk);
            cooperation.completed(chunk.len() as u64).await?;
        }
        Tensor::new(output, shape.to_vec())
            .map(Value::Tensor)
            .map_err(|source| optim_error("optim", source.to_string()))
    }
}
