pub(in crate::builtins::math::optim) mod cubic_rbf;
pub(in crate::builtins::math::optim) mod nelder_mead;
use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_builtins::{OptimOptionField, OptimSolver};
use runmat_value::{StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

pub(crate) mod conic;
pub(super) mod constrained_numeric;
pub(in crate::builtins::math::optim) mod feasible_cone;
pub(in crate::builtins::math::optim) mod model_dimensions;
pub(in crate::builtins::math::optim) mod model_indexing;
pub(in crate::builtins::math::optim) mod model_policy;
pub(in crate::builtins::math::optim) mod model_storage;
mod scalar_cooperation;
pub(crate) mod trust_quadratic;
pub(crate) use scalar_cooperation::ScalarCooperation;
mod host_arrays;
pub(crate) use host_arrays::{first_nonfinite, logical_values, tensor_values};
pub(crate) mod affine_coordinate;
pub(crate) mod anneal_options;
pub(crate) mod callback_source;
pub(crate) mod derivative_scaling;
pub(crate) mod global_hybrid;
pub(crate) mod global_time;
pub(crate) mod output_callback;
pub(crate) mod population;
pub(crate) mod quasi_newton_controls;
pub(crate) mod retained_control;
mod scalar_options;
pub(crate) mod solver_timing;
pub(crate) mod surrogate_options;
pub(crate) mod swarm_options;
pub(crate) use scalar_options::{
    prepare_fmincon_options, prepare_modeled_fminunc_options, prepare_modeled_residual_options,
    prepare_multiobjective_options, prepare_solver_options, switch_value, PreparedSolverOptions,
    ScalarOptionFields,
};

mod catalog_preflight;
pub(crate) use catalog_preflight::{preflight_catalog_facts, preflight_catalog_inputs};

pub(crate) fn optim_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(crate) fn preflight_real_array(
    name: &str,
    value: &Value,
    role: &str,
    empty: bool,
) -> BuiltinResult<()> {
    if !runmat_builtins::nonlinear_real_numeric_metadata(
        &crate::value_fact::value_fact(value),
        empty,
    ) {
        return Err(optim_error(
            name,
            format!("{name}: {role} must be a supported real numeric array"),
        ));
    }
    Ok(())
}

fn canonicalize_callback_handle(handle: &Value) -> BuiltinResult<Value> {
    crate::canonicalize_callback_handle_for_semantic_resolution(handle.clone())
}

pub(crate) async fn call_function(
    handle: &Value,
    args: impl AsRef<[Value]>,
) -> BuiltinResult<crate::execution::RootedValue> {
    call_function_in_context(None, handle, args, 1).await
}

/// Resolve under the original invocation synchronously, then dispatch through
/// its explicit call port. No ambient entry guard survives an await.
pub(crate) async fn call_function_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &Value,
    args: impl AsRef<[Value]>,
    requested_outputs: usize,
) -> BuiltinResult<crate::execution::RootedValue> {
    let result =
        call_function_sequence_in_context(runtime, handle, args, requested_outputs).await?;
    if requested_outputs == 1 {
        let _entry = runtime.map(crate::context::RuntimeContext::enter);
        result.try_into_single().map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
    } else {
        crate::call::arguments::project_rooted_legacy_builtin_value_abi(result)
    }
}

/// Canonical callback results preserve roots and comma-separated cardinality.
/// Consumers decoding several values can retain the rooted stream rather than
/// projecting it into the older multi-output Value carrier.
pub(crate) async fn call_function_sequence_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &Value,
    args: impl AsRef<[Value]>,
    requested_outputs: usize,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let callback = {
        let _entry = runtime.map(crate::context::RuntimeContext::enter);
        canonicalize_callback_handle(handle)?
    };
    match runtime {
        Some(runtime) => {
            crate::call_feval_sequence_async_with_outputs_in_context(
                runtime,
                callback,
                args.as_ref(),
                requested_outputs,
            )
            .await
        }
        None => {
            crate::call_feval_sequence_async_with_outputs(
                callback,
                args.as_ref(),
                requested_outputs,
            )
            .await
        }
    }
}

pub(crate) async fn call_scalar_function(
    name: &str,
    solver: OptimSolver,
    handle: &Value,
    x: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    call_scalar_function_with_precision(name, solver, handle, x, false, cooperation).await
}

pub(crate) async fn call_scalar_function_with_precision(
    name: &str,
    solver: OptimSolver,
    handle: &Value,
    x: f64,
    single: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    call_scalar_function_with_precision_info(name, solver, handle, x, single, cooperation)
        .await
        .map(|(value, _)| value)
}

pub(crate) async fn call_scalar_function_with_precision_info(
    name: &str,
    solver: OptimSolver,
    handle: &Value,
    x: f64,
    single: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(f64, bool)> {
    let argument = if single {
        Value::Tensor(
            Tensor::new_with_dtype(vec![x], vec![1, 1], runmat_value::NumericDType::F32)
                .map_err(|error| optim_error(name, format!("{name}: {error}")))?,
        )
    } else {
        Value::Num(x)
    };
    let value = call_function_in_context(cooperation.runtime(), handle, [argument], 1).await?;
    let output_single = match value.value() {
        Value::Tensor(tensor) => tensor.numeric_dtype() == runmat_value::NumericDType::F32,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    };
    value
        .with_owned_async(|value| async move {
            let value = match solver {
                OptimSolver::Fminbnd => {
                    prepare_floating_value_with_cooperation(
                        name,
                        value,
                        &runmat_builtins::FMINBND_CALLBACK_NUMERIC_EXTENSION,
                        &runmat_builtins::FMINBND_RESIDENT_EXTENSION,
                        "objective value",
                        cooperation,
                    )
                    .await?
                }
                OptimSolver::Fzero => {
                    prepare_floating_value_with_cooperation(
                        name,
                        value,
                        &runmat_builtins::FZERO_CALLBACK_NUMERIC_EXTENSION,
                        &runmat_builtins::FZERO_RESIDENT_EXTENSION,
                        "function value",
                        cooperation,
                    )
                    .await?
                }
                _ => {
                    return Err(optim_error(
                        name,
                        "unsupported scalar solver callback identity",
                    ))
                }
            };
            value_to_scalar(name, value).map(|value| (value, output_single))
        })
        .await
}

pub(crate) fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(crate) fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

#[derive(Clone, Copy)]
pub(crate) enum ScalarOperandGeometry {
    Scalar,
    ScalarOrPair,
}

impl ScalarOperandGeometry {
    fn accepts(self, count: usize) -> bool {
        match self {
            Self::Scalar => count == 1,
            Self::ScalarOrPair => count == 1 || count == 2,
        }
    }
}

/// Checks host-visible operand facts before the first provider operation. It
/// deliberately does not inspect resident numerical payloads.
pub(crate) fn preflight_host_residency(
    name: &str,
    value: &Value,
    explicit_extension: Option<&BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    use runmat_accelerate_api::GpuHandleProvenance;
    let detail = match runmat_accelerate_api::handle_provenance(handle) {
        Some(GpuHandleProvenance::Automatic) => return Ok(()),
        Some(GpuHandleProvenance::Explicit) => {
            if let Some(extension) = explicit_extension {
                return crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                    runtime, extension, name,
                );
            }
            "explicit GPU operands require an explicit gather before this host solver"
        }
        None => "device operands with unproven placement require an explicit gather",
    };
    let mut error = optim_error(name, format!("{name}: {detail}"));
    error.gpu_gather_retry = GpuGatherRetry::Never;
    Err(error)
}

pub(crate) fn preflight_scalar_operand(
    solver: OptimSolver,
    value: &Value,
    geometry: ScalarOperandGeometry,
    role: &str,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let name = solver.name();
    preflight_host_residency(name, value, Some(resident_extension), runtime)?;
    if is_typed_integer_value(value) || is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            numeric_extension,
            name,
        )?;
    }
    let invalid = |detail: &str| {
        build_runtime_error(format!("{name}: {role} {detail}"))
            .with_builtin(name)
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build()
    };
    match value {
        Value::Num(number) if geometry.accepts(1) && number.is_finite() => Ok(()),
        Value::Int(_) if geometry.accepts(1) => {
            if crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
                Ok(())
            } else {
                Err(invalid("must be exactly representable as double"))
            }
        }
        Value::Bool(_) if geometry.accepts(1) => Ok(()),
        Value::LogicalArray(array) if geometry.accepts(array.data.len()) => Ok(()),
        Value::Tensor(tensor) if geometry.accepts(tensor.len()) => {
            if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
                Err(invalid("must be exactly representable as double"))
            } else if (0..tensor.len())
                .all(|index| tensor::tensor_value_f64(tensor, index).is_finite())
            {
                Ok(())
            } else {
                Err(invalid("must be finite"))
            }
        }
        Value::GpuTensor(handle) => {
            let count = handle
                .shape()
                .iter()
                .try_fold(1_usize, |total, axis| total.checked_mul(*axis));
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
                || !count.is_some_and(|count| geometry.accepts(count))
            {
                return Err(invalid(
                    "must be a real scalar or admitted two-element interval",
                ));
            }
            handle.allocation_lease().validate_live().map_err(|error| {
                build_runtime_error(format!(
                    "{name}: {role} has an invalid resident lease: {error}"
                ))
                .with_builtin(name)
                .with_source(error)
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .build()
            })?;
            if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err() {
                return Err(invalid("has no owning provider"));
            }
            let element =
                crate::builtins::common::gpu_helpers::expected_handle_numeric_element_type(handle)
                    .map_err(|error| invalid(&format!("has invalid physical metadata: {error}")))?;
            if !crate::builtins::common::gpu_helpers::numeric_descriptor_matches(
                handle,
                element,
                runmat_accelerate_api::GpuTensorStorage::Real,
            ) {
                return Err(invalid("has inconsistent physical metadata"));
            }
            Ok(())
        }
        _ => Err(invalid(
            "must be a finite real scalar or admitted two-element interval",
        )),
    }
}

#[cfg(test)]
pub(crate) async fn prepare_floating_value(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<Value> {
    preflight_floating_value(
        name,
        &value,
        numeric_extension,
        resident_extension,
        role,
        None,
    )?;
    gather_exact_floating_value(name, value, role).await
}

pub(crate) async fn prepare_floating_value_with_cooperation(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    preflight_floating_value(
        name,
        &value,
        numeric_extension,
        resident_extension,
        role,
        cooperation.runtime(),
    )?;
    gather_exact_floating_value_with_cooperation(name, value, role, cooperation).await
}

fn preflight_floating_value(
    name: &str,
    value: &Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    preflight_host_residency(name, value, Some(resident_extension), runtime)?;
    if is_typed_integer_value(value) || is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            numeric_extension,
            name,
        )?;
    }
    preflight_real_array(name, value, role, true)
}

#[cfg(test)]
async fn gather_exact_floating_value(name: &str, value: Value, role: &str) -> BuiltinResult<Value> {
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "optim" },
        None,
    );
    gather_exact_floating_value_with_cooperation(name, value, role, &mut cooperation).await
}

pub(crate) async fn gather_exact_floating_value_with_cooperation(
    name: &str,
    value: Value,
    role: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let prepared = cooperation.integer_boundary(&value).await?;
    if !prepared.is_exact() {
        return Err(optim_error(
            name,
            format!("{name}: integer {role} must be exactly representable as double"),
        ));
    }
    if let Some(payload) = prepared.into_resident_payload() {
        return payload
            .into_value()
            .map_err(|error| optim_error(name, format!("{name}: {error}")));
    }
    crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime()).await
}

pub(crate) async fn initial_guess_with_extensions(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<InitialGuess> {
    preflight_floating_value(
        name,
        &value,
        numeric_extension,
        resident_extension,
        "initial point",
        cooperation.runtime(),
    )?;
    preflight_real_array(name, &value, "initial guess", false)?;
    let value =
        gather_exact_floating_value_with_cooperation(name, value, "initial point", cooperation)
            .await?;
    initial_guess_with_cooperation(name, value, cooperation).await
}

pub(crate) async fn value_to_real_vector_with_extensions(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    preflight_floating_value(
        name,
        &value,
        numeric_extension,
        resident_extension,
        role,
        cooperation.runtime(),
    )?;
    let value =
        gather_exact_floating_value_with_cooperation(name, value, role, cooperation).await?;
    value_to_real_vector_with_cooperation(name, value, cooperation).await
}

pub(crate) async fn value_to_scalar_with_extensions(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    preflight_floating_value(
        name,
        &value,
        numeric_extension,
        resident_extension,
        role,
        cooperation.runtime(),
    )?;
    if crate::value_fact::value_fact(&value)
        .shape
        .element_count()
        .is_some_and(|count| count != 1)
    {
        return Err(optim_error(name, format!("{name}: {role} must be scalar")));
    }
    let value =
        gather_exact_floating_value_with_cooperation(name, value, role, cooperation).await?;
    value_to_scalar(name, value)
}

pub(crate) fn ensure_selected_option_extensions(
    solver: OptimSolver,
    options: Option<&StructValue>,
    option_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let Some(options) = options else {
        return Ok(());
    };
    let name = solver.name();
    let fields = ScalarOptionFields::from_struct(Some(options));
    for field in OptimOptionField::ALL {
        if !solver.accepts(field) {
            continue;
        }
        let Some(value) = fields.get(field) else {
            continue;
        };
        preflight_host_residency(name, value, Some(resident_extension), runtime)?;
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                option_extension,
                name,
            )?;
        }
    }
    Ok(())
}

fn ensure_exact_binary64_integer(name: &str, value: &Value, role: &str) -> BuiltinResult<()> {
    let exact = crate::builtins::common::validation::integer_is_exact_f64;
    let valid = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(optim_error(
            name,
            format!("{name}: integer {role} must be exactly representable as double"),
        ))
    }
}

pub(crate) fn value_to_scalar(name: &str, value: Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => ensure_finite(name, n),
        Value::Int(i) => ensure_finite(name, i.to_f64()),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) => {
            if tensor.len() == 1 {
                ensure_finite(name, tensor::tensor_value_f64(&tensor, 0))
            } else {
                Err(optim_error(
                    name,
                    format!("{name}: function value must be a scalar"),
                ))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
            } else {
                Err(optim_error(
                    name,
                    format!("{name}: function value must be a scalar"),
                ))
            }
        }
        other => Err(optim_error(
            name,
            format!("{name}: function value must be real numeric, got {other:?}"),
        )),
    }
}

#[cfg(test)]
pub(crate) async fn value_to_real_vector(name: &str, value: Value) -> BuiltinResult<Vec<f64>> {
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "optim" },
        None,
    );
    value_to_real_vector_with_cooperation(name, value, &mut cooperation).await
}

pub(crate) async fn value_to_real_vector_with_cooperation(
    name: &str,
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let value =
        crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime()).await?;
    match value {
        Value::Num(n) => Ok(vec![ensure_finite(name, n)?]),
        Value::Int(i) => Ok(vec![ensure_finite(name, i.to_f64())?]),
        Value::Bool(b) => Ok(vec![if b { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => {
            let values = tensor_values(&tensor, cooperation).await?;
            if let Some(value) = first_nonfinite(&values, cooperation).await? {
                ensure_finite(name, value)?;
            }
            Ok(values)
        }
        Value::LogicalArray(logical) => logical_values(&logical.data, cooperation).await,
        other => Err(optim_error(
            name,
            format!("{name}: function value must be a real numeric vector, got {other:?}"),
        )),
    }
}

#[cfg(test)]
pub(crate) async fn initial_guess(name: &str, value: Value) -> BuiltinResult<InitialGuess> {
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "optim" },
        None,
    );
    initial_guess_with_cooperation(name, value, &mut cooperation).await
}

pub(crate) async fn initial_guess_with_cooperation(
    name: &str,
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<InitialGuess> {
    preflight_real_array(name, &value, "initial guess", false)?;
    let value =
        crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime()).await?;
    match value {
        Value::Num(n) => Ok(InitialGuess {
            values: vec![ensure_finite(name, n)?],
            shape: vec![1, 1],
            scalar: true,
        }),
        Value::Int(i) => Ok(InitialGuess {
            values: vec![ensure_finite(name, i.to_f64())?],
            shape: vec![1, 1],
            scalar: true,
        }),
        Value::Bool(b) => Ok(InitialGuess {
            values: vec![if b { 1.0 } else { 0.0 }],
            shape: vec![1, 1],
            scalar: true,
        }),
        Value::Tensor(tensor) => {
            let values = tensor_values(&tensor, cooperation).await?;
            if values.is_empty() {
                return Err(optim_error(
                    name,
                    format!("{name}: initial guess cannot be empty"),
                ));
            }
            if let Some(value) = first_nonfinite(&values, cooperation).await? {
                ensure_finite(name, value)?;
            }
            Ok(InitialGuess {
                values,
                shape: tensor.shape().to_vec(),
                scalar: false,
            })
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Err(optim_error(
                    name,
                    format!("{name}: initial guess cannot be empty"),
                ));
            }
            Ok(InitialGuess {
                values: logical_values(&logical.data, cooperation).await?,
                shape: logical.shape.to_vec(),
                scalar: false,
            })
        }
        other => Err(optim_error(
            name,
            format!("{name}: initial guess must be real numeric, got {other:?}"),
        )),
    }
}

pub(crate) fn vector_to_value(
    name: &str,
    values: Vec<f64>,
    shape: &[usize],
    scalar: bool,
) -> BuiltinResult<Value> {
    if scalar {
        Ok(Value::Num(values[0]))
    } else {
        Tensor::new(values, shape.to_vec())
            .map(Value::Tensor)
            .map_err(|e| optim_error(name, format!("{name}: {e}")))
    }
}

pub(crate) fn field_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::CharArray(array) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data()).map_err(|error| {
                optim_error(
                    "optimset",
                    format!("optimset: option name contains {error}"),
                )
            })
        }
        other => Err(optim_error(
            "optimset",
            format!("optimset: option names must be strings, got {other:?}"),
        )),
    }
}

pub(crate) fn lookup_option<'a>(options: &'a StructValue, name: &str) -> Option<&'a Value> {
    options
        .fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

pub(crate) fn preflight_option_scalar(
    builtin: &str,
    value: &Value,
    field: runmat_builtins::OptimOptionField,
) -> BuiltinResult<()> {
    let invalid = || {
        optim_error(
            builtin,
            format!(
                "{builtin}: option {} must be a full real numeric scalar",
                field.name()
            ),
        )
    };
    // Reject nonnumeric carriers before deriving recursive value facts or
    // formatting captures/properties that are not part of this scalar role.
    if !matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) {
        return Err(invalid());
    }
    let fact = crate::value_fact::value_fact(value);
    let numeric = matches!(fact.kind, runmat_types::ValueKindFact::Numeric(numeric)
        if numeric.domain == runmat_types::NumericDomain::Real)
        || matches!(fact.kind, runmat_types::ValueKindFact::Unknown);
    if !numeric
        || fact.storage == runmat_types::StorageFact::Sparse
        || fact.shape.element_count().is_some_and(|count| count != 1)
    {
        return Err(invalid());
    }
    Ok(())
}

pub(crate) fn option_f64_value(
    builtin: &str,
    value: Option<&Value>,
    field: &str,
    default: f64,
) -> BuiltinResult<f64> {
    let Some(value) = value else {
        return Ok(default);
    };
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) if crate::builtins::common::validation::integer_is_exact_f64(i) => i.to_f64(),
        Value::Int(_) => {
            return Err(optim_error(
                builtin,
                format!("{builtin}: option {field} must be exactly representable as double"),
            ))
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            ensure_exact_binary64_integer(builtin, value, &format!("option {field}"))?;
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(optim_error(
                builtin,
                format!("{builtin}: option {field} must be numeric, got {other:?}"),
            ))
        }
    };
    ensure_finite(builtin, parsed)
}

pub(crate) fn option_usize_value(
    builtin: &str,
    raw: Option<&Value>,
    field: &str,
    default: usize,
) -> BuiltinResult<usize> {
    let Some(raw) = raw else {
        return Ok(default);
    };
    if let Some(integer) = tensor::scalar_integer_value(raw) {
        return integer.try_to_usize().ok_or_else(|| {
            optim_error(
                builtin,
                format!("{builtin}: option {field} must be non-negative"),
            )
        });
    }
    let value = option_f64_value(builtin, Some(raw), field, default as f64)?;
    if value < 0.0 {
        return Err(optim_error(
            builtin,
            format!("{builtin}: option {field} must be non-negative"),
        ));
    }
    if value.fract() != 0.0 {
        return Err(optim_error(
            builtin,
            format!("{builtin}: option {field} must be an integer"),
        ));
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return Err(optim_error(
            builtin,
            format!("{builtin}: option {field} exceeds maximum supported size"),
        ));
    }
    Ok(value as usize)
}

pub(crate) fn option_string_value(
    value: Option<&Value>,
    field: &str,
    default: &str,
) -> BuiltinResult<String> {
    let Some(value) = value else {
        return Ok(default.to_string());
    };
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_ascii_lowercase()),
        Value::CharArray(array) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data())
                .map(|text| text.to_ascii_lowercase())
                .map_err(|error| {
                    optim_error("optim", format!("optim option {field} contains {error}"))
                })
        }
        other => Err(optim_error(
            "optim",
            format!("optim option {field} must be a string, got {other:?}"),
        )),
    }
}

fn ensure_finite(name: &str, value: f64) -> BuiltinResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(optim_error(
            name,
            format!("{name}: function value must be finite"),
        ))
    }
}

#[derive(Debug)]
pub(crate) struct InitialGuess {
    pub values: Vec<f64>,
    pub shape: Vec<usize>,
    pub scalar: bool,
}

#[cfg(test)]
mod tests {
    use super::{
        call_function_in_context, canonicalize_callback_handle, initial_guess,
        initial_guess_with_extensions, output_callback, prepare_floating_value,
        value_to_real_vector, value_to_scalar, ScalarCooperation, ScalarOptionFields,
    };
    use crate::build_runtime_error;
    use futures::executor::block_on;
    use runmat_builtins::{OptimOptionField, OptimSolver};
    use runmat_value::{CharArray, Closure, IntegerStorage, StringArray, Tensor, Value};
    use std::sync::Arc;

    #[test]
    fn scalar_option_fields_decode_public_aliases_to_one_typed_control_plan() {
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("StepTolerance", Value::Num(0.125));
        options.insert("MAXITER", Value::Num(17.0));
        options.insert("Display", Value::String("iter".to_string().into()));
        let options = options.finish();
        let admitted = ScalarOptionFields::from_struct(Some(&options));
        assert!(matches!(admitted.get(OptimOptionField::TolX), Some(Value::Num(v)) if *v == 0.125));
        assert!(
            matches!(admitted.get(OptimOptionField::MaxIter), Some(Value::Num(v)) if *v == 17.0)
        );
        assert!(
            matches!(admitted.get(OptimOptionField::Display), Some(Value::String(v)) if v == "iter")
        );
        assert!(admitted.get(OptimOptionField::Algorithm).is_none());
    }

    #[test]
    fn value_to_scalar_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-7]), vec![1, 1]).unwrap();

        let parsed = value_to_scalar("optim_test", Value::Tensor(tensor)).unwrap();

        assert_eq!(parsed, -7.0);
    }

    #[test]
    fn value_to_real_vector_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![3, 5, 8]), vec![1, 3]).unwrap();

        let parsed = block_on(value_to_real_vector("optim_test", Value::Tensor(tensor))).unwrap();

        assert_eq!(parsed, vec![3.0, 5.0, 8.0]);
    }

    #[test]
    fn initial_guess_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I32(vec![11, -13]), vec![2, 1]).unwrap();

        let guess = block_on(initial_guess("optim_test", Value::Tensor(tensor))).unwrap();

        assert_eq!(guess.values, vec![11.0, -13.0]);
        assert_eq!(guess.shape, vec![2, 1]);
        assert!(!guess.scalar);
    }

    #[test]
    fn solver_numeric_extension_covers_all_integer_classes_and_rejects_wide_values() {
        let storages = [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in storages {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            let guess = block_on(initial_guess_with_extensions(
                "fsolve",
                value,
                &super::super::fsolve::FSOLVE_INPUT_NUMERIC_EXTENSION,
                &super::super::fsolve::FSOLVE_RESIDENT_EXTENSION,
                &mut super::ScalarCooperation::new(runmat_builtins::OptimSolver::Fsolve, None),
            ))
            .unwrap();
            assert_eq!(guess.values, vec![1.0]);
        }
        let wide = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
        );
        let error = block_on(initial_guess_with_extensions(
            "fsolve",
            wide,
            &super::super::fsolve::FSOLVE_INPUT_NUMERIC_EXTENSION,
            &super::super::fsolve::FSOLVE_RESIDENT_EXTENSION,
            &mut super::ScalarCooperation::new(runmat_builtins::OptimSolver::Fsolve, None),
        ))
        .expect_err("wide integer cannot cross exactly");
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn optimizer_integer_metadata_covers_every_runmat_only_role() {
        let capability_sets: [&[runmat_builtins::BuiltinIntegerCapabilityDescriptor]; 4] = [
            &runmat_builtins::FMINBND_INTEGER_CAPABILITIES,
            &runmat_builtins::FMINUNC_INTEGER_CAPABILITIES,
            &runmat_builtins::FSOLVE_INTEGER_CAPABILITIES,
            &runmat_builtins::FZERO_INTEGER_CAPABILITIES,
        ];
        assert_eq!(
            capability_sets.map(|capabilities| capabilities.len()),
            [4, 6, 4, 4]
        );
        for capabilities in capability_sets {
            assert!(capabilities.iter().any(|capability| {
                capability.backend == runmat_builtins::BuiltinIntegerBackendRule::GatherFallback
            }));
            assert!(capabilities.iter().any(|capability| {
                capability.computation_domain
                    == runmat_builtins::BuiltinIntegerComputationDomain::Structural
            }));
            for input in capabilities.iter().flat_map(|capability| capability.inputs) {
                assert_eq!(
                    input.availability,
                    runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly
                );
                assert_eq!(input.classes.len(), 8);
            }
        }
    }

    #[test]
    fn solver_strict_mode_rejects_integer_logical_and_resident_values_before_gather() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for value in [
            Value::Tensor(Tensor::new_integer(IntegerStorage::I8(vec![1]), vec![1, 1]).unwrap()),
            Value::Bool(true),
        ] {
            let error = block_on(prepare_floating_value(
                "fminunc",
                value,
                &super::super::fminunc::FMINUNC_INPUT_NUMERIC_EXTENSION,
                &super::super::fminunc::FMINUNC_RESIDENT_EXTENSION,
                "initial point",
            ))
            .expect_err("strict mode rejects extension");
            assert!(error
                .identifier()
                .is_some_and(|id| id.starts_with("RunMat:compatibility:")));
        }
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );
        let resident = Value::GpuTensor(
            allocation
                .handle
                .clone()
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap(),
        );
        let error = block_on(prepare_floating_value(
            "fminunc",
            resident,
            &super::super::fminunc::FMINUNC_INPUT_NUMERIC_EXTENSION,
            &super::super::fminunc::FMINUNC_RESIDENT_EXTENSION,
            "initial point",
        ))
        .expect_err("resident policy is checked before provider access");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FminuncResidentExtension")
        );
        let automatic = Value::GpuTensor(
            allocation
                .handle
                .clone()
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .unwrap(),
        );
        super::preflight_host_residency("fminunc", &automatic, None, None).unwrap();
        let unproven = Value::GpuTensor(allocation.handle.clone());
        let error = super::preflight_host_residency("fminunc", &unproven, None, None).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("CustomResident", unproven.clone());
        options.insert("CustomInteger", Value::Int(runmat_value::IntValue::I16(7)));
        options.insert("OutputFcn", unproven);
        options.insert("MaxIter", Value::Num(7.0));
        let options = options.finish();
        super::ensure_selected_option_extensions(
            OptimSolver::Fzero,
            Some(&options),
            &runmat_builtins::FZERO_OPTION_EXTENSION,
            &runmat_builtins::FZERO_RESIDENT_EXTENSION,
            None,
        )
        .expect("ignored fields do not acquire control admission requirements");
        let mut options = runmat_value::UnboundedStructValueBuilder::from_structure(options);
        options.insert("OutputFcn", Value::Int(runmat_value::IntValue::I16(7)));
        let options = options.finish();
        super::ensure_selected_option_extensions(
            OptimSolver::Fzero,
            Some(&options),
            &runmat_builtins::FZERO_OPTION_EXTENSION,
            &runmat_builtins::FZERO_RESIDENT_EXTENSION,
            None,
        )
        .expect("an unconsumed callback field is not a numeric control");
        let mut options = runmat_value::UnboundedStructValueBuilder::from_structure(options);
        options.insert("MaxIter", Value::Int(runmat_value::IntValue::I16(7)));
        let options = options.finish();
        let error = super::ensure_selected_option_extensions(
            OptimSolver::Fzero,
            Some(&options),
            &runmat_builtins::FZERO_OPTION_EXTENSION,
            &runmat_builtins::FZERO_RESIDENT_EXTENSION,
            None,
        )
        .expect_err("a consumed native integer control retains its extension gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FzeroOptionExtension")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(allocation.probe.retirements(), 0);
        // Both derived fixture owners are still live until explicitly retired.
        // Release them before asking the original handle to retire its lease.
        drop(options);
        drop(automatic);
        allocation.handle.release().unwrap();
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn option_usize_reads_typed_integer_storage_exactly() {
        let max_iter = Tensor::new_integer(IntegerStorage::U16(vec![37]), vec![1, 1]).unwrap();
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("MaxIter", Value::Tensor(max_iter));
        let options = options.finish();

        assert_eq!(
            ScalarOptionFields::from_struct(Some(&options))
                .count("optim_test", OptimOptionField::MaxIter, 400)
                .unwrap(),
            37
        );
    }

    #[test]
    fn option_usize_rejects_negative_typed_integer_storage_exactly() {
        let max_iter = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).unwrap();
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("MaxIter", Value::Tensor(max_iter));
        let options = options.finish();

        assert!(ScalarOptionFields::from_struct(Some(&options))
            .count("optim_test", OptimOptionField::MaxIter, 400)
            .is_err());
    }

    #[test]
    fn option_usize_accepts_wide_typed_integer_storage_despite_poisoned_mirror() {
        let wide = 9_007_199_254_740_993_u64;
        let max_iter = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("MaxIter", Value::Tensor(max_iter));
        let options = options.finish();

        let parsed = ScalarOptionFields::from_struct(Some(&options)).count(
            "optim_test",
            OptimOptionField::MaxIter,
            400,
        );
        if let Ok(expected) = usize::try_from(wide) {
            assert_eq!(parsed.unwrap(), expected);
        } else {
            assert!(
                parsed.is_err(),
                "a structural count must fit the actual target"
            );
        }
        let field = runmat_builtins::OptimOptionField::MaxFunEvals;
        assert!(ScalarOptionFields::from_struct(None)
            .count_product_default("fsolve", field, 100, usize::MAX)
            .is_err());
        let mut options = runmat_value::UnboundedStructValueBuilder::from_structure(options);
        options.insert("MaxFunEvals", Value::Int(runmat_value::IntValue::U64(17)));
        let options = options.finish();
        assert_eq!(
            ScalarOptionFields::from_struct(Some(&options))
                .count_product_default("fsolve", field, 100, usize::MAX)
                .unwrap(),
            17
        );
    }

    #[test]
    fn option_usize_rejects_fractional_and_unrepresentable_double_values() {
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("MaxIter", Value::Num(3.5));
        let mut options = options.finish();
        assert!(ScalarOptionFields::from_struct(Some(&options))
            .count("optim_test", OptimOptionField::MaxIter, 400)
            .is_err());

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        options
            .try_replace_unique("MaxIter", Value::Num(boundary))
            .expect("fresh unique options");
        assert!(ScalarOptionFields::from_struct(Some(&options))
            .count("optim_test", OptimOptionField::MaxIter, 400)
            .is_err());
    }

    #[test]
    fn callback_handle_canonicalizer_binds_function_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(42)
            })))
            .expect("call-port scope");
        let canonical =
            canonicalize_callback_handle(&Value::FunctionHandle("decay".into())).unwrap();
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "decay".into(),
                    function: 42,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(42),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_qualified_external_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(43)
            })))
            .expect("call-port scope");
        let canonical =
            canonicalize_callback_handle(&Value::ExternalFunctionHandle("pkg.decay".into()))
                .unwrap();
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "pkg.decay".into(),
                    function: 43,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(43),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_keeps_malformed_external_handle_name_shaped() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg..decay").then_some(44)
            })))
            .expect("call-port scope");
        let raw = Value::ExternalFunctionHandle("pkg..decay".into());
        let canonical = canonicalize_callback_handle(&raw).unwrap();
        assert_eq!(canonical, raw);
    }

    #[test]
    fn callback_handle_canonicalizer_binds_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(45)
            })))
            .expect("call-port scope");
        let canonical =
            canonicalize_callback_handle(&Value::String("@decay".to_string().into())).unwrap();
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "decay".into(),
                    function: 45,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(45),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_trims_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(145)
            })))
            .expect("call-port scope");
        let canonical =
            canonicalize_callback_handle(&Value::String("  @decay  ".to_string().into())).unwrap();
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "decay".into(),
                    function: 145,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(145),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_string_array_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(46)
            })))
            .expect("call-port scope");
        let canonical = canonicalize_callback_handle(&Value::StringArray(
            StringArray::new(vec!["@pkg.decay".to_string()], vec![1, 1]).expect("string array"),
        ))
        .unwrap();
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "pkg.decay".into(),
                    function: 46,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(46),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_trims_string_array_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(146)
            })))
            .expect("call-port scope");
        let canonical = canonicalize_callback_handle(&Value::StringArray(
            StringArray::new(vec!["  @pkg.decay  ".to_string()], vec![1, 1]).expect("string array"),
        ))
        .unwrap();
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "pkg.decay".into(),
                    function: 146,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(146),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_char_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(47)
            })))
            .expect("call-port scope");
        let canonical =
            canonicalize_callback_handle(&Value::CharArray(CharArray::new_row("@decay"))).unwrap();
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "decay".into(),
                    function: 47,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(47),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_trims_char_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(147)
            })))
            .expect("call-port scope");
        let canonical =
            canonicalize_callback_handle(&Value::CharArray(CharArray::new_row("  @decay  ")))
                .unwrap();
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "decay".into(),
                    function: 147,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(147),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                }
            ))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_name_only_closure_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(48)
            })))
            .expect("call-port scope");
        let raw = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "decay".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::Num(9.0)].into(),
        }));
        let canonical = canonicalize_callback_handle(&raw).unwrap();
        assert_eq!(
            canonical,
            Value::Closure(Closure::from_data(runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "decay".into(),
                binding: Some(runmat_value::BoundFunctionBinding::unavailable(48)),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: vec![Value::Num(9.0)].into(),
            }))
        );
    }

    #[test]
    fn callback_handle_canonicalizer_keeps_name_only_closure_without_resolver() {
        let raw = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "decay".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::Num(9.0)].into(),
        }));
        let canonical = canonicalize_callback_handle(&raw).unwrap();
        assert_eq!(canonical, raw);
    }

    #[test]
    fn explicit_callback_owner_beats_foreign_ambient_for_objective_analytic_and_observer() {
        use super::output_callback::{IterationValues, OutputState};
        use std::rc::Rc;
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let original = crate::user_functions::test_support::exact_fixture_runtime()
            .with_cancellation_token(cancellation.token());
        original.set_error_namespace("OriginalOptimization");
        let foreign = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        foreign.set_error_namespace("ForeignOptimization");
        let _original_bindings =
            crate::user_functions::test_support::install_exact_named_forwarders(
                &original,
                &[("context_objective", 3301), ("context_observer", 3302)],
            )
            .unwrap();
        let original_entry = original.enter();
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let _original_invoker = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(move |id, args, requested| {
                counted.fetch_add(1, Ordering::SeqCst);
                let values = if id == 3302 {
                    assert_eq!(requested, 1);
                    let Value::Struct(metadata) = &args[1] else {
                        panic!("observer metadata")
                    };
                    assert_eq!(metadata.fields.len(), 10);
                    vec![Value::Bool(false)]
                } else {
                    assert_eq!(id, 3301);
                    if matches!(args.first(), Some(Value::Num(-1.0))) {
                        Vec::new()
                    } else {
                        (0..requested)
                            .map(|index| Value::Num(71.0 + index as f64))
                            .collect()
                    }
                };
                Box::pin(async move {
                    // Compiled callbacks may return a one-element CSE even
                    // for scalar objectives and logical observer results.
                    let sequence = crate::sequence::ValueSequence::comma_separated(values)
                        .map_err(crate::sequence::sequence_error_to_runtime)?;
                    crate::execution::RootedValueSequence::new(sequence)
                        .map_err(|error| error.into_runtime_error())
                })
            }),
        ))
        .unwrap();
        drop(original_entry);
        let _foreign_bindings =
            crate::user_functions::test_support::install_exact_named_forwarders(
                &foreign,
                &[("context_objective", 4401), ("context_observer", 4402)],
            )
            .unwrap();
        let _foreign_entry = foreign.enter();
        let foreign_calls = Arc::new(AtomicUsize::new(0));
        let counted = foreign_calls.clone();
        let _foreign_invoker = crate::user_functions::install_semantic_function_invoker(Some(
            Arc::new(move |_, _, _| {
                counted.fetch_add(1, Ordering::SeqCst);
                Box::pin(async {
                    Err(build_runtime_error("foreign callback must not run").build())
                })
            }),
        ))
        .unwrap();
        for requested in [1, 2] {
            let result = futures::executor::block_on(call_function_in_context(
                Some(&original),
                &Value::FunctionHandle("context_objective".into()),
                [Value::Num(0.0)],
                requested,
            ))
            .unwrap();
            if requested == 1 {
                assert!(matches!(result.value(), Value::Num(71.0)));
            } else {
                let Value::OutputList(values) = result.value() else {
                    panic!("analytic outputs")
                };
                assert!(matches!(values[0], Value::Num(71.0)));
                assert!(matches!(values[1], Value::Num(72.0)));
            }
        }
        let error = futures::executor::block_on(call_function_in_context(
            Some(&original),
            &Value::FunctionHandle("context_objective".into()),
            [Value::Num(-1.0)],
            1,
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("OriginalOptimization:CommaSeparatedListRequiresSingleValue")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let mut cooperation = ScalarCooperation::new(OptimSolver::Fminunc, Some(&original));
        assert!(!futures::executor::block_on(output_callback::notify(
            Some(&Value::FunctionHandle("context_observer".into())),
            OutputState::Init,
            IterationValues {
                x: &[1.0],
                shape: &[1, 1],
                scalar: true,
                iteration: 0,
                func_count: 1,
                fval: 71.0,
                gradient: &[0.0],
                first_order_optimality: 0.0,
                step_size: 0.0,
                search_direction: &[],
                line_step_length: None,
                directional_derivative: None
            },
            &mut cooperation
        ))
        .unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!(foreign_calls.load(Ordering::SeqCst), 0);
        for allowed in [true, false] {
            original.set_runmat_extensions_enabled(allowed);
            foreign.set_runmat_extensions_enabled(!allowed);
            let converted = block_on(super::value_to_scalar_with_extensions(
                "fminunc",
                Value::Int(runmat_value::IntValue::I16(73)),
                &runmat_builtins::FMINUNC_CALLBACK_NUMERIC_EXTENSION,
                &runmat_builtins::FMINUNC_RESIDENT_EXTENSION,
                "objective value",
                &mut cooperation,
            ));
            if allowed {
                assert_eq!(converted.unwrap(), 73.0);
            } else {
                assert!(converted
                    .unwrap_err()
                    .identifier()
                    .is_some_and(|id| id.starts_with("RunMat:compatibility:")));
            }
        }
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let error = futures::executor::block_on(call_function_in_context(
            Some(&original),
            &Value::FunctionHandle("context_objective".into()),
            [Value::Num(0.0)],
            1,
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("OriginalOptimization:Cancelled"));
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!(foreign_calls.load(Ordering::SeqCst), 0);
    }
}

pub(crate) mod multiobjective;

pub(crate) mod semi_options;

pub(crate) mod proof_rational;

pub(crate) mod csc_rows;
pub(crate) mod native_class_registration;
pub(crate) mod solver_workspace;
pub(crate) mod working_set;
