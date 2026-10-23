use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_builtins::OptimSolver;
use runmat_value::{StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

mod scalar_cooperation;
pub(crate) use scalar_cooperation::ScalarCooperation;
mod host_arrays;
pub(crate) use host_arrays::{first_nonfinite, logical_values, tensor_values};
mod scalar_options;
pub(crate) use scalar_options::{prepare_solver_options, ScalarOptionFields};

/// Reuse the canonical input contract before provider work. Payload values,
/// finite checks and authentic resident allocation admission remain runtime-owned.
pub(crate) fn preflight_catalog_inputs<'a>(
    entry: &'static runmat_builtins::BuiltinCatalogEntry,
    error: &'static runmat_builtins::BuiltinErrorDescriptor,
    values: impl IntoIterator<Item = &'a Value>,
) -> BuiltinResult<()> {
    let inference = runmat_builtins::infer_catalog_call(
        entry,
        &runmat_types::CallRequest {
            arguments: values
                .into_iter()
                .map(crate::value_fact::value_fact)
                .collect(),
            literals: runmat_types::LiteralContext::default(),
            outputs: runmat_types::OutputSelection::new(
                runmat_types::RequestedOutputCount::Exactly(1),
            ),
        },
    );
    if let Some(diagnostic) = inference
        .diagnostics
        .into_iter()
        .find(|diagnostic| diagnostic.severity == runmat_types::InferenceSeverity::Error)
    {
        let mut builder =
            build_runtime_error(format!("{}: {}", entry.identity.name, diagnostic.message))
                .with_builtin(entry.identity.name);
        if let Some(identifier) = error.identifier {
            builder = builder.with_identifier(identifier);
        }
        return Err(builder.build());
    }
    Ok(())
}

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
    let callback = canonicalize_callback_handle(handle)?;
    crate::call_feval_async_with_outputs(callback, args.as_ref(), 1).await
}

pub(crate) async fn call_scalar_function(
    name: &str,
    solver: OptimSolver,
    handle: &Value,
    x: f64,
) -> BuiltinResult<f64> {
    call_scalar_function_with_precision(name, solver, handle, x, false).await
}

pub(crate) async fn call_scalar_function_with_precision(
    name: &str,
    solver: OptimSolver,
    handle: &Value,
    x: f64,
    single: bool,
) -> BuiltinResult<f64> {
    call_scalar_function_with_precision_info(name, solver, handle, x, single)
        .await
        .map(|(value, _)| value)
}

pub(crate) async fn call_scalar_function_with_precision_info(
    name: &str,
    solver: OptimSolver,
    handle: &Value,
    x: f64,
    single: bool,
) -> BuiltinResult<(f64, bool)> {
    let argument = if single {
        Value::Tensor(
            Tensor::new_with_dtype(vec![x], vec![1, 1], runmat_value::NumericDType::F32)
                .map_err(|error| optim_error(name, format!("{name}: {error}")))?,
        )
    } else {
        Value::Num(x)
    };
    let value = call_function(handle, vec![argument]).await?;
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
                    prepare_floating_value(
                        name,
                        value,
                        &runmat_builtins::FMINBND_CALLBACK_NUMERIC_EXTENSION,
                        &runmat_builtins::FMINBND_RESIDENT_EXTENSION,
                        "objective value",
                    )
                    .await?
                }
                OptimSolver::Fzero => {
                    prepare_floating_value(
                        name,
                        value,
                        &runmat_builtins::FZERO_CALLBACK_NUMERIC_EXTENSION,
                        &runmat_builtins::FZERO_RESIDENT_EXTENSION,
                        "function value",
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
pub(crate) fn preflight_scalar_operand(
    solver: OptimSolver,
    value: &Value,
    geometry: ScalarOperandGeometry,
    role: &str,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    let name = solver.name();
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(resident_extension, name)?;
    }
    if is_typed_integer_value(value) || is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(numeric_extension, name)?;
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
            if crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle).is_none() {
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

pub(crate) async fn prepare_floating_value(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<Value> {
    preflight_floating_value(name, &value, numeric_extension, resident_extension, role)?;
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
    preflight_floating_value(name, &value, numeric_extension, resident_extension, role)?;
    gather_exact_floating_value_with_cooperation(name, value, role, cooperation).await
}

fn preflight_floating_value(
    name: &str,
    value: &Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(resident_extension, name)?;
    }
    if is_typed_integer_value(value) || is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(numeric_extension, name)?;
    }
    preflight_real_array(name, value, role, true)
}

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
    crate::dispatcher::gather_if_needed_async(&value).await
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
    preflight_floating_value(name, &value, numeric_extension, resident_extension, role)?;
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
) -> BuiltinResult<f64> {
    preflight_floating_value(name, &value, numeric_extension, resident_extension, role)?;
    if crate::value_fact::value_fact(&value)
        .shape
        .element_count()
        .is_some_and(|count| count != 1)
    {
        return Err(optim_error(name, format!("{name}: {role} must be scalar")));
    }
    let value = gather_exact_floating_value(name, value, role).await?;
    value_to_scalar(name, value)
}

pub(crate) fn ensure_option_extensions(
    name: &str,
    options: Option<&StructValue>,
    option_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    let Some(options) = options else {
        return Ok(());
    };
    for value in options.fields.values() {
        if matches!(value, Value::GpuTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(resident_extension, name)?;
        }
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(option_extension, name)?;
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
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
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
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
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
        Value::String(s) => Ok(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 => Ok(sa.data()[0].as_str().to_owned()),
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

pub(crate) fn option_f64(
    builtin: &str,
    options: Option<&StructValue>,
    field: &str,
    default: f64,
) -> BuiltinResult<f64> {
    let Some(options) = options else {
        return Ok(default);
    };
    option_f64_value(builtin, lookup_option(options, field), field, default)
}

pub(crate) fn preflight_option_scalar(
    builtin: &str,
    value: &Value,
    field: runmat_builtins::OptimOptionField,
) -> BuiltinResult<()> {
    let fact = crate::value_fact::value_fact(value);
    let numeric = matches!(fact.kind, runmat_types::ValueKindFact::Numeric(numeric)
        if numeric.domain == runmat_types::NumericDomain::Real)
        || matches!(fact.kind, runmat_types::ValueKindFact::Unknown);
    if !numeric
        || fact.storage == runmat_types::StorageFact::Sparse
        || fact.shape.element_count().is_some_and(|count| count != 1)
    {
        return Err(optim_error(
            builtin,
            format!(
                "{builtin}: option {} must be numeric scalar, got {value:?}",
                field.name()
            ),
        ));
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

pub(crate) fn option_usize(
    builtin: &str,
    options: Option<&StructValue>,
    field: &str,
    default: usize,
) -> BuiltinResult<usize> {
    let Some(options) = options else {
        return Ok(default);
    };
    option_usize_value(builtin, lookup_option(options, field), field, default)
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
        Value::String(s) => Ok(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data().len() == 1 => Ok(sa.data()[0].to_ascii_lowercase()),
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
        canonicalize_callback_handle, initial_guess, initial_guess_with_extensions, option_usize,
        prepare_floating_value, value_to_real_vector, value_to_scalar, ScalarOptionFields,
    };
    use futures::executor::block_on;
    use runmat_builtins::OptimOptionField;
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
            [4, 5, 4, 4]
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
        let resident = Value::GpuTensor(allocation.handle.clone());
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
        assert_eq!(allocation.probe.retirements(), 0);
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
            option_usize("optim_test", Some(&options), "MaxIter", 400).unwrap(),
            37
        );
    }

    #[test]
    fn option_usize_rejects_negative_typed_integer_storage_exactly() {
        let max_iter = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).unwrap();
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("MaxIter", Value::Tensor(max_iter));
        let options = options.finish();

        assert!(option_usize("optim_test", Some(&options), "MaxIter", 400).is_err());
    }

    #[test]
    fn option_usize_accepts_wide_typed_integer_storage_despite_poisoned_mirror() {
        let wide = 9_007_199_254_740_993_u64;
        let max_iter = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("MaxIter", Value::Tensor(max_iter));
        let options = options.finish();

        let parsed = option_usize("optim_test", Some(&options), "MaxIter", 400);
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
        assert!(option_usize("optim_test", Some(&options), "MaxIter", 400).is_err());

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        options
            .try_replace_unique("MaxIter", Value::Num(boundary))
            .expect("fresh unique options");
        assert!(option_usize("optim_test", Some(&options), "MaxIter", 400).is_err());
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
}
