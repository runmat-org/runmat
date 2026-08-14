use runmat_builtins::{BuiltinExtensionDescriptor, CharArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) fn optim_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn canonicalize_callback_handle(handle: &Value) -> Value {
    crate::canonicalize_callback_handle_for_semantic_resolution(handle.clone())
}

pub(crate) async fn call_function(handle: &Value, args: Vec<Value>) -> BuiltinResult<Value> {
    let callback = canonicalize_callback_handle(handle);
    crate::call_feval_async_with_outputs(callback, &args, 1).await
}

pub(crate) async fn call_scalar_function(name: &str, handle: &Value, x: f64) -> BuiltinResult<f64> {
    call_scalar_function_with_precision(name, handle, x, false).await
}

pub(crate) async fn call_scalar_function_with_precision(
    name: &str,
    handle: &Value,
    x: f64,
    single: bool,
) -> BuiltinResult<f64> {
    call_scalar_function_with_precision_info(name, handle, x, single)
        .await
        .map(|(value, _)| value)
}

pub(crate) async fn call_scalar_function_with_precision_info(
    name: &str,
    handle: &Value,
    x: f64,
    single: bool,
) -> BuiltinResult<(f64, bool)> {
    let argument = if single {
        Value::Tensor(
            Tensor::new_with_dtype(vec![x], vec![1, 1], runmat_builtins::NumericDType::F32)
                .map_err(|error| optim_error(name, format!("{name}: {error}")))?,
        )
    } else {
        Value::Num(x)
    };
    let value = call_function(handle, vec![argument]).await?;
    let output_single = match &value {
        Value::Tensor(tensor) => tensor.numeric_dtype() == runmat_builtins::NumericDType::F32,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    };
    let value = match name {
        "fminbnd" => {
            prepare_floating_value(
                name,
                value,
                &super::fminbnd::FMINBND_CALLBACK_NUMERIC_EXTENSION,
                &super::fminbnd::FMINBND_RESIDENT_EXTENSION,
                "objective value",
            )
            .await?
        }
        "fzero" => {
            prepare_floating_value(
                name,
                value,
                &super::fzero::FZERO_CALLBACK_NUMERIC_EXTENSION,
                &super::fzero::FZERO_RESIDENT_EXTENSION,
                "function value",
            )
            .await?
        }
        _ => crate::dispatcher::gather_if_needed_async(&value).await?,
    };
    value_to_scalar(name, value).map(|value| (value, output_single))
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

pub(crate) async fn prepare_floating_value(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(resident_extension, name)?;
    }
    if is_typed_integer_value(&value) || is_logical_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(numeric_extension, name)?;
    }
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value).await?
    {
        return Err(optim_error(
            name,
            format!("{name}: integer {role} must be exactly representable as double"),
        ));
    }
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    Ok(value)
}

pub(crate) async fn initial_guess_with_extensions(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<InitialGuess> {
    let value = prepare_floating_value(
        name,
        value,
        numeric_extension,
        resident_extension,
        "initial point",
    )
    .await?;
    initial_guess(name, value).await
}

pub(crate) async fn value_to_real_vector_with_extensions(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<Vec<f64>> {
    let value =
        prepare_floating_value(name, value, numeric_extension, resident_extension, role).await?;
    value_to_real_vector(name, value).await
}

pub(crate) async fn value_to_scalar_with_extensions(
    name: &str,
    value: Value,
    numeric_extension: &BuiltinExtensionDescriptor,
    resident_extension: &BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<f64> {
    let value =
        prepare_floating_value(name, value, numeric_extension, resident_extension, role).await?;
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
    let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
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
            let values = tensor::tensor_values_f64(&tensor);
            if values.len() == 1 {
                ensure_finite(name, values[0])
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

pub(crate) async fn value_to_real_vector(name: &str, value: Value) -> BuiltinResult<Vec<f64>> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    match value {
        Value::Num(n) => Ok(vec![ensure_finite(name, n)?]),
        Value::Int(i) => Ok(vec![ensure_finite(name, i.to_f64())?]),
        Value::Bool(b) => Ok(vec![if b { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => finite_vec(name, tensor::tensor_values_f64(&tensor)),
        Value::LogicalArray(logical) => Ok(logical
            .data
            .iter()
            .map(|&v| if v != 0 { 1.0 } else { 0.0 })
            .collect()),
        other => Err(optim_error(
            name,
            format!("{name}: function value must be a real numeric vector, got {other:?}"),
        )),
    }
}

pub(crate) async fn initial_guess(name: &str, value: Value) -> BuiltinResult<InitialGuess> {
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
            let values = tensor::tensor_values_f64(&tensor);
            if values.is_empty() {
                return Err(optim_error(
                    name,
                    format!("{name}: initial guess cannot be empty"),
                ));
            }
            Ok(InitialGuess {
                values: finite_vec(name, values)?,
                shape: tensor.shape,
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
                values: logical
                    .data
                    .iter()
                    .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                    .collect(),
                shape: logical.shape,
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
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(CharArray { data, rows: 1, .. }) => Ok(data.iter().collect()),
        other => Err(optim_error(
            "optimset",
            format!("optimset: option names must be strings, got {other:?}"),
        )),
    }
}

pub(crate) fn canonical_option_name(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "tolx" => "TolX".to_string(),
        "tolfun" | "functiontolerance" | "optimalitytolerance" => "TolFun".to_string(),
        "steptolerance" => "TolX".to_string(),
        "maxiter" => "MaxIter".to_string(),
        "maxfunevals" => "MaxFunEvals".to_string(),
        "display" => "Display".to_string(),
        "algorithm" => "Algorithm".to_string(),
        "specifyobjectivegradient" | "gradobj" => "SpecifyObjectiveGradient".to_string(),
        _ => name.to_string(),
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
    let Some(value) = lookup_option(options, field) else {
        return Ok(default);
    };
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) if crate::builtins::math::trigonometry::cos::integer_is_exact_f64(i) => {
            i.to_f64()
        }
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
    let Some(raw) = lookup_option(options, field) else {
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
    let value = option_f64(builtin, Some(options), field, default as f64)?;
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

pub(crate) fn option_string(
    options: Option<&StructValue>,
    field: &str,
    default: &str,
) -> BuiltinResult<String> {
    let Some(options) = options else {
        return Ok(default.to_string());
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(default.to_string());
    };
    match value {
        Value::String(s) => Ok(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(CharArray { data, rows: 1, .. }) => {
            Ok(data.iter().collect::<String>().to_ascii_lowercase())
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

fn finite_vec(name: &str, values: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(values)
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
        prepare_floating_value, value_to_real_vector, value_to_scalar,
    };
    use futures::executor::block_on;
    use runmat_builtins::{
        CharArray, Closure, IntegerStorage, StringArray, StructValue, Tensor, Value,
    };
    use std::sync::Arc;

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
        ))
        .expect_err("wide integer cannot cross exactly");
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn optimizer_integer_metadata_covers_every_runmat_only_role() {
        let capability_sets: [&[runmat_builtins::BuiltinIntegerCapabilityDescriptor]; 4] = [
            &super::super::fminbnd::FMINBND_INTEGER_CAPABILITIES,
            &super::super::fminunc::FMINUNC_INTEGER_CAPABILITIES,
            &super::super::fsolve::FSOLVE_INTEGER_CAPABILITIES,
            &super::super::fzero::FZERO_INTEGER_CAPABILITIES,
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
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_408_001,
        });
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
    }

    #[test]
    fn option_usize_reads_typed_integer_storage_exactly() {
        let max_iter = Tensor::new_integer(IntegerStorage::U16(vec![37]), vec![1, 1]).unwrap();
        let mut options = StructValue::new();
        options.insert("MaxIter", Value::Tensor(max_iter));

        assert_eq!(
            option_usize("optim_test", Some(&options), "MaxIter", 400).unwrap(),
            37
        );
    }

    #[test]
    fn option_usize_rejects_negative_typed_integer_storage_exactly() {
        let max_iter = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).unwrap();
        let mut options = StructValue::new();
        options.insert("MaxIter", Value::Tensor(max_iter));

        assert!(option_usize("optim_test", Some(&options), "MaxIter", 400).is_err());
    }

    #[test]
    fn option_usize_accepts_wide_typed_integer_storage_despite_poisoned_mirror() {
        let wide = 9_007_199_254_740_993_u64;
        let max_iter = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let mut options = StructValue::new();
        options.insert("MaxIter", Value::Tensor(max_iter));

        assert_eq!(
            option_usize("optim_test", Some(&options), "MaxIter", 400).unwrap(),
            wide as usize
        );
    }

    #[test]
    fn option_usize_rejects_fractional_and_unrepresentable_double_values() {
        let mut options = StructValue::new();
        options.insert("MaxIter", Value::Num(3.5));
        assert!(option_usize("optim_test", Some(&options), "MaxIter", 400).is_err());

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        options.insert("MaxIter", Value::Num(boundary));
        assert!(option_usize("optim_test", Some(&options), "MaxIter", 400).is_err());
    }

    #[test]
    fn callback_handle_canonicalizer_binds_function_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(42)
            })));
        let canonical = canonicalize_callback_handle(&Value::FunctionHandle("decay".to_string()));
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle {
                name: "decay".to_string(),
                function: 42,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_qualified_external_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(43)
            })));
        let canonical =
            canonicalize_callback_handle(&Value::ExternalFunctionHandle("pkg.decay".to_string()));
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle {
                name: "pkg.decay".to_string(),
                function: 43,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_keeps_malformed_external_handle_name_shaped() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg..decay").then_some(44)
            })));
        let raw = Value::ExternalFunctionHandle("pkg..decay".to_string());
        let canonical = canonicalize_callback_handle(&raw);
        assert_eq!(canonical, raw);
    }

    #[test]
    fn callback_handle_canonicalizer_binds_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(45)
            })));
        let canonical = canonicalize_callback_handle(&Value::String("@decay".to_string()));
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle {
                name: "decay".to_string(),
                function: 45,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_trims_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(145)
            })));
        let canonical = canonicalize_callback_handle(&Value::String("  @decay  ".to_string()));
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle {
                name: "decay".to_string(),
                function: 145,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_string_array_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(46)
            })));
        let canonical = canonicalize_callback_handle(&Value::StringArray(
            StringArray::new(vec!["@pkg.decay".to_string()], vec![1, 1]).expect("string array"),
        ));
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle {
                name: "pkg.decay".to_string(),
                function: 46,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_trims_string_array_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(146)
            })));
        let canonical = canonicalize_callback_handle(&Value::StringArray(
            StringArray::new(vec!["  @pkg.decay  ".to_string()], vec![1, 1]).expect("string array"),
        ));
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle {
                name: "pkg.decay".to_string(),
                function: 146,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_char_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(47)
            })));
        let canonical =
            canonicalize_callback_handle(&Value::CharArray(CharArray::new_row("@decay")));
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle {
                name: "decay".to_string(),
                function: 47,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_trims_char_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(147)
            })));
        let canonical =
            canonicalize_callback_handle(&Value::CharArray(CharArray::new_row("  @decay  ")));
        assert_eq!(
            canonical,
            Value::BoundFunctionHandle {
                name: "decay".to_string(),
                function: 147,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_name_only_closure_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(48)
            })));
        let raw = Value::Closure(Closure {
            function_name: "decay".to_string(),
            bound_function: None,
            captures: vec![Value::Num(9.0)],
        });
        let canonical = canonicalize_callback_handle(&raw);
        assert_eq!(
            canonical,
            Value::Closure(Closure {
                function_name: "decay".to_string(),
                bound_function: Some(48),
                captures: vec![Value::Num(9.0)],
            })
        );
    }

    #[test]
    fn callback_handle_canonicalizer_keeps_name_only_closure_without_resolver() {
        let raw = Value::Closure(Closure {
            function_name: "decay".to_string(),
            bound_function: None,
            captures: vec![Value::Num(9.0)],
        });
        let canonical = canonicalize_callback_handle(&raw);
        assert_eq!(canonical, raw);
    }
}
