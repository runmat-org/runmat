use super::*;

pub(super) fn normal_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:InvalidArgument"))
        .build()
}

pub(super) fn with_distribution_context(name: &str, mut error: RuntimeError) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(name);
    }
    error
}

pub(super) fn integer_numeric_scalar_is_exact_f64(value: NumericScalar) -> Option<bool> {
    use runmat_value::IntValue;
    let integer = match value {
        NumericScalar::I8(value) => IntValue::I8(value),
        NumericScalar::I16(value) => IntValue::I16(value),
        NumericScalar::I32(value) => IntValue::I32(value),
        NumericScalar::I64(value) => IntValue::I64(value),
        NumericScalar::U8(value) => IntValue::U8(value),
        NumericScalar::U16(value) => IntValue::U16(value),
        NumericScalar::U32(value) => IntValue::U32(value),
        NumericScalar::U64(value) => IntValue::U64(value),
        _ => return None,
    };
    Some(crate::builtins::common::validation::integer_is_exact_f64(
        &integer,
    ))
}

pub(super) fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

pub(super) fn ensure_normal_extensions(
    name: &str,
    first: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    integer_roles: [&BuiltinExtensionDescriptor; 3],
    logical: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    for (value, extension) in std::iter::once(first).chain(rest.iter()).zip(integer_roles) {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, name)?;
        }
    }
    if std::iter::once(first)
        .chain(rest.iter())
        .any(is_logical_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(logical, name)?;
    }
    Ok(())
}

pub(super) fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F32))
}

pub(super) fn ensure_exact_integer_boundary(
    name: &str,
    tensor: &Tensor,
    role: &str,
) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for index in 0..tensor.len() {
        if tensor
            .numeric_value_at(index)
            .and_then(integer_numeric_scalar_is_exact_f64)
            == Some(false)
        {
            return Err(normal_error(
                name,
                format!("{name}: integer {role} values must be exactly representable as double"),
            ));
        }
    }
    Ok(())
}

pub(super) async fn normal_value_to_tensor(
    name: &str,
    role: &str,
    value: Value,
) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|error| with_distribution_context(name, error))?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
    ensure_exact_integer_boundary(name, &tensor, role)?;
    tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| normal_error(name, format!("{name}: {err}")))
}

#[derive(Clone, Copy)]
enum NormalOutputPrecision {
    Double,
    Single,
}

pub(super) struct NormalOutputPlan {
    precision: NormalOutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
}

impl NormalOutputPlan {
    pub(super) fn inspect(
        name: &str,
        first: &Value,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        let precision = if std::iter::once(first)
            .chain(rest.iter())
            .any(is_single_value)
        {
            NormalOutputPrecision::Single
        } else {
            NormalOutputPrecision::Double
        };
        let gpu_source = gpu_helpers::select_resident_output_source(
            std::iter::once(first)
                .chain(rest.iter())
                .filter_map(|value| match value {
                    Value::GpuTensor(handle) => Some(handle.clone()),
                    _ => None,
                }),
            name,
        )?;
        Ok(Self {
            precision,
            gpu_source,
        })
    }

    pub(super) fn finish(
        self,
        name: &str,
        shape: Vec<usize>,
        data: Vec<f64>,
    ) -> BuiltinResult<Value> {
        let tensor = match self.precision {
            NormalOutputPrecision::Double => Tensor::new(data, shape),
            NormalOutputPrecision::Single => {
                Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
            }
        }
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
        let Some(source) = self.gpu_source else {
            return Ok(match self.precision {
                NormalOutputPrecision::Double => tensor::tensor_into_value(tensor),
                NormalOutputPrecision::Single => Value::Tensor(tensor),
            });
        };
        let restored =
            gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(tensor), name)
                .map_err(|error| with_distribution_context(name, error))?;
        if runmat_accelerate_api::handle_is_explicit(&source)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(normal_error(
                name,
                format!("{name}: provider cannot preserve explicit gpuArray output residency"),
            ));
        }
        Ok(match (self.precision, restored) {
            (NormalOutputPrecision::Double, Value::Tensor(tensor)) => {
                tensor::tensor_into_value(tensor)
            }
            (_, value) => value,
        })
    }
}

pub(super) async fn value_to_tensor(name: &str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|error| with_distribution_context(name, error))?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| normal_error(name, format!("{name}: {err}")))?;
    tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| normal_error(name, format!("{name}: {err}")))
}
