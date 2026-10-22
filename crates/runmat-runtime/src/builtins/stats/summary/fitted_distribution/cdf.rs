use super::*;

pub(in crate::builtins::stats::summary) async fn evaluate_cdf_entry(
    distribution: Value,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let tail = if rest
        .last()
        .map(keyword_of)
        .transpose()?
        .flatten()
        .is_some_and(|value| value.eq_ignore_ascii_case("upper"))
    {
        CdfTail::Upper
    } else {
        CdfTail::Lower
    };
    let rest = if matches!(tail, CdfTail::Upper) {
        let end = rest.len() - 1;
        rest.into_range(..end)?
    } else {
        rest
    };
    let selection =
        admission::evaluation(FittedOperation::Cumulative, &distribution, &x, rest.view())?;
    evaluate_cdf(selection, x, rest, tail).await
}

#[derive(Clone, Copy)]
pub(super) enum CdfOutputPrecision {
    Double,
    Single,
}

pub(super) async fn evaluate_cdf(
    selection: DistributionSelection,
    input: Value,
    rest: crate::call::BuiltinArguments,
    tail: CdfTail,
) -> BuiltinResult<Value> {
    let kind = match selection {
        DistributionSelection::Fitted(fit) => {
            ensure_cdf_extensions(&input, crate::call::FunctionArgumentView::from_slice(&[]))?;
            let precision = cdf_output_precision(&[&input]);
            let residency = EvaluationResidency::capture(&[&input]);
            let x = cdf_value_to_tensor(input).await?;
            ensure_exact_cdf_integer_boundary(&x, "x")?;
            let shape = x.shape().to_vec();
            let data = tensor::tensor_into_values_f64(x)
                .into_iter()
                .map(|value| cdf_scalar_with_tail(&fit, value, tail))
                .collect();
            return finish_cdf(shape, data, precision, residency);
        }
        DistributionSelection::Named(kind) => kind,
    };
    ensure_cdf_extensions(&input, rest.view())?;
    let mut original_inputs = Vec::with_capacity(rest.len() + 1);
    original_inputs.push(&input);
    original_inputs.extend(rest.iter());
    let precision = cdf_output_precision(&original_inputs);
    let residency = EvaluationResidency::capture(&original_inputs);
    let x = cdf_value_to_tensor(input).await?;
    ensure_exact_cdf_integer_boundary(&x, "x")?;
    let parameters = parse_cdf_parameter_tensors(kind, rest.view()).await?;
    for (index, parameter) in parameters.iter().enumerate() {
        ensure_exact_cdf_integer_boundary(parameter, kind.parameter_names()[index])?;
    }
    let mut tensors = Vec::with_capacity(parameters.len() + 1);
    tensors.push(&x);
    tensors.extend(parameters.iter());
    let (mut values, shape) = broadcast_tensors_for(CDF_NAME, &tensors)?;
    let x_values = values.remove(0);
    let mut data = Vec::with_capacity(x_values.len());
    for index in 0..x_values.len() {
        let fit = FittedDistribution {
            kind,
            parameters: values.iter().map(|parameter| parameter[index]).collect(),
            nlogl: f64::NAN,
            observations: f64::NAN,
        };
        data.push(cdf_scalar_with_tail(&fit, x_values[index], tail));
    }
    finish_cdf(shape, data, precision, residency)
}

pub(super) fn ensure_cdf_extensions(
    input: &Value,
    parameters: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if is_typed_integer_value(input) {
        crate::compatibility::ensure_builtin_extension_enabled(&CDF_INTEGER_X_EXTENSION, CDF_NAME)?;
    }
    if parameters.iter().any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CDF_INTEGER_PARAMETER_EXTENSION,
            CDF_NAME,
        )?;
    }
    if is_logical_value(input) || parameters.iter().any(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CDF_LOGICAL_INPUT_EXTENSION,
            CDF_NAME,
        )?;
    }
    Ok(())
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

pub(super) fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F32))
}

pub(super) fn cdf_output_precision(inputs: &[&Value]) -> CdfOutputPrecision {
    if inputs.iter().any(|value| is_single_value(value)) {
        CdfOutputPrecision::Single
    } else {
        CdfOutputPrecision::Double
    }
}

pub(super) async fn cdf_value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| with_fitted_context(CDF_NAME, err))?;
    tensor::value_into_tensor_for(CDF_NAME, gathered)
        .map_err(|_| invalid_for(CDF_NAME, "cdf: expected numeric input"))
}

pub(super) async fn parse_cdf_parameter_tensors(
    kind: DistributionKind,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<Tensor>> {
    match kind {
        DistributionKind::Normal | DistributionKind::Lognormal => {
            match (rest.len(), rest.get(0), rest.get(1)) {
                (0, _, _) => Ok(vec![scalar_tensor(0.0), scalar_tensor(1.0)]),
                (1, Some(mu), _) => Ok(vec![
                    cdf_value_to_tensor(mu.clone()).await?,
                    scalar_tensor(1.0),
                ]),
                (2, Some(mu), Some(sigma)) => Ok(vec![
                    cdf_value_to_tensor(mu.clone()).await?,
                    cdf_value_to_tensor(sigma.clone()).await?,
                ]),
                _ => Err(invalid_for(
                    CDF_NAME,
                    format!(
                        "cdf: {} distribution expects x, x and mu, or x, mu, sigma",
                        kind.canonical_name()
                    ),
                )),
            }
        }
        DistributionKind::Exponential | DistributionKind::Poisson => {
            if rest.len() != 1 {
                return Err(invalid_for(
                    CDF_NAME,
                    format!(
                        "cdf: {} distribution expects one parameter",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![cdf_value_to_tensor(rest[0].clone()).await?])
        }
        DistributionKind::Gamma | DistributionKind::Weibull => {
            if rest.len() != 2 {
                return Err(invalid_for(
                    CDF_NAME,
                    format!(
                        "cdf: {} distribution expects two parameters",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![
                cdf_value_to_tensor(rest[0].clone()).await?,
                cdf_value_to_tensor(rest[1].clone()).await?,
            ])
        }
    }
}

pub(super) fn ensure_exact_cdf_integer_boundary(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for integer in tensor
        .integer_storage()
        .expect("integer storage checked above")
        .exact_values()
    {
        if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
            return Err(invalid_for(
                CDF_NAME,
                format!("cdf: integer {name} values must be exactly representable as double"),
            ));
        }
    }
    Ok(())
}

pub(super) fn finish_cdf(
    shape: Vec<usize>,
    data: Vec<f64>,
    precision: CdfOutputPrecision,
    residency: EvaluationResidency,
) -> BuiltinResult<Value> {
    let tensor = match precision {
        CdfOutputPrecision::Double => Tensor::new(data, shape),
        CdfOutputPrecision::Single => {
            Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
        }
    }
    .map_err(|err| internal_for(CDF_NAME, format!("cdf: {err}")))?;
    if let Some(source) = residency.source() {
        let protected = residency.protected();
        let provider = runmat_accelerate_api::provider_for_handle(source).ok_or_else(|| {
            invalid_for(
                CDF_NAME,
                "cdf: no acceleration provider registered for GPU output",
            )
        })?;
        let handle = gpu_helpers::upload_tensor_protecting_typed(provider, &tensor, &protected)
            .map_err(|err| {
                crate::runtime_error::with_provider_execution_source(
                    invalid_for(CDF_NAME, format!("cdf: {err}")),
                    runmat_accelerate_api::ProviderExecutionError::from_source(err),
                )
            })?;
        let handle =
            gpu_helpers::propagate_owned_output_provenance(handle, protected.iter().copied())
                .map_err(|error| with_fitted_context(CDF_NAME, error))?;
        return gpu_helpers::resident_gpu_value(handle)
            .map_err(|error| with_fitted_context(CDF_NAME, error));
    }
    match precision {
        CdfOutputPrecision::Double => Ok(tensor::tensor_into_value(tensor)),
        CdfOutputPrecision::Single => Ok(Value::Tensor(tensor)),
    }
}
