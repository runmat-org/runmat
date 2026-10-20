use super::*;

pub(super) async fn parse_named_eval_parameters(
    builtin: &'static str,
    kind: DistributionKind,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<f64>> {
    let tensors = parse_parameter_tensors(builtin, kind, rest).await?;
    let (values, _shape) = broadcast_tensors_for(builtin, &tensors.iter().collect::<Vec<_>>())?;
    if values.iter().all(|values| values.len() == 1) {
        return Ok(values.into_iter().map(|values| values[0]).collect());
    }
    Err(invalid_for(
        builtin,
        format!("{builtin}: named distribution parameters must be scalar for this overload"),
    ))
}

pub(super) async fn parse_named_random_args(
    kind: DistributionKind,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<(FittedDistribution, Vec<usize>)> {
    let parameter_count = kind.parameter_names().len();
    if rest.len() < parameter_count {
        return Err(invalid_for(
            RANDOM_NAME,
            format!(
                "random: {} distribution requires {} parameter argument(s)",
                kind.canonical_name(),
                parameter_count
            ),
        ));
    }
    let parameter_values = rest.view().range(..parameter_count)?;
    let shape_args = rest.view().range(parameter_count..)?;
    let params = parse_named_eval_parameters(RANDOM_NAME, kind, parameter_values).await?;
    let fit = FittedDistribution {
        kind,
        parameters: params,
        nlogl: f64::NAN,
        observations: f64::NAN,
    };
    let shape = parse_shape_args(shape_args).await?;
    Ok((fit, shape))
}

pub(super) async fn parse_parameter_tensors(
    builtin: &'static str,
    kind: DistributionKind,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<Tensor>> {
    match kind {
        DistributionKind::Normal | DistributionKind::Lognormal => {
            match (rest.len(), rest.get(0), rest.get(1)) {
                (0, _, _) => Ok(vec![scalar_tensor(0.0), scalar_tensor(1.0)]),
                (1, Some(mu), _) => Ok(vec![
                    value_to_tensor(builtin, mu.clone()).await?,
                    scalar_tensor(1.0),
                ]),
                (2, Some(mu), Some(sigma)) => Ok(vec![
                    value_to_tensor(builtin, mu.clone()).await?,
                    value_to_tensor(builtin, sigma.clone()).await?,
                ]),
                _ => Err(invalid_for(
                    builtin,
                    format!(
                        "{builtin}: {} distribution expects x, x and mu, or x, mu, sigma",
                        kind.canonical_name()
                    ),
                )),
            }
        }
        DistributionKind::Exponential | DistributionKind::Poisson => {
            if rest.len() != 1 {
                return Err(invalid_for(
                    builtin,
                    format!(
                        "{builtin}: {} distribution expects one parameter",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![value_to_tensor(builtin, rest[0].clone()).await?])
        }
        DistributionKind::Gamma | DistributionKind::Weibull => {
            if rest.len() != 2 {
                return Err(invalid_for(
                    builtin,
                    format!(
                        "{builtin}: {} distribution expects two parameters",
                        kind.canonical_name()
                    ),
                ));
            }
            Ok(vec![
                value_to_tensor(builtin, rest[0].clone()).await?,
                value_to_tensor(builtin, rest[1].clone()).await?,
            ])
        }
    }
}

pub(super) async fn parse_fit_options(frequency: Option<&Value>) -> BuiltinResult<FitOptions> {
    let mut options = FitOptions::default();
    if let Some(frequency) = frequency {
        let tensor = fitdist_value_to_tensor(frequency.clone(), "Frequency").await?;
        options.frequency = Some(tensor::tensor_into_values_f64(tensor));
    }
    Ok(options)
}

pub(super) fn ensure_fitdist_integer_extensions(
    data: &Value,
    frequency: Option<&Value>,
) -> BuiltinResult<()> {
    if is_typed_integer_value(data) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FITDIST_INTEGER_DATA_EXTENSION,
            FITDIST_NAME,
        )?;
    }
    if frequency.is_some_and(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FITDIST_INTEGER_FREQUENCY_EXTENSION,
            FITDIST_NAME,
        )?;
    }
    Ok(())
}

pub(super) fn ensure_fitdist_resident_fallback_extension(
    data: &Value,
    frequency: Option<&Value>,
) -> BuiltinResult<()> {
    if matches!(data, Value::GpuTensor(_))
        || frequency.is_some_and(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FITDIST_RESIDENT_FALLBACK_EXTENSION,
            FITDIST_NAME,
        )?;
    }
    Ok(())
}

pub(super) async fn fitdist_value_to_tensor(value: Value, role: &str) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| with_fitted_context(FITDIST_NAME, err))?;
    let tensor = tensor::value_into_tensor_for(FITDIST_NAME, gathered)
        .map_err(|_| invalid("fitdist: expected numeric input"))?;
    ensure_exact_fitdist_integer_boundary(&tensor, role)?;
    tensor::integer_tensor_to_f64(tensor).map_err(|err| invalid(format!("fitdist: {err}")))
}

pub(super) fn ensure_exact_fitdist_integer_boundary(
    tensor: &Tensor,
    role: &str,
) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for integer in tensor
        .integer_storage()
        .expect("integer storage checked above")
        .exact_values()
    {
        if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
            return Err(invalid(format!(
                "fitdist: integer {role} values must be exactly representable as double"
            )));
        }
    }
    Ok(())
}

pub(super) async fn value_to_tensor(name: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| with_fitted_context(name, err))?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|_| invalid_for(name, format!("{name}: expected numeric input")))?;
    tensor::integer_tensor_to_f64(tensor).map_err(|err| invalid_for(name, format!("{name}: {err}")))
}

pub(super) fn scalar_tensor(value: f64) -> Tensor {
    Tensor::new(vec![value], vec![1, 1]).expect("scalar tensor shape is valid")
}

pub(super) fn broadcast_tensors_for(
    builtin: &'static str,
    inputs: &[&Tensor],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let Some(first) = inputs.first() else {
        return Ok((Vec::new(), vec![1, 1]));
    };
    let mut shape = first.shape().to_vec();
    for tensor in inputs.iter().skip(1) {
        shape = broadcast::broadcast_shapes(builtin, &shape, tensor.shape())
            .map_err(|err| invalid_for(builtin, err))?;
    }
    let mut values = Vec::with_capacity(inputs.len());
    for tensor in inputs {
        values.push(broadcast_tensor_to(builtin, tensor, &shape)?);
    }
    Ok((values, shape))
}

pub(super) fn broadcast_tensor_to(
    builtin: &'static str,
    tensor: &Tensor,
    out_shape: &[usize],
) -> BuiltinResult<Vec<f64>> {
    let len = tensor::element_count(out_shape);
    if len == 0 {
        return Ok(Vec::new());
    }
    let in_shape = broadcast::align_shape(tensor.shape(), out_shape.len());
    let strides = broadcast::compute_strides(&in_shape);
    let mut out = Vec::with_capacity(len);
    let input_len = tensor::tensor_element_len(tensor);
    for idx in 0..len {
        let source_idx = broadcast::broadcast_index(idx, out_shape, &in_shape, &strides);
        if source_idx >= input_len {
            return Err(invalid_for(
                builtin,
                format!("{builtin}: tensor data does not match tensor shape"),
            ));
        }
        out.push(tensor::tensor_value_f64(tensor, source_idx));
    }
    Ok(out)
}
