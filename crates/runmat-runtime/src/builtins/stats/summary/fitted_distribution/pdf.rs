use super::*;

pub(in crate::builtins::stats::summary) async fn evaluate_pdf_entry(
    distribution: Value,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let selection =
        admission::evaluation(FittedOperation::Density, &distribution, &x, rest.view())?;
    ensure_pdf_integer_extensions(&selection, &x, rest.view()).await?;
    evaluate_pdf(selection, x, rest).await
}

pub(super) async fn ensure_pdf_integer_extensions(
    selection: &DistributionSelection,
    input: &Value,
    parameters: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if is_typed_integer_value(input) {
        crate::compatibility::ensure_builtin_extension_enabled(&PDF_INTEGER_X_EXTENSION, PDF_NAME)?;
        ensure_exact_pdf_integer_boundary(input, "x").await?;
    }
    if matches!(selection, DistributionSelection::Named(_)) {
        for parameter in parameters
            .iter()
            .filter(|value| is_typed_integer_value(value))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &PDF_INTEGER_PARAMETER_EXTENSION,
                PDF_NAME,
            )?;
            ensure_exact_pdf_integer_boundary(parameter, "distribution parameter").await?;
        }
    }
    Ok(())
}

pub(super) async fn ensure_exact_pdf_integer_boundary(
    value: &Value,
    role: &str,
) -> BuiltinResult<()> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value).await? {
        return Err(invalid_for(
            PDF_NAME,
            format!("pdf: integer {role} values must be exactly representable as double"),
        ));
    }
    Ok(())
}

pub(super) async fn evaluate_pdf(
    selection: DistributionSelection,
    input: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let kind = match selection {
        DistributionSelection::Fitted(fit) => {
            let precision = cdf_output_precision(&[&input]);
            let residency = EvaluationResidency::capture(&[&input]);
            let x = value_to_tensor(PDF_NAME, input).await?;
            let shape = x.shape().to_vec();
            let data = tensor::tensor_into_values_f64(x)
                .into_iter()
                .map(|value| pdf_scalar(&fit, value))
                .collect();
            return finish_pdf(shape, data, precision, residency);
        }
        DistributionSelection::Named(kind) => kind,
    };
    let mut original_inputs = Vec::with_capacity(rest.len() + 1);
    original_inputs.push(&input);
    original_inputs.extend(rest.iter());
    let precision = cdf_output_precision(&original_inputs);
    let residency = EvaluationResidency::capture(&original_inputs);
    let x = value_to_tensor(PDF_NAME, input).await?;
    let parameters = parse_parameter_tensors(PDF_NAME, kind, rest.view()).await?;
    let mut tensors = Vec::with_capacity(parameters.len() + 1);
    tensors.push(&x);
    tensors.extend(parameters.iter());
    let (mut values, shape) = broadcast_tensors_for(PDF_NAME, &tensors)?;
    let x_values = values.remove(0);
    let mut data = Vec::with_capacity(x_values.len());
    for index in 0..x_values.len() {
        let fit = FittedDistribution {
            kind,
            parameters: values.iter().map(|parameter| parameter[index]).collect(),
            nlogl: f64::NAN,
            observations: f64::NAN,
        };
        data.push(pdf_scalar(&fit, x_values[index]));
    }
    finish_pdf(shape, data, precision, residency)
}

pub(super) fn finish_pdf(
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
    .map_err(|error| internal_for(PDF_NAME, format!("pdf: {error}")))?;
    let output = match precision {
        CdfOutputPrecision::Double => tensor::tensor_into_value(tensor),
        CdfOutputPrecision::Single => Value::Tensor(tensor),
    };
    let Some(source) = residency.source() else {
        return Ok(output);
    };
    let protected = residency.protected();
    let restored = gpu_helpers::restore_class_preserving_value_protecting(
        source, output, PDF_NAME, &protected,
    )
    .map_err(|error| with_fitted_context(PDF_NAME, error))?;
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(invalid_for(
            PDF_NAME,
            "pdf: provider cannot preserve explicit gpuArray output residency and precision",
        ));
    }
    Ok(restored)
}
