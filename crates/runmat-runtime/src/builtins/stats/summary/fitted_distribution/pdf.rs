use super::*;

pub(super) async fn evaluate_pdf_entry(
    distribution: Value,
    x: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    admission::evaluation(
        PDF_NAME,
        &distribution,
        &x,
        &rest,
        admission::EvaluationKind::Density,
    )?;
    ensure_pdf_integer_extensions(&distribution, &x, &rest).await?;
    evaluate_pdf(distribution, x, rest).await
}

pub(super) async fn ensure_pdf_integer_extensions(
    distribution: &Value,
    input: &Value,
    parameters: &[Value],
) -> BuiltinResult<()> {
    if is_typed_integer_value(input) {
        crate::compatibility::ensure_builtin_extension_enabled(&PDF_INTEGER_X_EXTENSION, PDF_NAME)?;
        ensure_exact_pdf_integer_boundary(input, "x").await?;
    }
    if !matches!(distribution, Value::Object(_)) {
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
    distribution: Value,
    input: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if matches!(distribution, Value::Object(_)) {
        if !rest.is_empty() {
            return Err(invalid_for(
                PDF_NAME,
                "pdf: fitted distribution object form accepts exactly two inputs",
            ));
        }
        let precision = cdf_output_precision(&[&input]);
        let gpu_source = first_gpu_source(&[&input]);
        let fit = distribution_from_value_for(FittedOperation::Density, &distribution)?;
        let x = value_to_tensor(PDF_NAME, input).await?;
        let shape = x.shape.clone();
        let data = tensor::tensor_into_values_f64(x)
            .into_iter()
            .map(|value| pdf_scalar(&fit, value))
            .collect();
        return finish_pdf(shape, data, precision, gpu_source);
    }

    let kind = parse_distribution_name_for(PDF_NAME, &distribution)?;
    let mut original_inputs = Vec::with_capacity(rest.len() + 1);
    original_inputs.push(&input);
    original_inputs.extend(rest.iter());
    let precision = cdf_output_precision(&original_inputs);
    let gpu_source = first_gpu_source(&original_inputs);
    let x = value_to_tensor(PDF_NAME, input).await?;
    let parameters = parse_parameter_tensors(PDF_NAME, kind, rest).await?;
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
    finish_pdf(shape, data, precision, gpu_source)
}

pub(super) fn finish_pdf(
    shape: Vec<usize>,
    data: Vec<f64>,
    precision: CdfOutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
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
    let Some(source) = gpu_source else {
        return Ok(output);
    };
    let restored = gpu_helpers::restore_class_preserving_value(&source, output, PDF_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&source)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(invalid_for(
            PDF_NAME,
            "pdf: provider cannot preserve explicit gpuArray output residency and precision",
        ));
    }
    Ok(restored)
}
