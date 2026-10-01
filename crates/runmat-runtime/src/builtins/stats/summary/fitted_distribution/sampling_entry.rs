use super::*;

pub(super) async fn sample_distribution(
    distribution: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    admission::random(&distribution, &rest)?;
    ensure_random_extensions(&distribution, &rest).await?;
    let gpu_source = random_gpu_source(&distribution, &rest)?;
    let (fit, shape) = if matches!(distribution, Value::Object(_)) {
        (
            distribution_from_value(&distribution)?,
            parse_shape_args(&rest).await?,
        )
    } else {
        parse_named_random_args(distribution, rest).await?
    };
    let len = tensor::element_count(&shape);
    let data = random_samples(&fit, len)?;
    finish_random(shape, data, gpu_source)
}

pub(super) fn random_gpu_source(
    distribution: &Value,
    rest: &[Value],
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let parameter_count = if matches!(distribution, Value::Object(_)) {
        0
    } else {
        parse_distribution_name_for(RANDOM_NAME, distribution)?
            .parameter_names()
            .len()
    };
    gpu_helpers::select_resident_output_source(
        rest.iter()
            .take(parameter_count)
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
        RANDOM_NAME,
    )
}

pub(super) fn finish_random(
    shape: Vec<usize>,
    data: Vec<f64>,
    gpu_source: Option<GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let tensor = Tensor::new(data, shape)
        .map_err(|err| internal_for(RANDOM_NAME, format!("random: {err}")))?;
    let Some(source) = gpu_source else {
        return Ok(tensor::tensor_into_value(tensor));
    };
    let restored =
        gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(tensor), RANDOM_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&source)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(internal_for(
            RANDOM_NAME,
            "random: provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

pub(super) async fn ensure_random_extensions(
    distribution: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    for value in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, RANDOM_NAME)?;
    }
    let parameter_count = if matches!(distribution, Value::Object(_)) {
        0
    } else {
        parse_distribution_name_for(RANDOM_NAME, distribution)?
            .parameter_names()
            .len()
    };
    for (index, value) in rest.iter().enumerate() {
        if index < parameter_count {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &RANDOM_INTEGER_PARAMETER_EXTENSION,
                RANDOM_NAME,
                "distribution parameter",
            )
            .await?;
        } else if crate::builtins::common::validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &RANDOM_INTEGER_SIZE_EXTENSION,
                RANDOM_NAME,
            )?;
        }
    }
    Ok(())
}
