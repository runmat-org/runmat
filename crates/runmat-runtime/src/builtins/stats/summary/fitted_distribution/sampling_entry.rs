use super::*;

pub(in crate::builtins::stats::summary) async fn sample_distribution(
    distribution: Value,
    rest: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    check_fitted_cancellation(runtime, FittedOperation::Sampling)?;
    let selection = DistributionSelection::from_value(FittedOperation::Sampling, &distribution)?;
    admission::random(&selection, rest.view())?;
    ensure_random_extensions(&selection, rest.view()).await?;
    let gpu_source = random_gpu_source(&selection, rest.view())?;
    let (fit, shape) = match selection {
        DistributionSelection::Fitted(fit) => (fit, parse_shape_args(rest.view()).await?),
        DistributionSelection::Named(kind) => parse_named_random_args(kind, rest).await?,
    };
    let len = checked_output_len(&shape)?;
    let mut turn = crate::builtins::common::random::acquire_turn(runtime, RANDOM_NAME).await?;
    let data = random_samples(&mut turn, &fit, len, || {
        check_fitted_cancellation(runtime, FittedOperation::Sampling)
    })?;
    finish_random(shape, data, gpu_source)
}

pub(super) fn random_gpu_source(
    selection: &DistributionSelection,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        rest.iter()
            .take(selection.parameter_count())
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
        RANDOM_NAME,
    )
    .map_err(|error| with_fitted_context(RANDOM_NAME, error))
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
        gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(tensor), RANDOM_NAME)
            .map_err(|error| with_fitted_context(RANDOM_NAME, error))?;
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
    selection: &DistributionSelection,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for value in rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, RANDOM_NAME)?;
    }
    let parameter_count = selection.parameter_count();
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
