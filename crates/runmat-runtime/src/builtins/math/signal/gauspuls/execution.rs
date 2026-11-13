use super::admission::*;
use super::parameters::*;
use super::*;

pub(super) async fn gauspuls_value(t: Value, params: GauspulsParams) -> BuiltinResult<Value> {
    let tensor = sample_tensor(t).await?;
    gauspuls_tensor(tensor, params)
        .map(tensor_into_value)
        .map_err(map_gauspuls_tensor_error)
}

pub(super) async fn gauspuls_with_output_count(
    t: Value,
    params: GauspulsParams,
    out_count: usize,
) -> BuiltinResult<Value> {
    if out_count == 0 {
        let tensor = sample_tensor(t).await?;
        super::math::ensure_exact_integer_tensor_boundary(&tensor, "sample time")
            .map_err(map_gauspuls_tensor_error)?;
        return Ok(Value::output_list(Vec::new()));
    }
    if out_count <= 1 {
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![gauspuls_value(t, params).await?],
        ));
    }
    let (in_phase, quadrature, envelope) = gauspuls_components_value(t, params).await?;
    Ok(crate::output_count::output_list_with_padding(
        out_count,
        vec![in_phase, quadrature, envelope],
    ))
}

pub(super) async fn gauspuls_mode(
    mode: String,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !mode.eq_ignore_ascii_case("cutoff") {
        return Err(gauspuls_error_with_detail(
            &GAUSPULS_ERROR_INVALID_INPUT,
            format!("unsupported mode '{mode}'"),
        ));
    }
    if rest.len() > 4 {
        return Err(gauspuls_error_with_detail(
            &GAUSPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    ensure_gauspuls_control_extensions(rest.view())?;
    let params = parse_params(
        rest.view()
            .range(..rest.len().min(3))
            .expect("gauspuls prefix is in bounds"),
    )
    .await?;
    let tpe = match rest.get(3) {
        Some(value) if !is_empty_default_placeholder(value) => {
            validate_tpe(scalar_parameter(value, "TPE").await?).map_err(|err| {
                gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, err.as_str())
            })?
        }
        _ => DEFAULT_TPE,
    };
    let value = Value::Num(gauspuls_cutoff(params, tpe));
    if let Some(out_count) = crate::output_count::current_output_count() {
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![value],
        ));
    }
    Ok(value)
}

async fn sample_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    gauspuls_error_with_source(
                        &GAUSPULS_ERROR_INTERNAL,
                        "gpu gather failed",
                        source,
                    )
                })
        }
        Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_) => Err(gauspuls_error(&GAUSPULS_ERROR_INVALID_INPUT)),
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_INPUT, err)),
    }
}
pub(super) async fn gauspuls_components_value(
    value: Value,
    params: GauspulsParams,
) -> BuiltinResult<(Value, Value, Value)> {
    let tensor = sample_tensor(value).await?;
    let (in_phase, quadrature, envelope) =
        gauspuls_components_tensor(tensor, params).map_err(map_gauspuls_tensor_error)?;
    Ok((
        tensor_into_value(in_phase),
        tensor_into_value(quadrature),
        tensor_into_value(envelope),
    ))
}

pub(super) fn map_gauspuls_tensor_error(error: String) -> RuntimeError {
    let descriptor = if error.contains("exactly representable as double") {
        &GAUSPULS_ERROR_INVALID_INPUT
    } else {
        &GAUSPULS_ERROR_INTERNAL
    };
    gauspuls_error_with_detail(descriptor, error)
}
