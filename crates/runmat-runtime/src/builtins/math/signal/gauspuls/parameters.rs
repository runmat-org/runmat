use super::*;

pub(super) async fn parse_params(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<GauspulsParams> {
    if rest.len() > 3 {
        return Err(gauspuls_error_with_detail(
            &GAUSPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    let mut params = default_params();
    if let Some(value) = rest
        .first()
        .filter(|value| !is_empty_default_placeholder(value))
    {
        params.fc = scalar_parameter(value, "FC").await?;
    }
    if let Some(value) = rest
        .get(1)
        .filter(|value| !is_empty_default_placeholder(value))
    {
        params.bw = scalar_parameter(value, "BW").await?;
    }
    if let Some(value) = rest
        .get(2)
        .filter(|value| !is_empty_default_placeholder(value))
    {
        params.bwr = scalar_parameter(value, "BWR").await?;
    }
    validate_params(params)
        .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, err.as_str()))
}

pub(super) fn is_empty_default_placeholder(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.is_empty())
}

pub(super) async fn scalar_parameter(value: &Value, label: &str) -> BuiltinResult<f64> {
    let host = crate::dispatcher::gather_if_needed_async(value)
        .await
        .map_err(|err| {
            gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, format!("{label}: {err}"))
        })?;
    if let Some(integer) = tensor::scalar_integer_value(&host) {
        if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
            return Err(gauspuls_error_with_detail(
                &GAUSPULS_ERROR_INVALID_PARAMETER,
                format!("{label}: integer value must be exactly representable as double"),
            ));
        }
    }
    scalar_f64_from_value_async(&host)
        .await
        .map_err(|err| {
            gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, format!("{label}: {err}"))
        })?
        .ok_or_else(|| {
            gauspuls_error_with_detail(
                &GAUSPULS_ERROR_INVALID_PARAMETER,
                format!("{label}: expected scalar"),
            )
        })
}

pub(super) fn text_scalar(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            )
        }
        Value::CharArray(array) if array.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(array.data()).map_err(|source| {
                let mut error = gauspuls_error_with_detail(
                    &GAUSPULS_ERROR_INVALID_PARAMETER,
                    source.to_string(),
                );
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}
