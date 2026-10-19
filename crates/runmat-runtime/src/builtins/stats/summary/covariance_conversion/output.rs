use super::*;

pub(super) fn sigma_as_row(name: &'static str, sigma: Value) -> BuiltinResult<Value> {
    match sigma {
        Value::Tensor(tensor) => {
            let len = tensor.len();
            tensor
                .reshape(vec![1, len])
                .map(tensor::tensor_into_value)
                .map_err(|err| internal_error(name, format!("{name}: {err}")))
        }
        scalar => Ok(scalar),
    }
}

pub(super) fn output_values(
    kind: ConversionKind,
    name: &'static str,
    r: Value,
    sigma: Value,
) -> BuiltinResult<Value> {
    let (first, second) = match kind {
        ConversionKind::Corrcov => (r, sigma),
        ConversionKind::Cov2corr => (sigma_as_row(name, sigma)?, r),
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![first])),
        Some(2) => Ok(Value::output_list(vec![first, second])),
        Some(_) => Err(conversion_error(
            name,
            format!("{name}: too many output arguments; maximum is 2"),
        )),
        None => Ok(first),
    }
}
