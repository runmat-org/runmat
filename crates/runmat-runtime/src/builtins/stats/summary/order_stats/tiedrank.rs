use super::*;
use runmat_builtins::TiedRankMode;

fn flag(value: &Value) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.to_i128() {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(order_error(
                "tiedrank",
                "tiedrank: flags must be zero or one",
            )),
        };
    }
    let scalar = match value {
        Value::Num(value) => Some(*value),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    };
    match scalar {
        Some(0.0) => Ok(false),
        Some(1.0) => Ok(true),
        _ => Err(order_error(
            "tiedrank",
            "tiedrank: flags must be scalar zero or one",
        )),
    }
}

#[runtime_builtin(
    name = "tiedrank",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::order_stats::tiedrank"
)]
pub(crate) async fn tiedrank_builtin(
    value: Value,
    flags: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if flags.len() > 2 {
        return Err(order_error(
            "tiedrank",
            "tiedrank: expected at most two mode flags",
        ));
    }
    let first = flags.first().map(flag).transpose()?;
    let second = flags.get(1).map(flag).transpose()?;
    let mode = TiedRankMode::from_flags(first, second)
        .ok_or_else(|| order_error("tiedrank", "tiedrank: unsupported mode flag combination"))?;
    if super::is_real_typed_integer_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::TIEDRANK_DATA_EXTENSION,
            "tiedrank",
        )?;
    }
    let input = super::value_to_tensor("tiedrank", value).await?;
    let (ranks, tieadj) = super::tiedrank_tensor(input, mode)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![ranks])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![ranks, tieadj],
        )),
        None => Ok(ranks),
    }
}
