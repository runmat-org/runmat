use runmat_builtins::{Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::plotting::plotting_error;
use crate::BuiltinResult;

#[derive(Clone, Debug)]
pub enum LimitCommand {
    Query,
    Set(Option<(f64, f64)>),
}

pub fn parse_limit_command(builtin: &'static str, args: &[Value]) -> BuiltinResult<LimitCommand> {
    if args.is_empty() {
        return Ok(LimitCommand::Query);
    }
    if args.len() > 1 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected at most one argument"),
        ));
    }
    let arg = &args[0];
    if let Some(text) = crate::builtins::plotting::style::value_as_string(arg) {
        let normalized = text.trim().to_ascii_lowercase();
        return match normalized.as_str() {
            "auto" | "tight" => Ok(LimitCommand::Set(None)),
            "manual" => Ok(LimitCommand::Query),
            _ => Err(plotting_error(
                builtin,
                format!("{builtin}: unsupported mode `{normalized}`"),
            )),
        };
    }
    let limits = limits_from_value(arg, builtin)?;
    Ok(LimitCommand::Set(Some(limits)))
}

pub fn limits_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<(f64, f64)> {
    let tensor =
        Tensor::try_from(value).map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?;
    let data = tensor::tensor_values_f64(&tensor);
    if data.len() != 2 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: expected a 2-element numeric vector"),
        ));
    }
    let lo = data[0];
    let hi = data[1];
    if !lo.is_finite() || !hi.is_finite() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: limits must be finite"),
        ));
    }
    if hi <= lo {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: lower limit must be less than upper limit"),
        ));
    }
    Ok((lo, hi))
}

pub fn limit_value(limits: Option<(f64, f64)>) -> Value {
    let data = match limits {
        Some((lo, hi)) => vec![lo, hi],
        None => vec![f64::NAN, f64::NAN],
    };
    Value::Tensor(Tensor::new(data, vec![1, 2]).expect("limit vector shape"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn limits_read_typed_integer_storage() {
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![10, 20]), vec![1, 2]).expect("limits"),
        );

        assert_eq!(
            limits_from_value(&value, "xlim").expect("limits"),
            (10.0, 20.0)
        );
    }
}
