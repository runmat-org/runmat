//! Case-insensitive prefix comparison for host text.

use runmat_builtins::{STRNCMPI_DESCRIPTOR, STRNCMPI_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericScalar, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::contains_numeric_or_resident_text_input;
use crate::builtins::strings::core::comparison_observation::{
    host_string_lengths, ComparisonMode, PrefixCarrier, PrefixFacts, ScalarComparisonWorkload,
};
use crate::builtins::strings::core::compat::{
    broadcast_flat_index, broadcast_shape, compat_error, logical_value, TextList,
};
use crate::{gather_if_needed_async, BuiltinResult};

#[runtime_builtin(
    name = "strncmpi",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::strncmpi",
    requested_outputs = true,
    call_observation = true
)]
pub(crate) async fn strncmpi_builtin(
    requested_outputs: Option<usize>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    a: Value,
    b: Value,
    n: Value,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &STRNCMPI_DESCRIPTOR,
        &STRNCMPI_ERROR_TOO_MANY_OUTPUTS,
        "strncmpi",
    )?;
    if contains_numeric_or_resident_text_input(&a) || contains_numeric_or_resident_text_input(&b) {
        return Ok(Value::Bool(false));
    }
    let host_lengths = host_string_lengths(&a, &b);
    let prefix_carrier = PrefixCarrier::immediate(&n);
    let a = gather_if_needed_async(&a)
        .await
        .map_err(|err| map_control_flow_with_builtin(err, "strncmpi"))?;
    let b = gather_if_needed_async(&b)
        .await
        .map_err(|err| map_control_flow_with_builtin(err, "strncmpi"))?;
    let n = gather_if_needed_async(&n)
        .await
        .map_err(|err| map_control_flow_with_builtin(err, "strncmpi"))?;
    let n = parse_prefix_count(&n)?;
    let left = TextList::from_value(a, "strncmpi")?;
    let right = TextList::from_value(b, "strncmpi")?;
    let shape = broadcast_shape(&left.shape, &right.shape, "strncmpi")?;
    let total: usize = shape.iter().product();
    if total == 1 && left.shape == [1, 1] && right.shape == [1, 1] {
        if let (Some((left_utf8_bytes, right_utf8_bytes)), Some(carrier), Some(observation)) =
            (host_lengths, prefix_carrier, observation)
        {
            observation.admit(ScalarComparisonWorkload {
                left_utf8_bytes,
                right_utf8_bytes,
                mode: ComparisonMode::PrefixInsensitive,
                prefix: Some(PrefixFacts {
                    limit_chars: n,
                    carrier,
                }),
                requested_outputs,
            });
        }
    }
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let li = broadcast_flat_index(idx, &shape, &left.shape);
        let ri = broadcast_flat_index(idx, &shape, &right.shape);
        let matched = match (&left.items[li], &right.items[ri]) {
            (Some(a), Some(b)) => prefix_eq_ignore_case(a, b, n),
            _ => false,
        };
        out.push(u8::from(matched));
    }
    logical_value(out, shape, "strncmpi")
}

fn parse_prefix_count(value: &Value) -> BuiltinResult<usize> {
    let invalid = || compat_error("strncmpi", "strncmpi: expected an integer scalar count");
    match value {
        Value::Int(value) => {
            if value.try_to_i64().is_some_and(|value| value < 0) {
                Ok(0)
            } else {
                value.try_to_usize().ok_or_else(invalid)
            }
        }
        Value::Num(value) => {
            if !value.is_finite() || value.fract() != 0.0 {
                return Err(invalid());
            }
            if *value < 0.0 {
                return Ok(0);
            }
            if *value > usize::MAX as f64 || (usize::BITS == 64 && *value == usize::MAX as f64) {
                return Err(invalid());
            }
            Ok(*value as usize)
        }
        Value::Bool(value) => Ok(usize::from(*value)),
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor
            .numeric_value_at(0)
            .expect("scalar tensor has one numeric value")
        {
            NumericScalar::F64(value) => parse_prefix_count(&Value::Num(value)),
            NumericScalar::F32(value) => parse_prefix_count(&Value::Num(f64::from(value))),
            value => parse_prefix_count(&Value::Int(
                value
                    .into_int_value()
                    .expect("non-floating scalar is integer"),
            )),
        },
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(usize::from(array.data[0] != 0)),
        _ => Err(invalid()),
    }
}

fn prefix_eq_ignore_case(a: &str, b: &str, n: usize) -> bool {
    a.chars()
        .take(n)
        .map(|ch| ch.to_lowercase().collect::<String>())
        .eq(b
            .chars()
            .take(n)
            .map(|ch| ch.to_lowercase().collect::<String>()))
        && a.chars().count() >= n
        && b.chars().count() >= n
}
