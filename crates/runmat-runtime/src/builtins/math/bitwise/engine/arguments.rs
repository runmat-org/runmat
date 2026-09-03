use super::*;

pub(super) async fn binary_bitwise_from_args(
    name: &'static str,
    args: Vec<Value>,
    operator: BinaryBitwiseOperator,
) -> BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "expected A, B, and an optional assumedtype",
        ));
    }
    let mut args = args.into_iter();
    let lhs = args.next().expect("A");
    let rhs = args.next().expect("B");
    let assumed = args
        .next()
        .map(|value| parse_assumed_type(name, value))
        .transpose()?;
    binary_bitwise(name, lhs, rhs, assumed, operator).await
}

pub(super) fn apply_binary_operator(operator: BinaryBitwiseOperator, left: u64, right: u64) -> u64 {
    match operator {
        BinaryBitwiseOperator::And => left & right,
        BinaryBitwiseOperator::Or => left | right,
        BinaryBitwiseOperator::Xor => left ^ right,
    }
}

pub(super) fn unary_args(
    name: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<(Value, Option<IntegerClass>)> {
    if !(1..=2).contains(&args.len()) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "expected A and an optional assumedtype",
        ));
    }
    let mut args = args.into_iter();
    let value = args.next().expect("A");
    let assumed = args
        .next()
        .map(|value| parse_assumed_type(name, value))
        .transpose()?;
    Ok((value, assumed))
}

pub(super) fn value_bit_args(
    name: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<(Value, Value, Option<IntegerClass>)> {
    if !(2..=3).contains(&args.len()) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "expected A, bit, and an optional assumedtype",
        ));
    }
    let mut args = args.into_iter();
    let value = args.next().expect("A");
    let bit = args.next().expect("bit");
    let assumed = args
        .next()
        .map(|value| parse_assumed_type(name, value))
        .transpose()?;
    Ok((value, bit, assumed))
}

pub(super) fn bitset_args(
    args: Vec<Value>,
) -> BuiltinResult<(Value, Value, Option<Value>, Option<IntegerClass>)> {
    if !(2..=4).contains(&args.len()) {
        return Err(error_with_detail(
            BITSET_NAME,
            &ERROR_INVALID_INPUT,
            "expected A, bit, optional V, and optional assumedtype",
        ));
    }
    let mut args = args.into_iter();
    let value = args.next().expect("A");
    let bit = args.next().expect("bit");
    let third = args.next();
    let fourth = args.next();
    match (third, fourth) {
        (None, None) => Ok((value, bit, None, None)),
        (Some(third), None) if is_text_value(&third) => Ok((
            value,
            bit,
            None,
            Some(parse_assumed_type(BITSET_NAME, third)?),
        )),
        (Some(third), None) => Ok((value, bit, Some(third), None)),
        (Some(third), Some(fourth)) => Ok((
            value,
            bit,
            Some(third),
            Some(parse_assumed_type(BITSET_NAME, fourth)?),
        )),
        (None, Some(_)) => unreachable!("fourth argument requires third argument"),
    }
}

pub(super) fn is_text_value(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

pub(super) fn parse_assumed_type(name: &'static str, value: Value) -> BuiltinResult<IntegerClass> {
    let Some(keyword) = keyword_of(&value) else {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "assumedtype must be an integer class name",
        ));
    };
    IntegerClass::from_class_name(&keyword).ok_or_else(|| {
        error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("unsupported assumedtype {keyword:?}"),
        )
    })
}
