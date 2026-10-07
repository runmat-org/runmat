use super::*;
use runmat_value::{IntValue, NumericScalar, Tensor};

pub(super) fn compatible_integer_column(
    values: &[Value],
    rows: usize,
) -> BuiltinResult<Option<Value>> {
    let Some(prototype) = values
        .iter()
        .find_map(|value| integer_row(value).map(|row| row.0))
    else {
        return Ok(None);
    };
    let Some(width) = values.first().and_then(integer_compatible_row_width) else {
        return Ok(None);
    };
    if values.iter().skip(1).any(|value| {
        integer_compatible_row_width(value).is_none_or(|next_width| next_width != width)
    }) {
        return Ok(None);
    }
    let has_scalar_double = values.iter().any(|value| matches!(value, Value::Num(_)));
    if has_scalar_double
        && matches!(
            prototype,
            runmat_value::IntegerStorage::I64(_) | runmat_value::IntegerStorage::U64(_)
        )
    {
        return Ok(None);
    }
    let total = rows
        .checked_mul(width)
        .ok_or_else(|| invalid_variable("cell2table: integer variable shape exceeds limits"))?;
    let mut output = Tensor::new_integer(prototype.zeros_like(total), vec![rows, width])
        .map_err(|err| invalid_variable(format!("cell2table: {err}")))?;
    for (row, value) in values.iter().enumerate() {
        let actual_width = integer_compatible_row_width(value).expect("validated integer row");
        for col in 0..actual_width {
            let scalar = integer_row_value(value, col).expect("validated integer row value");
            output
                .set_numeric_assignment_at(row + col * rows, scalar)
                .map_err(|err| invalid_variable(format!("cell2table: {err}")))?;
        }
    }
    Ok(Some(Value::Tensor(output)))
}

pub(super) fn integer_compatible_row_width(value: &Value) -> Option<usize> {
    match value {
        Value::Num(_) | Value::Int(_) => Some(1),
        _ => integer_row(value).map(|(_, width)| width),
    }
}

pub(super) fn integer_row(value: &Value) -> Option<(&runmat_value::IntegerStorage, usize)> {
    match value {
        Value::Int(value) => Some((integer_scalar_prototype(value), 1)),
        Value::Tensor(tensor)
            if tensor.shape().len() <= 2
                && tensor.shape().first().copied().unwrap_or(1) == 1
                && tensor.integer_storage().is_some() =>
        {
            Some((tensor.integer_storage().expect("checked"), tensor.len()))
        }
        _ => None,
    }
}

pub(super) fn integer_scalar_prototype(value: &IntValue) -> &'static runmat_value::IntegerStorage {
    use runmat_value::IntegerStorage;
    static I8: IntegerStorage = IntegerStorage::I8(Vec::new());
    static I16: IntegerStorage = IntegerStorage::I16(Vec::new());
    static I32: IntegerStorage = IntegerStorage::I32(Vec::new());
    static I64: IntegerStorage = IntegerStorage::I64(Vec::new());
    static U8: IntegerStorage = IntegerStorage::U8(Vec::new());
    static U16: IntegerStorage = IntegerStorage::U16(Vec::new());
    static U32: IntegerStorage = IntegerStorage::U32(Vec::new());
    static U64: IntegerStorage = IntegerStorage::U64(Vec::new());
    match value {
        IntValue::I8(_) => &I8,
        IntValue::I16(_) => &I16,
        IntValue::I32(_) => &I32,
        IntValue::I64(_) => &I64,
        IntValue::U8(_) => &U8,
        IntValue::U16(_) => &U16,
        IntValue::U32(_) => &U32,
        IntValue::U64(_) => &U64,
    }
}

pub(super) fn integer_row_value(value: &Value, index: usize) -> Option<NumericScalar> {
    match value {
        Value::Num(value) if index == 0 => Some(NumericScalar::F64(*value)),
        Value::Int(value) if index == 0 => Some(NumericScalar::from(value.clone())),
        Value::Tensor(tensor) => tensor
            .integer_storage()?
            .value_at(index)
            .map(NumericScalar::from),
        _ => None,
    }
}
