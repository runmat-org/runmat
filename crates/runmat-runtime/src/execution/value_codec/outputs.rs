//! Ordinal outputs are protocol structure; a MATLAB cell remains a single value.
use super::{decode_inline_value, encode_inline_value, ValueCodecError};
use runmat_execution::value::ValuePayload;
use runmat_value::Value;

/// Encode one ordinal output per requested output from a typed sequence.
pub fn encode_inline_sequence(
    outputs: &runmat_value::ValueSequence,
    requested: usize,
) -> Result<Vec<ValuePayload>, ValueCodecError> {
    if outputs.len() != requested {
        return Err(ValueCodecError::invalid(
            "$",
            "output count differs from the original contract",
        ));
    }
    outputs.iter().map(encode_inline_value).collect()
}

pub fn encode_inline_outputs(
    value: &Value,
    requested: usize,
) -> Result<Vec<ValuePayload>, ValueCodecError> {
    validate_output_arity(value, requested)?;
    if requested == 1 {
        return encode_inline_value(value).map(|value| vec![value]);
    }
    let Value::OutputList(values) = value else {
        unreachable!("ordinal shape was validated");
    };
    values.iter().map(encode_inline_value).collect()
}

/// Validate ordinal shape without requiring portable storage (local values may be opaque).
pub fn validate_output_arity(value: &Value, requested: usize) -> Result<(), ValueCodecError> {
    if requested == 1 {
        if matches!(value, Value::OutputList(_)) {
            return Err(ValueCodecError::invalid(
                "$",
                "single output cannot be an ordinal list",
            ));
        }
        return Ok(());
    }
    let Value::OutputList(values) = value else {
        return Err(ValueCodecError::invalid(
            "$",
            "expected an ordinal output list",
        ));
    };
    if values.len() != requested {
        return Err(ValueCodecError::invalid(
            "$",
            "output count differs from the original contract",
        ));
    }
    Ok(())
}

pub fn decode_inline_outputs(
    payloads: &[ValuePayload],
    requested: usize,
) -> Result<Value, ValueCodecError> {
    if payloads.len() != requested {
        return Err(ValueCodecError::invalid(
            "$",
            "output count differs from the original contract",
        ));
    }
    if requested == 1 {
        return decode_inline_value(&payloads[0]);
    }
    payloads
        .iter()
        .map(decode_inline_value)
        .collect::<Result<Vec<_>, _>>()
        .map(Value::output_list)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::CellArray;

    #[test]
    fn original_output_ordinals_and_cell_shape_survive_transport() {
        let cell =
            Value::Cell(CellArray::new(vec![Value::Num(7.0), Value::Num(9.0)], 1, 2).unwrap());
        for (value, count) in [
            (Value::output_list(vec![]), 0),
            (cell.clone(), 1),
            (Value::output_list(vec![cell.clone(), Value::Num(11.0)]), 2),
        ] {
            let payloads = encode_inline_outputs(&value, count).unwrap();
            assert_eq!(decode_inline_outputs(&payloads, count).unwrap(), value);
            assert!(decode_inline_outputs(&payloads, count + 1).is_err());
        }
        assert!(encode_inline_outputs(&cell, 2).is_err());
        assert!(encode_inline_outputs(&Value::output_list(vec![cell]), 2).is_err());
    }
}
