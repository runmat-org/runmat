//! Request-local literal facts for exact external operands only.
//!
//! A produced ValueId is absent from `FusionInvocationLeaves`, so this cannot
//! turn an unevaluated intermediate into a source literal. Input IDs override
//! captured plan constants in that shared owner before projection.

use runmat_types::{LiteralContext, LiteralValue, NumericClass};
use runmat_value::{IntValue, NumericScalar, Value};

use crate::graph::ValueId;

// Catalog control vectors are small; bounding request-local projection keeps
// this proof preparation independent of a data tensor's element count.
const MAX_EXACT_CONTROL_ELEMENTS: usize = 16;

fn integer(value: IntValue) -> LiteralValue {
    let class = match value {
        IntValue::I8(_) => NumericClass::Int8,
        IntValue::I16(_) => NumericClass::Int16,
        IntValue::I32(_) => NumericClass::Int32,
        IntValue::I64(_) => NumericClass::Int64,
        IntValue::U8(_) => NumericClass::UInt8,
        IntValue::U16(_) => NumericClass::UInt16,
        IntValue::U32(_) => NumericClass::UInt32,
        IntValue::U64(_) => NumericClass::UInt64,
    };
    LiteralValue::Integer {
        text: value.to_i128().to_string(),
        class,
    }
}

fn scalar(value: NumericScalar) -> LiteralValue {
    match value {
        NumericScalar::F64(value) => LiteralValue::Number(value),
        NumericScalar::F32(value) => LiteralValue::Real {
            text: value.to_string(),
            class: NumericClass::Single,
        },
        other => other
            .into_int_value()
            .map_or(LiteralValue::Unknown, integer),
    }
}

fn exact(value: &Value) -> LiteralValue {
    match value {
        Value::Num(value) => LiteralValue::Number(*value),
        Value::Int(value) => integer(value.clone()),
        Value::Bool(value) => LiteralValue::Bool(*value),
        Value::String(value) => value
            .as_str()
            .map(|text| LiteralValue::String(text.to_owned()))
            .unwrap_or(LiteralValue::Unknown),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .numeric_value_at(0)
            .map(scalar)
            .unwrap_or(LiteralValue::Unknown),
        Value::Tensor(tensor)
            if tensor.len() <= MAX_EXACT_CONTROL_ELEMENTS
                && tensor.shape().len() == 2
                && (tensor.rows() == 1 || tensor.cols() == 1) =>
        {
            let values = (0..tensor.len())
                .map(|index| tensor.numeric_value_at(index).map(scalar))
                .collect::<Option<Vec<_>>>();
            values.map_or(LiteralValue::Unknown, LiteralValue::Vector)
        }
        _ => LiteralValue::Unknown,
    }
}

pub(super) fn for_operands<'a>(
    ids: &[ValueId],
    source_value: impl Fn(ValueId) -> Option<&'a Value>,
) -> LiteralContext {
    LiteralContext::new(
        ids.iter()
            .map(|id| source_value(*id).map_or(LiteralValue::Unknown, exact))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::Tensor;
    use std::collections::HashMap;

    #[test]
    fn exact_controls_project_without_inventing_computed_literals() {
        let axes = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap());
        let output = Value::String("native".into());
        let leaves = HashMap::from([(7, &axes), (8, &output)]);
        let projected = for_operands(&[6, 7, 8], |id| leaves.get(&id).copied());
        assert_eq!(projected.literal_args[0], LiteralValue::Unknown);
        assert_eq!(
            projected.literal_args[1],
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)])
        );
        assert_eq!(
            projected.literal_args[2],
            LiteralValue::String("native".into())
        );
    }

    #[test]
    fn typed_integer_controls_remain_exact_and_do_not_become_binary64() {
        let integer = Value::Tensor(
            Tensor::from_numeric_storage(
                runmat_value::NumericStorage::U64(vec![u64::MAX]),
                vec![1, 1],
            )
            .unwrap(),
        );
        let leaves = HashMap::from([(3, &integer)]);
        assert_eq!(
            for_operands(&[3], |id| leaves.get(&id).copied()).literal_args,
            [LiteralValue::Integer {
                text: u64::MAX.to_string(),
                class: NumericClass::UInt64,
            }]
        );
    }
}
