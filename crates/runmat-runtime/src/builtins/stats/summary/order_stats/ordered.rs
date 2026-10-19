use super::*;

/// A value used only while ordering observations.  Integer order statistics
/// return doubles, but their input ordering and tie detection must not pass
/// through the lossy f64 mirror first (notably for uint64 values above 2^53).
#[derive(Clone)]
pub(super) enum OrderedValue {
    Float(f64),
    Integer(IntValue),
}

impl OrderedValue {
    pub(super) fn is_nan(&self) -> bool {
        matches!(self, Self::Float(value) if value.is_nan())
    }

    pub(super) fn as_f64(&self) -> f64 {
        match self {
            Self::Float(value) => *value,
            Self::Integer(value) => value.to_f64(),
        }
    }
}

pub(super) fn ordered_input_values(input: &Tensor) -> Vec<OrderedValue> {
    if let Some(storage) = input.integer_storage() {
        return storage
            .exact_values()
            .iter()
            .cloned()
            .map(OrderedValue::Integer)
            .collect();
    }
    tensor::tensor_values_f64_cow(input)
        .iter()
        .copied()
        .map(OrderedValue::Float)
        .collect()
}

pub(super) fn compare_ordered_values(left: &OrderedValue, right: &OrderedValue) -> Ordering {
    match (left, right) {
        (OrderedValue::Float(left), OrderedValue::Float(right)) => {
            left.partial_cmp(right).unwrap_or(Ordering::Greater)
        }
        (OrderedValue::Integer(left), OrderedValue::Integer(right)) => match (left, right) {
            (IntValue::I8(left), IntValue::I8(right)) => left.cmp(right),
            (IntValue::I16(left), IntValue::I16(right)) => left.cmp(right),
            (IntValue::I32(left), IntValue::I32(right)) => left.cmp(right),
            (IntValue::I64(left), IntValue::I64(right)) => left.cmp(right),
            (IntValue::U8(left), IntValue::U8(right)) => left.cmp(right),
            (IntValue::U16(left), IntValue::U16(right)) => left.cmp(right),
            (IntValue::U32(left), IntValue::U32(right)) => left.cmp(right),
            (IntValue::U64(left), IntValue::U64(right)) => left.cmp(right),
            _ => unreachable!("integer tensor storage is homogeneous"),
        },
        _ => unreachable!("an order-statistics input has one numeric representation"),
    }
}

pub(super) fn ordered_values_equal(left: &OrderedValue, right: &OrderedValue) -> bool {
    match (left, right) {
        (OrderedValue::Float(left), OrderedValue::Float(right)) => left == right,
        (OrderedValue::Integer(left), OrderedValue::Integer(right)) => left == right,
        _ => false,
    }
}
