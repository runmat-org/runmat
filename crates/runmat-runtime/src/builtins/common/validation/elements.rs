//! Borrowed numeric element traversal, including implicit sparse zeros.

use super::{integer_f64_order, tensor, IntValue, IntegerStorage, Ordering, Value};

pub(super) fn complex_real_values_all(
    value: &Value,
    float_pred: impl Fn(f64) -> bool + Copy,
    integer_pred: impl Fn(&IntValue) -> bool + Copy,
) -> Option<bool> {
    match value {
        Value::Complex(real, _) => Some(float_pred(*real)),
        Value::ComplexTensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                Some(integer_storage_all(storage.real(), integer_pred))
            } else {
                Some(
                    tensor
                        .materialize_f64()
                        .iter()
                        .all(|(real, _)| float_pred(*real)),
                )
            }
        }
        _ => None,
    }
}

pub(super) fn exact_integer_values_all(
    value: &Value,
    pred: impl Fn(&IntValue) -> bool + Copy,
) -> Option<bool> {
    match value {
        Value::Int(value) => Some(pred(value)),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| integer_storage_all(storage, pred)),
        Value::SparseTensor(tensor) => tensor.integer_storage().map(|storage| {
            let numel = tensor.rows.saturating_mul(tensor.cols);
            integer_storage_all(storage, pred)
                && (storage.len() >= numel || integer_storage_zero_satisfies(storage, pred))
        }),
        Value::ComplexTensor(tensor) => tensor.integer_storage().map(|storage| {
            (0..storage.len()).all(|index| {
                let Some(real) = storage.real().value_at(index) else {
                    return false;
                };
                let Some(imag) = storage.imag().value_at(index) else {
                    return false;
                };
                imag.is_zero() && pred(&real)
            })
        }),
        _ => None,
    }
}

fn integer_storage_all(storage: &IntegerStorage, pred: impl Fn(&IntValue) -> bool + Copy) -> bool {
    (0..storage.len()).all(|index| {
        storage
            .value_at(index)
            .as_ref()
            .is_some_and(|value| pred(value))
    })
}

fn integer_storage_zero_satisfies(
    storage: &IntegerStorage,
    pred: impl Fn(&IntValue) -> bool,
) -> bool {
    storage.zeros_like(1).value_at(0).as_ref().is_some_and(pred)
}

pub(super) fn int_f64_matches(
    integer: &IntValue,
    threshold: f64,
    pred: impl Fn(Ordering) -> bool,
) -> bool {
    integer_f64_order(integer.clone(), threshold).is_some_and(pred)
}

pub(super) fn int_is_positive(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value > 0,
        IntValue::I16(value) => *value > 0,
        IntValue::I32(value) => *value > 0,
        IntValue::I64(value) => *value > 0,
        IntValue::U8(value) => *value > 0,
        IntValue::U16(value) => *value > 0,
        IntValue::U32(value) => *value > 0,
        IntValue::U64(value) => *value > 0,
    }
}

pub(super) fn int_is_negative(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value < 0,
        IntValue::I16(value) => *value < 0,
        IntValue::I32(value) => *value < 0,
        IntValue::I64(value) => *value < 0,
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => false,
    }
}

pub(super) fn int_is_nonnegative(value: &IntValue) -> bool {
    !int_is_negative(value)
}

pub(super) fn int_is_nonpositive(value: &IntValue) -> bool {
    !int_is_positive(value)
}

pub(super) fn numeric_values_all(value: &Value, pred: impl Fn(f64) -> bool) -> bool {
    match value {
        Value::Num(v) => pred(*v),
        Value::Int(v) => pred(v.to_f64()),
        Value::Bool(v) => pred(if *v { 1.0 } else { 0.0 }),
        Value::LogicalArray(a) => a.data.iter().map(|v| f64::from(*v != 0)).all(pred),
        Value::Tensor(t) => tensor::tensor_values_f64(t).into_iter().all(pred),
        Value::SparseTensor(t) if t.integer_storage().is_some() => {
            let storage = t
                .integer_storage()
                .expect("integer storage was checked above");
            let numel = t.rows.saturating_mul(t.cols);
            (0..storage.len()).all(|index| {
                pred(
                    storage
                        .value_at(index)
                        .expect("sparse integer storage length is consistent")
                        .to_f64(),
                )
            }) && (storage.len() >= numel || pred(0.0))
        }
        Value::SparseTensor(t) if t.is_complex() => {
            let values = t.materialize_complex_f64().expect("complex sparse storage");
            let numel = t.rows.saturating_mul(t.cols);
            values.iter().all(|value| value.1 == 0.0 && pred(value.0))
                && (values.len() >= numel || pred(0.0))
        }
        Value::SparseTensor(t) => {
            let numel = t.rows.saturating_mul(t.cols);
            let values = t.materialize_f64();
            values.iter().copied().all(&pred) && (values.len() >= numel || pred(0.0))
        }
        Value::Complex(re, im) => *im == 0.0 && pred(*re),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => {
            let storage = t
                .integer_storage()
                .expect("integer storage was checked above");
            (0..storage.len()).all(|index| {
                let real = storage
                    .real()
                    .value_at(index)
                    .expect("complex integer real storage length is consistent");
                let imag = storage
                    .imag()
                    .value_at(index)
                    .expect("complex integer imaginary storage length is consistent");
                imag.is_zero() && pred(real.to_f64())
            })
        }
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| *im == 0.0 && pred(*re)),
        _ => false,
    }
}
