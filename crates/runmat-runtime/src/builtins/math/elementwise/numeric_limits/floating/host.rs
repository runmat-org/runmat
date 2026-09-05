use runmat_builtins::FloatingLimitKind;
use runmat_value::{ComplexTensor, SparseTensor, Tensor, Value};

use crate::BuiltinResult;

pub(super) fn value(
    class: runmat_types::NumericClass,
    complex: bool,
    sparse: bool,
    kind: FloatingLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let shape = vec![1, 1];
    match (class, complex, sparse) {
        (runmat_types::NumericClass::Double, false, false) => Ok(Value::Num(f64_value(kind))),
        (runmat_types::NumericClass::Single, false, false) => {
            Tensor::from_f32(vec![f32_value(kind)], shape)
                .map(Value::Tensor)
                .map_err(|error| super::super::errors::internal(builtin, error))
        }
        (runmat_types::NumericClass::Double, true, false) => {
            Ok(Value::Complex(f64_value(kind), 0.0))
        }
        (runmat_types::NumericClass::Single, true, false) => {
            ComplexTensor::from_f32(vec![(f32_value(kind), 0.0)], shape)
                .map(Value::ComplexTensor)
                .map_err(|error| super::super::errors::internal(builtin, error))
        }
        (runmat_types::NumericClass::Double, false, true) => {
            SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![f64_value(kind)])
                .map(Value::SparseTensor)
                .map_err(|error| super::super::errors::internal(builtin, error))
        }
        (runmat_types::NumericClass::Single, false, true) => {
            SparseTensor::new_f32(1, 1, vec![0, 1], vec![0], vec![f32_value(kind)])
                .map(Value::SparseTensor)
                .map_err(|error| super::super::errors::internal(builtin, error))
        }
        (runmat_types::NumericClass::Double, true, true) => {
            SparseTensor::new_complex(1, 1, vec![0, 1], vec![0], vec![(f64_value(kind), 0.0)])
                .map(Value::SparseTensor)
                .map_err(|error| super::super::errors::internal(builtin, error))
        }
        (runmat_types::NumericClass::Single, true, true) => {
            SparseTensor::new_complex_f32(1, 1, vec![0, 1], vec![0], vec![(f32_value(kind), 0.0)])
                .map(Value::SparseTensor)
                .map_err(|error| super::super::errors::internal(builtin, error))
        }
        _ => Err(super::super::errors::invalid_floating_prototype(builtin)),
    }
}

pub(super) fn f64_value(kind: FloatingLimitKind) -> f64 {
    match kind {
        FloatingLimitKind::SmallestNormal => f64::MIN_POSITIVE,
        FloatingLimitKind::LargestFinite => f64::MAX,
        FloatingLimitKind::LargestConsecutiveInteger => 2f64.powi(53),
    }
}

pub(super) fn f32_value(kind: FloatingLimitKind) -> f32 {
    match kind {
        FloatingLimitKind::SmallestNormal => f32::MIN_POSITIVE,
        FloatingLimitKind::LargestFinite => f32::MAX,
        FloatingLimitKind::LargestConsecutiveInteger => 2f32.powi(24),
    }
}
