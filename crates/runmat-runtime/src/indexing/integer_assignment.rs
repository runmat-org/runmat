use crate::indexing::plan::IndexPlan;
use crate::runtime_error::semantic_error as mex;
use crate::RuntimeError;
use runmat_value::{IntValue, IntegerStorage};

#[derive(Clone)]
pub(crate) enum IntegerAssignmentValue {
    Exact(IntValue),
    Float(f64),
}

#[derive(Clone)]
pub(crate) struct ComplexIntegerAssignmentValue {
    pub real: IntegerAssignmentValue,
    pub imag: IntegerAssignmentValue,
}

pub(crate) fn values(storage: &IntegerStorage) -> Vec<IntValue> {
    match storage {
        IntegerStorage::I8(values) => values.iter().copied().map(IntValue::I8).collect(),
        IntegerStorage::I16(values) => values.iter().copied().map(IntValue::I16).collect(),
        IntegerStorage::I32(values) => values.iter().copied().map(IntValue::I32).collect(),
        IntegerStorage::I64(values) => values.iter().copied().map(IntValue::I64).collect(),
        IntegerStorage::U8(values) => values.iter().copied().map(IntValue::U8).collect(),
        IntegerStorage::U16(values) => values.iter().copied().map(IntValue::U16).collect(),
        IntegerStorage::U32(values) => values.iter().copied().map(IntValue::U32).collect(),
        IntegerStorage::U64(values) => values.iter().copied().map(IntValue::U64).collect(),
    }
}

/// Converts a scalar assignment value to the exact class of `storage`.
///
/// This is shared by dense and sparse integer assignment so they retain the
/// same round-and-saturate behavior without converting wide integers through
/// the floating compatibility view.
pub(crate) fn scalar_value(storage: &IntegerStorage, value: &IntegerAssignmentValue) -> IntValue {
    match value {
        IntegerAssignmentValue::Exact(value) => storage.cast_exact_assignment(value),
        IntegerAssignmentValue::Float(value) => storage.cast_f64_assignment(*value),
    }
}

pub(crate) fn scatter(
    storage: &mut IntegerStorage,
    plan: &IndexPlan,
    rhs_values: &[IntegerAssignmentValue],
) -> Result<(), RuntimeError> {
    if rhs_values.len() != plan.indices.len() {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    for (&dst, value) in plan.indices.iter().zip(rhs_values.iter()) {
        match value {
            IntegerAssignmentValue::Exact(value) => storage
                .set_exact_assignment(dst as usize, value)
                .map_err(|err| mex("Assignment", &err))?,
            IntegerAssignmentValue::Float(value) => storage
                .set_f64_assignment(dst as usize, *value)
                .map_err(|err| mex("Assignment", &err))?,
        }
    }
    Ok(())
}
