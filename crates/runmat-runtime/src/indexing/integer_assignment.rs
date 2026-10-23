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
