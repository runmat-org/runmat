use super::*;
use crate::builtins::fea::values::PRESSURE_CLASS;

define_quantity_constraint!(
    acoustic_prescribed_pressure_builtin,
    FEA_ACOUSTIC_CONSTRAINT_PRESCRIBED_PRESSURE_DESCRIPTOR,
    "fea.AcousticConstraint.prescribedPressure",
    FEA_ACOUSTIC_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::acoustic",
    PRESSURE_CLASS,
    "pascals",
    |constraint_id, target, pascals| AcousticConstraint {
        constraint_id,
        kind: AcousticConstraintKind::PrescribedPressure {
            target,
            value: PrescribedValue::Constant(Pressure::new(pascals)),
        },
    }
);
