use super::*;
use crate::builtins::fea::values::PRESSURE_CLASS;

define_quantity_constraint!(
    acoustic_prescribed_pressure_builtin,
    ACOUSTIC_PRESCRIBED_PRESSURE_DESCRIPTOR,
    ACOUSTIC_PRESCRIBED_PRESSURE_SIGNATURES,
    ACOUSTIC_PRESCRIBED_PRESSURE_INPUTS,
    "fea.AcousticConstraint.prescribedPressure",
    "constraint = fea.AcousticConstraint.prescribedPressure(id, region, pressure)",
    "Create a prescribed acoustic-pressure constraint.",
    FEA_ACOUSTIC_CONSTRAINT_CLASS,
    "pressure",
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
