use super::support::periodic_transform_from_value;
use super::*;
use crate::builtins::fea::values::{
    quantity_scalar, COMPLEX_PHASE_CLASS, TEMPERATURE_DIFFERENCE_CLASS,
};

fn periodic_from_values(
    builtin: &'static str,
    pair: PairedConstraint,
    transform: &Value,
) -> BuiltinResult<PeriodicConstraint> {
    Ok(PeriodicConstraint {
        pair,
        transform: periodic_transform_from_value(builtin, transform)?,
    })
}

macro_rules! tie_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $constraint:ident, $kind:ident $(, $field:ident : $field_value:expr)?) => {
        #[runtime_builtin(
            name = $name,
            binding_variant = "default",
            builtin_path = "crate::builtins::fea::constraints::paired"
        )]
        pub async fn $function(
            id: String,
            primary: Value,
            secondary: Value,
            pairing: Value,
        ) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            let (id, geometry_id, geometry_revision, pair) =
                bound_region_pair($name, id, &primary, &secondary, &pairing)?;
            bound_constraint_to_object(
                $name,
                $class,
                geometry_id,
                geometry_revision,
                $constraint {
                    constraint_id: id,
                    $($field: $field_value,)?
                    kind: $kind::Tie(pair),
                },
            )
        }
    };
}

macro_rules! periodic_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $constraint:ident, $kind:ident $(, $field:ident : $field_value:expr)?) => {
        #[runtime_builtin(
            name = $name,
            binding_variant = "default",
            builtin_path = "crate::builtins::fea::constraints::paired"
        )]
        pub async fn $function(
            id: String,
            primary: Value,
            secondary: Value,
            transform: Value,
            pairing: Value,
        ) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            let (id, geometry_id, geometry_revision, pair) =
                bound_region_pair($name, id, &primary, &secondary, &pairing)?;
            let periodic = periodic_from_values($name, pair, &transform)?;
            bound_constraint_to_object(
                $name,
                $class,
                geometry_id,
                geometry_revision,
                $constraint {
                    constraint_id: id,
                    $($field: $field_value,)?
                    kind: $kind::Periodic(periodic),
                },
            )
        }
    };
}

tie_builtin!(
    structural_tie_builtin,
    FEA_STRUCTURAL_CONSTRAINT_TIE_DESCRIPTOR,
    "fea.StructuralConstraint.tie",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    StructuralConstraint,
    StructuralConstraintKind,
    frame: CoordinateFrame::Global
);
tie_builtin!(
    thermal_tie_builtin,
    FEA_THERMAL_CONSTRAINT_TIE_DESCRIPTOR,
    "fea.ThermalConstraint.tie",
    FEA_THERMAL_CONSTRAINT_CLASS,
    ThermalConstraint,
    ThermalConstraintKind
);

periodic_builtin!(
    structural_periodic_builtin,
    FEA_STRUCTURAL_CONSTRAINT_PERIODIC_DESCRIPTOR,
    "fea.StructuralConstraint.periodic",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    StructuralConstraint,
    StructuralConstraintKind,
    frame: CoordinateFrame::Global
);
periodic_builtin!(
    thermal_periodic_builtin,
    FEA_THERMAL_CONSTRAINT_PERIODIC_DESCRIPTOR,
    "fea.ThermalConstraint.periodic",
    FEA_THERMAL_CONSTRAINT_CLASS,
    ThermalConstraint,
    ThermalConstraintKind
);
periodic_builtin!(
    electrical_periodic_builtin,
    FEA_ELECTRICAL_CONSTRAINT_PERIODIC_DESCRIPTOR,
    "fea.ElectricalConstraint.periodic",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    ElectricalConstraint,
    ElectricalConstraintKind
);
periodic_builtin!(
    magnetic_periodic_builtin,
    FEA_MAGNETIC_CONSTRAINT_PERIODIC_DESCRIPTOR,
    "fea.MagneticConstraint.periodic",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    MagneticConstraint,
    MagneticConstraintKind
);
periodic_builtin!(
    acoustic_periodic_builtin,
    FEA_ACOUSTIC_CONSTRAINT_PERIODIC_DESCRIPTOR,
    "fea.AcousticConstraint.periodic",
    FEA_ACOUSTIC_CONSTRAINT_CLASS,
    AcousticConstraint,
    AcousticConstraintKind
);
periodic_builtin!(
    fluid_periodic_builtin,
    FEA_FLUID_CONSTRAINT_PERIODIC_DESCRIPTOR,
    "fea.FluidConstraint.periodic",
    FEA_FLUID_CONSTRAINT_CLASS,
    FluidConstraint,
    FluidConstraintKind
);

macro_rules! phase_periodic_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $constraint:ident, $kind:ident) => {
        #[runtime_builtin(
                                    name = $name,
                                    binding_variant = "default",
                                    builtin_path = "crate::builtins::fea::constraints::paired"
                                )]
        pub async fn $function(
            id: String,
            primary: Value,
            secondary: Value,
            transform: Value,
            pairing: Value,
            phase: Value,
        ) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            let (id, geometry_id, geometry_revision, pair) =
                bound_region_pair($name, id, &primary, &secondary, &pairing)?;
            let periodic = periodic_from_values($name, pair, &transform)?;
            let phase = ComplexPhase::new(quantity_scalar(
                $name,
                &phase,
                COMPLEX_PHASE_CLASS,
                "radians",
            )?);
            bound_constraint_to_object(
                $name,
                $class,
                geometry_id,
                geometry_revision,
                $constraint {
                    constraint_id: id,
                    kind: $kind::PhaseShiftedPeriodic(PhaseShiftedPeriodicConstraint {
                        periodic,
                        phase,
                    }),
                },
            )
        }
    };
}

phase_periodic_builtin!(
    magnetic_phase_periodic_builtin,
    FEA_MAGNETIC_CONSTRAINT_PHASE_SHIFTED_PERIODIC_DESCRIPTOR,
    "fea.MagneticConstraint.phaseShiftedPeriodic",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    MagneticConstraint,
    MagneticConstraintKind
);
phase_periodic_builtin!(
    acoustic_phase_periodic_builtin,
    FEA_ACOUSTIC_CONSTRAINT_PHASE_SHIFTED_PERIODIC_DESCRIPTOR,
    "fea.AcousticConstraint.phaseShiftedPeriodic",
    FEA_ACOUSTIC_CONSTRAINT_CLASS,
    AcousticConstraint,
    AcousticConstraintKind
);

#[runtime_builtin(
    name = "fea.ThermalConstraint.affineJump",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::constraints::paired"
)]
pub async fn thermal_affine_jump_builtin(
    id: String,
    primary: Value,
    secondary: Value,
    transform: Value,
    pairing: Value,
    jump: Value,
) -> BuiltinResult<Value> {
    fixed_outputs(
        &runmat_builtins::FEA_THERMAL_CONSTRAINT_AFFINE_JUMP_DESCRIPTOR,
        "fea.ThermalConstraint.affineJump",
    )?;
    let (id, geometry_id, geometry_revision, pair) = bound_region_pair(
        "fea.ThermalConstraint.affineJump",
        id,
        &primary,
        &secondary,
        &pairing,
    )?;
    let periodic = periodic_from_values("fea.ThermalConstraint.affineJump", pair, &transform)?;
    let jump = TemperatureDifference::new(quantity_scalar(
        "fea.ThermalConstraint.affineJump",
        &jump,
        TEMPERATURE_DIFFERENCE_CLASS,
        "kelvin",
    )?);
    bound_constraint_to_object(
        "fea.ThermalConstraint.affineJump",
        FEA_THERMAL_CONSTRAINT_CLASS,
        geometry_id,
        geometry_revision,
        ThermalConstraint {
            constraint_id: id,
            kind: ThermalConstraintKind::AffineJump {
                periodic,
                temperature_jump: PrescribedValue::Constant(jump),
            },
        },
    )
}
