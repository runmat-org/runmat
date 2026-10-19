use super::support::periodic_transform_from_value;
use super::*;
use crate::builtins::fea::values::{
    quantity_scalar, COMPLEX_PHASE_CLASS, TEMPERATURE_DIFFERENCE_CLASS,
};

const TIE_INPUTS: [BuiltinParamDescriptor; 4] = [
    parameter(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier.",
    ),
    parameter("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    parameter(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region.",
    ),
    parameter(
        "pairing",
        BuiltinParamType::Any,
        "Typed fea.PeriodicPairing policy.",
    ),
];
const PERIODIC_INPUTS: [BuiltinParamDescriptor; 5] = [
    parameter(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier.",
    ),
    parameter("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    parameter(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region.",
    ),
    parameter(
        "transform",
        BuiltinParamType::Any,
        "Typed fea.PeriodicTransform.",
    ),
    parameter(
        "pairing",
        BuiltinParamType::Any,
        "Typed fea.PeriodicPairing policy.",
    ),
];
const PHASE_PERIODIC_INPUTS: [BuiltinParamDescriptor; 6] = [
    parameter(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier.",
    ),
    parameter("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    parameter(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region.",
    ),
    parameter(
        "transform",
        BuiltinParamType::Any,
        "Typed fea.PeriodicTransform.",
    ),
    parameter(
        "pairing",
        BuiltinParamType::Any,
        "Typed fea.PeriodicPairing policy.",
    ),
    parameter(
        "phase",
        BuiltinParamType::Any,
        "Typed fea.ComplexPhase shift.",
    ),
];
const AFFINE_JUMP_INPUTS: [BuiltinParamDescriptor; 6] = [
    parameter(
        "id",
        BuiltinParamType::StringScalar,
        "Constraint identifier.",
    ),
    parameter("primary", BuiltinParamType::Any, "Primary geometry.Region."),
    parameter(
        "secondary",
        BuiltinParamType::Any,
        "Secondary geometry.Region.",
    ),
    parameter(
        "transform",
        BuiltinParamType::Any,
        "Typed fea.PeriodicTransform.",
    ),
    parameter(
        "pairing",
        BuiltinParamType::Any,
        "Typed fea.PeriodicPairing policy.",
    ),
    parameter(
        "jump",
        BuiltinParamType::Any,
        "Typed fea.TemperatureDifference affine jump.",
    ),
];

const fn parameter(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

macro_rules! descriptor {
    ($name:ident, $signatures:ident, $label:literal, $inputs:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &$inputs,
            outputs: &OUT_ANY,
        }];
        pub(super) const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(
    STRUCTURAL_TIE_DESCRIPTOR,
    STRUCTURAL_TIE_SIGNATURES,
    "constraint = fea.StructuralConstraint.tie(id, primary, secondary, pairing)",
    TIE_INPUTS
);
descriptor!(
    THERMAL_TIE_DESCRIPTOR,
    THERMAL_TIE_SIGNATURES,
    "constraint = fea.ThermalConstraint.tie(id, primary, secondary, pairing)",
    TIE_INPUTS
);
descriptor!(
    STRUCTURAL_PERIODIC_DESCRIPTOR,
    STRUCTURAL_PERIODIC_SIGNATURES,
    "constraint = fea.StructuralConstraint.periodic(id, primary, secondary, transform, pairing)",
    PERIODIC_INPUTS
);
descriptor!(
    THERMAL_PERIODIC_DESCRIPTOR,
    THERMAL_PERIODIC_SIGNATURES,
    "constraint = fea.ThermalConstraint.periodic(id, primary, secondary, transform, pairing)",
    PERIODIC_INPUTS
);
descriptor!(
    ELECTRICAL_PERIODIC_DESCRIPTOR,
    ELECTRICAL_PERIODIC_SIGNATURES,
    "constraint = fea.ElectricalConstraint.periodic(id, primary, secondary, transform, pairing)",
    PERIODIC_INPUTS
);
descriptor!(
    MAGNETIC_PERIODIC_DESCRIPTOR,
    MAGNETIC_PERIODIC_SIGNATURES,
    "constraint = fea.MagneticConstraint.periodic(id, primary, secondary, transform, pairing)",
    PERIODIC_INPUTS
);
descriptor!(
    ACOUSTIC_PERIODIC_DESCRIPTOR,
    ACOUSTIC_PERIODIC_SIGNATURES,
    "constraint = fea.AcousticConstraint.periodic(id, primary, secondary, transform, pairing)",
    PERIODIC_INPUTS
);
descriptor!(
    FLUID_PERIODIC_DESCRIPTOR,
    FLUID_PERIODIC_SIGNATURES,
    "constraint = fea.FluidConstraint.periodic(id, primary, secondary, transform, pairing)",
    PERIODIC_INPUTS
);
descriptor!(MAGNETIC_PHASE_PERIODIC_DESCRIPTOR, MAGNETIC_PHASE_PERIODIC_SIGNATURES, "constraint = fea.MagneticConstraint.phaseShiftedPeriodic(id, primary, secondary, transform, pairing, phase)", PHASE_PERIODIC_INPUTS);
descriptor!(ACOUSTIC_PHASE_PERIODIC_DESCRIPTOR, ACOUSTIC_PHASE_PERIODIC_SIGNATURES, "constraint = fea.AcousticConstraint.phaseShiftedPeriodic(id, primary, secondary, transform, pairing, phase)", PHASE_PERIODIC_INPUTS);
descriptor!(
    AFFINE_JUMP_DESCRIPTOR,
    AFFINE_JUMP_SIGNATURES,
    "constraint = fea.ThermalConstraint.affineJump(id, primary, secondary, transform, pairing, jump)",
    AFFINE_JUMP_INPUTS
);

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
                            category = "fea",
                            summary = "Create a typed tie constraint between two geometry regions.",
                            keywords = "fea,constraint,tie,pairing,region,typed",
                            descriptor(crate::builtins::fea::constraints::paired::$descriptor),
                            integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
                            builtin_path = "crate::builtins::fea::constraints::paired"
                        )]
        pub async fn $function(
            id: String,
            primary: Value,
            secondary: Value,
            pairing: Value,
        ) -> BuiltinResult<Value> {
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
            category = "fea",
            summary = "Create a typed transformed periodic constraint between two geometry regions.",
            keywords = "fea,constraint,periodic,transform,pairing,region,typed",
            descriptor(crate::builtins::fea::constraints::paired::$descriptor),
            integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
            builtin_path = "crate::builtins::fea::constraints::paired"
        )]
        pub async fn $function(
            id: String,
            primary: Value,
            secondary: Value,
            transform: Value,
            pairing: Value,
        ) -> BuiltinResult<Value> {
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
    STRUCTURAL_TIE_DESCRIPTOR,
    "fea.StructuralConstraint.tie",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    StructuralConstraint,
    StructuralConstraintKind,
    frame: CoordinateFrame::Global
);
tie_builtin!(
    thermal_tie_builtin,
    THERMAL_TIE_DESCRIPTOR,
    "fea.ThermalConstraint.tie",
    FEA_THERMAL_CONSTRAINT_CLASS,
    ThermalConstraint,
    ThermalConstraintKind
);

periodic_builtin!(
    structural_periodic_builtin,
    STRUCTURAL_PERIODIC_DESCRIPTOR,
    "fea.StructuralConstraint.periodic",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    StructuralConstraint,
    StructuralConstraintKind,
    frame: CoordinateFrame::Global
);
periodic_builtin!(
    thermal_periodic_builtin,
    THERMAL_PERIODIC_DESCRIPTOR,
    "fea.ThermalConstraint.periodic",
    FEA_THERMAL_CONSTRAINT_CLASS,
    ThermalConstraint,
    ThermalConstraintKind
);
periodic_builtin!(
    electrical_periodic_builtin,
    ELECTRICAL_PERIODIC_DESCRIPTOR,
    "fea.ElectricalConstraint.periodic",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    ElectricalConstraint,
    ElectricalConstraintKind
);
periodic_builtin!(
    magnetic_periodic_builtin,
    MAGNETIC_PERIODIC_DESCRIPTOR,
    "fea.MagneticConstraint.periodic",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    MagneticConstraint,
    MagneticConstraintKind
);
periodic_builtin!(
    acoustic_periodic_builtin,
    ACOUSTIC_PERIODIC_DESCRIPTOR,
    "fea.AcousticConstraint.periodic",
    FEA_ACOUSTIC_CONSTRAINT_CLASS,
    AcousticConstraint,
    AcousticConstraintKind
);
periodic_builtin!(
    fluid_periodic_builtin,
    FLUID_PERIODIC_DESCRIPTOR,
    "fea.FluidConstraint.periodic",
    FEA_FLUID_CONSTRAINT_CLASS,
    FluidConstraint,
    FluidConstraintKind
);

macro_rules! phase_periodic_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $constraint:ident, $kind:ident) => {
        #[runtime_builtin(
            name = $name,
            category = "fea",
            summary = "Create a typed phase-shifted periodic constraint for a complex harmonic field.",
            keywords = "fea,constraint,periodic,bloch,phase,transform,pairing,typed",
            descriptor(crate::builtins::fea::constraints::paired::$descriptor),
            integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
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
    MAGNETIC_PHASE_PERIODIC_DESCRIPTOR,
    "fea.MagneticConstraint.phaseShiftedPeriodic",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    MagneticConstraint,
    MagneticConstraintKind
);
phase_periodic_builtin!(
    acoustic_phase_periodic_builtin,
    ACOUSTIC_PHASE_PERIODIC_DESCRIPTOR,
    "fea.AcousticConstraint.phaseShiftedPeriodic",
    FEA_ACOUSTIC_CONSTRAINT_CLASS,
    AcousticConstraint,
    AcousticConstraintKind
);

#[runtime_builtin(
    name = "fea.ThermalConstraint.affineJump",
    category = "fea",
    summary = "Create a transformed periodic thermal relation with a prescribed temperature jump.",
    keywords = "fea,thermal,constraint,periodic,affine,jump,typed",
    descriptor(crate::builtins::fea::constraints::paired::AFFINE_JUMP_DESCRIPTOR),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
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
