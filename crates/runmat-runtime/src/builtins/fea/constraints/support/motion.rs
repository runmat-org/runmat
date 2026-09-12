use std::collections::{HashMap, HashSet};
use std::sync::Once;

use super::super::*;
use super::{dof::translation_dof_from_value, register_runtime_class, static_method};
use crate::builtins::fea::values::{quantity_scalar, DISPLACEMENT_CLASS, VELOCITY_CLASS};
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};
use runmat_analysis_core::{
    AccelerationSignalId, CoordinateFrame, Displacement, PhysicalQuantityKind, SignalDomainKind,
    SignalPrecision, SignalShape, StructuralKinematicHistory, StructuralMotionReconstruction,
    TranslationDofMotion, Velocity,
};

const MOTION_CLASS: &str = "fea.StructuralMotion";
const TRANSLATION_MOTION_CLASS: &str = "fea.StructuralTranslationMotion";
const MOTION_FIELDS: &[&str] = &["dof", "history"];

const ACCELERATION_INPUTS: [BuiltinParamDescriptor; 4] = [
    required("dof", "Typed translational structural degree of freedom."),
    required(
        "acceleration",
        "Real64 three-component time-domain acceleration signal.",
    ),
    required(
        "initialDisplacement",
        "Support displacement at the start of the step.",
    ),
    required(
        "initialVelocity",
        "Support velocity at the start of the step.",
    ),
];
const ACCELERATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "motion = fea.StructuralMotion.acceleration(dof, acceleration, initialDisplacement, initialVelocity)",
    inputs: &ACCELERATION_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const ACCELERATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ACCELERATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const fn required(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

#[runtime_builtin(
    name = "fea.StructuralMotion.acceleration",
    category = "fea",
    summary = "Create derivative-consistent support motion from an acceleration history.",
    keywords = "fea,structural,constraint,support,motion,acceleration,transient",
    descriptor(crate::builtins::fea::constraints::support::motion::ACCELERATION_DESCRIPTOR),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::support::motion"
)]
pub async fn structural_acceleration_motion_builtin(
    dof: Value,
    acceleration: Value,
    initial_displacement: Value,
    initial_velocity: Value,
) -> BuiltinResult<Value> {
    let builtin = "fea.StructuralMotion.acceleration";
    let dof = translation_dof_from_value(builtin, &dof)?;
    let acceleration =
        crate::builtins::fea::signals::physical_signal_from_value(builtin, &acceleration)?;
    if acceleration.quantity() != PhysicalQuantityKind::Acceleration
        || acceleration.shape() != &SignalShape::Vector3
        || acceleration.frame() != Some(&CoordinateFrame::Global)
        || acceleration.domain() != SignalDomainKind::Time
        || acceleration.precision() != SignalPrecision::Real64
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "support motion requires a global-frame real64 three-component time-domain acceleration signal",
        ));
    }
    let initial_displacement = Displacement::new(quantity_scalar(
        builtin,
        &initial_displacement,
        DISPLACEMENT_CLASS,
        "meters",
    )?);
    let initial_velocity = Velocity::new(quantity_scalar(
        builtin,
        &initial_velocity,
        VELOCITY_CLASS,
        "meters_per_second",
    )?);
    ensure_motion_classes_registered();
    serializable_to_object_value(
        builtin,
        &ERROR_INTERNAL,
        TRANSLATION_MOTION_CLASS,
        &TranslationDofMotion {
            dof,
            history: StructuralKinematicHistory::AccelerationDriven {
                acceleration: AccelerationSignalId(acceleration.signal_id().clone()),
                initial_displacement,
                initial_velocity,
                reconstruction: StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
            },
        },
    )
    .map(Value::Object)
}

pub(crate) fn structural_motions_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<TranslationDofMotion>> {
    let entries = match value {
        Value::Cell(cell) => cell.data.iter().collect::<Vec<_>>(),
        value => vec![value],
    };
    if entries.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "at least one typed structural motion is required",
        ));
    }
    let mut motions: Vec<TranslationDofMotion> = Vec::with_capacity(entries.len());
    for entry in entries {
        motions.push(object_payload(builtin, entry, TRANSLATION_MOTION_CLASS)?);
    }
    if motions
        .iter()
        .map(|motion| motion.dof)
        .collect::<HashSet<_>>()
        .len()
        != motions.len()
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "a prescribed support-motion constraint cannot contain duplicate DOFs",
        ));
    }
    Ok(motions)
}

pub(super) fn ensure_motion_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(MOTION_CLASS);
    PORTABLE_SCHEMAS.call_once(|| {
        register_immutable_value_class(ImmutableValueClassSchema {
            class_name: TRANSLATION_MOTION_CLASS,
            type_identity: "runmat.fea.structural-translation-motion",
            schema_version: 1,
            allowed_fields: MOTION_FIELDS,
            required_fields: MOTION_FIELDS,
            validate: |_| Ok(()),
        })
        .unwrap_or_else(|error| panic!("failed to register structural motion class: {error}"));
    });
    RUNTIME_CLASSES.ensure(|| {
        register_runtime_class(
            MOTION_CLASS,
            &[],
            HashMap::from([(
                "acceleration".into(),
                static_method("acceleration", "fea.StructuralMotion.acceleration"),
            )]),
        );
        register_runtime_class(TRANSLATION_MOTION_CLASS, MOTION_FIELDS, HashMap::new());
    });
}
