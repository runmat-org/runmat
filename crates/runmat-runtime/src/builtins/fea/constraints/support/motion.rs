use std::collections::{HashMap, HashSet};
use std::sync::Once;

use super::super::*;
use super::dof::translation_dof_from_value;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::{quantity_scalar, DISPLACEMENT_CLASS, VELOCITY_CLASS};
use crate::class_registry::declaration::{immutable_properties, static_methods};
use crate::execution::value_codec::ImmutableValueClassSchema;
use runmat_analysis_core::{
    AccelerationSignalId, CoordinateFrame, Displacement, PhysicalQuantityKind, SignalDomainKind,
    SignalPrecision, SignalShape, StructuralKinematicHistory, StructuralMotionReconstruction,
    TranslationDofMotion, Velocity,
};

const MOTION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_MOTION_CLASS.display_name();
const TRANSLATION_MOTION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_TRANSLATION_MOTION_CLASS
        .display_name();
const MOTION_FIELDS: &[&str] = &["dof", "history"];

#[runtime_builtin(
    name = "fea.StructuralMotion.acceleration",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::constraints::support::motion"
)]
pub async fn structural_acceleration_motion_builtin(
    dof: Value,
    acceleration: Value,
    initial_displacement: Value,
    initial_velocity: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_STRUCTURAL_MOTION_ACCELERATION_DESCRIPTOR,
        "fea.StructuralMotion.acceleration",
    )?;
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
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
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
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            MOTION_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let schemas = vec![ImmutableValueClassSchema {
            class_name: TRANSLATION_MOTION_CLASS,
            type_identity: "runmat.fea.structural-translation-motion",
            schema_version: 1,
            allowed_fields: MOTION_FIELDS,
            required_fields: MOTION_FIELDS,
            validate: crate::builtins::fea::object_projection::validate,
        }];

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASSES.ensure(|| {
        register_value_class(
            MOTION_CLASS,
            HashMap::new(),
            static_methods([("acceleration", "fea.StructuralMotion.acceleration")]),
        );
        register_value_class(
            TRANSLATION_MOTION_CLASS,
            immutable_properties(MOTION_FIELDS),
            HashMap::new(),
        );
    });
}
