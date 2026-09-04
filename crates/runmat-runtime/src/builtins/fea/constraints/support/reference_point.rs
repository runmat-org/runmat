use std::collections::HashMap;
use std::sync::Once;

use super::super::*;
use super::{register_runtime_class, static_method};
use crate::builtins::fea::values::{position_meters, POSITION_CLASS};
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};
use runmat_analysis_core::{Position, ReferencePoint};

const REFERENCE_POINT_CLASS: &str = "fea.ReferencePoint";
const REFERENCE_POINT_FIELDS: &[&str] = &["reference_point_id", "position"];
const REFERENCE_POINT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference-point identifier.",
    },
    BuiltinParamDescriptor {
        name: "position",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Position coordinates.",
    },
];
const REFERENCE_POINT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "point = fea.ReferencePoint.fromCoordinates(id, position)",
    inputs: &REFERENCE_POINT_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const REFERENCE_POINT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REFERENCE_POINT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.ReferencePoint.fromCoordinates",
    category = "fea",
    summary = "Create a typed reference point from a position.",
    keywords = "fea,reference,point,position,coupling,typed",
    descriptor(
        crate::builtins::fea::constraints::support::reference_point::REFERENCE_POINT_DESCRIPTOR
    ),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::support::reference_point"
)]
pub async fn reference_point_from_coordinates_builtin(
    id: String,
    position: Value,
) -> BuiltinResult<Value> {
    validate_id("fea.ReferencePoint.fromCoordinates", &id)?;
    position_meters("fea.ReferencePoint.fromCoordinates", &position)?;
    ensure_reference_point_class_registered();
    let mut object = ObjectInstance::new(REFERENCE_POINT_CLASS.into());
    object
        .properties
        .insert("reference_point_id".into(), Value::String(id));
    object.properties.insert("position".into(), position);
    Ok(Value::Object(object))
}

pub(crate) fn reference_point_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ReferencePoint> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {REFERENCE_POINT_CLASS}"),
        ));
    };
    if object.class_name != REFERENCE_POINT_CLASS {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "expected {REFERENCE_POINT_CLASS}, got {}",
                object.class_name
            ),
        ));
    }
    let id = match object.properties.get("reference_point_id") {
        Some(Value::String(value)) if !value.trim().is_empty() => value.clone(),
        _ => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "fea.ReferencePoint.reference_point_id must be a non-empty string",
            ))
        }
    };
    let position = object.properties.get("position").ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            "fea.ReferencePoint.position is required",
        )
    })?;
    Ok(ReferencePoint {
        reference_point_id: id,
        position: Position::new(position_meters(builtin, position)?),
    })
}

pub(super) fn ensure_reference_point_class_registered() {
    static PORTABLE_SCHEMA: Once = Once::new();
    static RUNTIME_CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(REFERENCE_POINT_CLASS);
    PORTABLE_SCHEMA.call_once(|| {
        register_immutable_value_class(ImmutableValueClassSchema {
            class_name: REFERENCE_POINT_CLASS,
            type_identity: "runmat.fea.reference-point",
            schema_version: 1,
            allowed_fields: REFERENCE_POINT_FIELDS,
            required_fields: REFERENCE_POINT_FIELDS,
            validate: validate_reference_point_object,
        })
        .unwrap_or_else(|error| panic!("failed to register typed reference point: {error}"));
    });
    RUNTIME_CLASS.ensure(|| {
        register_runtime_class(
            REFERENCE_POINT_CLASS,
            REFERENCE_POINT_FIELDS,
            HashMap::from([(
                "fromCoordinates".into(),
                static_method("fromCoordinates", "fea.ReferencePoint.fromCoordinates"),
            )]),
        );
    });
}

fn validate_reference_point_object(object: &ObjectInstance) -> Result<(), String> {
    let id_valid = matches!(
        object.properties.get("reference_point_id"),
        Some(Value::String(value)) if !value.trim().is_empty()
    );
    let position_valid = matches!(
        object.properties.get("position"),
        Some(Value::Object(value)) if value.class_name == POSITION_CLASS
    );
    if !id_valid || !position_valid {
        return Err(
            "fea.ReferencePoint requires a non-empty identifier and typed fea.Position".into(),
        );
    }
    Ok(())
}

fn validate_id(builtin: &'static str, id: &str) -> BuiltinResult<()> {
    if id.trim().is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "reference-point id must not be empty",
        ));
    }
    Ok(())
}
