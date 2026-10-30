use std::sync::Once;

use super::super::*;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::{position_meters, POSITION_CLASS};
use crate::class_registry::declaration::{immutable_properties, static_methods};
use crate::execution::value_codec::ImmutableValueClassSchema;
use runmat_analysis_core::{Position, ReferencePoint};

const REFERENCE_POINT_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_REFERENCE_POINT_CLASS.display_name();
const REFERENCE_POINT_FIELDS: &[&str] = &["reference_point_id", "position"];
#[runtime_builtin(
    name = "fea.ReferencePoint.fromCoordinates",
    builtin_path = "crate::builtins::fea::constraints::support::reference_point",
    context = true,
    binding_variant = "default"
)]
pub async fn reference_point_from_coordinates_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    id: runmat_value::Utf8Text,
    position: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_REFERENCE_POINT_FROM_COORDINATES_DESCRIPTOR,
        "fea.ReferencePoint.fromCoordinates",
    )?;
    validate_id("fea.ReferencePoint.fromCoordinates", id.as_str())?;
    position_meters("fea.ReferencePoint.fromCoordinates", &position)?;
    ensure_reference_point_class_registered();
    let fields = [
        (
            "reference_point_id",
            Value::String(runmat_value::StringElement::Text(id)),
        ),
        ("position", position),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime.as_ref(),
        REFERENCE_POINT_CLASS,
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
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
    if !object
        .class_name()
        .is(runmat_builtins::fea_contract::model_types::classes::FEA_REFERENCE_POINT_CLASS)
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "expected {REFERENCE_POINT_CLASS}, got {}",
                object.class_name()
            ),
        ));
    }
    let id = match object.properties().get("reference_point_id") {
        Some(Value::String(runmat_value::StringElement::Text(value)))
            if !value.trim().is_empty() =>
        {
            value.to_string()
        }
        _ => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "fea.ReferencePoint.reference_point_id must be a non-empty string",
            ))
        }
    };
    let position = object.properties().get("position").ok_or_else(|| {
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
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            REFERENCE_POINT_CLASS,
        ));
    PORTABLE_SCHEMA.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let schemas = vec![ImmutableValueClassSchema {
            class_name: REFERENCE_POINT_CLASS,
            type_identity: "runmat.fea.reference-point",
            schema_version: 1,
            allowed_fields: REFERENCE_POINT_FIELDS,
            required_fields: REFERENCE_POINT_FIELDS,
            validate: validate_reference_point_object,
        }];

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASS.ensure(|| {
        register_value_class(
            REFERENCE_POINT_CLASS,
            immutable_properties(REFERENCE_POINT_FIELDS),
            static_methods([("fromCoordinates", "fea.ReferencePoint.fromCoordinates")]),
        );
    });
}

fn validate_reference_point_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    let id_valid = matches!(
        object.properties().get("reference_point_id"),
        Some(Value::String(runmat_value::StringElement::Text(value))) if !value.trim().is_empty()
    );
    let position_valid = matches!(
        object.properties().get("position"),
        Some(Value::Object(value)) if value.class_name().display_name() == POSITION_CLASS
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
