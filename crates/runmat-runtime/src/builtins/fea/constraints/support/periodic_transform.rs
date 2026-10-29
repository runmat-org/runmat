use std::collections::HashMap;
use std::sync::Once;

use super::super::*;
use super::direction_from_value;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::{
    position_meters, quantity_scalar, translation_meters, POSITION_CLASS, ROTATION_CLASS,
    TRANSLATION_CLASS,
};
use crate::class_registry::declaration::{immutable_properties, static_methods};
use crate::execution::value_codec::ImmutableValueClassSchema;

const PERIODIC_TRANSFORM_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_PERIODIC_TRANSFORM_CLASS
        .display_name();
const IDENTITY_TRANSFORM_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_IDENTITY_PERIODIC_TRANSFORM_CLASS
        .display_name();
const TRANSLATION_TRANSFORM_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_TRANSLATION_PERIODIC_TRANSFORM_CLASS
        .display_name();
const ROTATION_TRANSFORM_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_ROTATION_PERIODIC_TRANSFORM_CLASS
        .display_name();
const EMPTY_FIELDS: &[&str] = &[];
const TRANSLATION_FIELDS: &[&str] = &["translation"];
const ROTATION_FIELDS: &[&str] = &["origin", "axis", "angle"];
#[runtime_builtin(
    name = "fea.PeriodicTransform.identity",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::fea::constraints::support::periodic_transform"
)]
pub async fn periodic_transform_identity_builtin(
    runtime: Option<crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PERIODIC_TRANSFORM_IDENTITY_DESCRIPTOR,
        "fea.PeriodicTransform.identity",
    )?;
    ensure_periodic_transform_classes_registered();
    transform_object(runtime.as_ref(), IDENTITY_TRANSFORM_CLASS, [])
}

#[runtime_builtin(
    name = "fea.PeriodicTransform.translation",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::fea::constraints::support::periodic_transform"
)]
pub async fn periodic_transform_translation_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    translation: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PERIODIC_TRANSFORM_TRANSLATION_DESCRIPTOR,
        "fea.PeriodicTransform.translation",
    )?;
    translation_meters("fea.PeriodicTransform.translation", &translation)?;
    transform_object(
        runtime.as_ref(),
        TRANSLATION_TRANSFORM_CLASS,
        [("translation", translation)],
    )
}

#[runtime_builtin(
    name = "fea.PeriodicTransform.rotation",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::fea::constraints::support::periodic_transform"
)]
pub async fn periodic_transform_rotation_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    origin: Value,
    axis: Value,
    angle: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PERIODIC_TRANSFORM_ROTATION_DESCRIPTOR,
        "fea.PeriodicTransform.rotation",
    )?;
    position_meters("fea.PeriodicTransform.rotation", &origin)?;
    direction_from_value("fea.PeriodicTransform.rotation", &axis)?;
    quantity_scalar(
        "fea.PeriodicTransform.rotation",
        &angle,
        ROTATION_CLASS,
        "radians",
    )?;
    transform_object(
        runtime.as_ref(),
        ROTATION_TRANSFORM_CLASS,
        [("origin", origin), ("axis", axis), ("angle", angle)],
    )
}

pub(crate) fn periodic_transform_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<PeriodicTransform> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "expected a typed fea.PeriodicTransform value",
        ));
    };
    let property = |name: &str| {
        object.properties().get(name).ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("{}.{} is required", object.class_name(), name),
            )
        })
    };
    match object.class_name().display_name() {
        IDENTITY_TRANSFORM_CLASS => Ok(PeriodicTransform::Identity),
        TRANSLATION_TRANSFORM_CLASS => Ok(PeriodicTransform::Translation(Translation::new(
            translation_meters(builtin, property("translation")?)?,
        ))),
        ROTATION_TRANSFORM_CLASS => Ok(PeriodicTransform::Rotation {
            origin: runmat_analysis_core::Position::new(position_meters(
                builtin,
                property("origin")?,
            )?),
            axis: direction_from_value(builtin, property("axis")?)?,
            angle: Rotation::new(quantity_scalar(
                builtin,
                property("angle")?,
                ROTATION_CLASS,
                "radians",
            )?),
        }),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("unsupported periodic-transform class `{other}`"),
        )),
    }
}

fn transform_object<const N: usize>(
    runtime: Option<&crate::context::RuntimeContext>,
    class_name: &str,
    properties: [(&str, Value); N],
) -> BuiltinResult<Value> {
    ensure_periodic_transform_classes_registered();
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        class_name,
        properties.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

pub(super) fn ensure_periodic_transform_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            PERIODIC_TRANSFORM_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut schemas = Vec::new();
        for schema in [
            ImmutableValueClassSchema {
                class_name: IDENTITY_TRANSFORM_CLASS,
                type_identity: "runmat.fea.periodic-transform.identity",
                schema_version: 1,
                allowed_fields: EMPTY_FIELDS,
                required_fields: EMPTY_FIELDS,
                validate: validate_identity,
            },
            ImmutableValueClassSchema {
                class_name: TRANSLATION_TRANSFORM_CLASS,
                type_identity: "runmat.fea.periodic-transform.translation",
                schema_version: 1,
                allowed_fields: TRANSLATION_FIELDS,
                required_fields: TRANSLATION_FIELDS,
                validate: validate_translation,
            },
            ImmutableValueClassSchema {
                class_name: ROTATION_TRANSFORM_CLASS,
                type_identity: "runmat.fea.periodic-transform.rotation",
                schema_version: 1,
                allowed_fields: ROTATION_FIELDS,
                required_fields: ROTATION_FIELDS,
                validate: validate_rotation,
            },
        ] {
            schemas.push(schema);
        }

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASSES.ensure(|| {
        register_value_class(
            IDENTITY_TRANSFORM_CLASS,
            immutable_properties(EMPTY_FIELDS),
            HashMap::new(),
        );
        register_value_class(
            TRANSLATION_TRANSFORM_CLASS,
            immutable_properties(TRANSLATION_FIELDS),
            HashMap::new(),
        );
        register_value_class(
            ROTATION_TRANSFORM_CLASS,
            immutable_properties(ROTATION_FIELDS),
            HashMap::new(),
        );
        register_value_class(
            PERIODIC_TRANSFORM_CLASS,
            immutable_properties(EMPTY_FIELDS),
            static_methods([
                ("identity", "fea.PeriodicTransform.identity"),
                ("translation", "fea.PeriodicTransform.translation"),
                ("rotation", "fea.PeriodicTransform.rotation"),
            ]),
        );
    });
}

fn validate_identity(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    Ok(())
}

fn validate_translation(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    validate_nested_classes(object, &[("translation", TRANSLATION_CLASS)])
}

fn validate_rotation(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    validate_nested_classes(
        object,
        &[
            ("origin", POSITION_CLASS),
            ("axis", "fea.Direction"),
            ("angle", ROTATION_CLASS),
        ],
    )
}

fn validate_nested_classes(
    object: &ObjectInstance,
    expected: &[(&str, &str)],
) -> Result<(), String> {
    for (field, class_name) in expected {
        if !matches!(
            object.properties().get(field),
            Some(Value::Object(value)) if value.class_name().display_name() == *class_name
        ) {
            return Err(format!(
                "{}.{} must be a typed {class_name}",
                object.class_name(),
                field
            ));
        }
    }
    Ok(())
}
