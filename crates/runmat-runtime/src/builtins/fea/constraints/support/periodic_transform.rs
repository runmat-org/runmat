use std::collections::HashMap;
use std::sync::Once;

use super::super::*;
use super::{direction_from_value, register_runtime_class, static_method};
use crate::builtins::fea::values::{
    position_meters, quantity_scalar, translation_meters, POSITION_CLASS, ROTATION_CLASS,
    TRANSLATION_CLASS,
};
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};

const PERIODIC_TRANSFORM_CLASS: &str = "fea.PeriodicTransform";
const IDENTITY_TRANSFORM_CLASS: &str = "fea.IdentityPeriodicTransform";
const TRANSLATION_TRANSFORM_CLASS: &str = "fea.TranslationPeriodicTransform";
const ROTATION_TRANSFORM_CLASS: &str = "fea.RotationPeriodicTransform";
const EMPTY_FIELDS: &[&str] = &[];
const TRANSLATION_FIELDS: &[&str] = &["translation"];
const ROTATION_FIELDS: &[&str] = &["origin", "axis", "angle"];
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const TRANSLATION_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "translation",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed fea.Translation vector from primary to secondary geometry.",
}];
const ROTATION_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "origin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Position on the rotation axis.",
    },
    BuiltinParamDescriptor {
        name: "axis",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Direction rotation axis.",
    },
    BuiltinParamDescriptor {
        name: "angle",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Rotation angle from primary to secondary geometry.",
    },
];
const IDENTITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.identity()",
    inputs: &NO_INPUTS,
    outputs: &OUT_ANY,
}];
const TRANSLATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.translation(translation)",
    inputs: &TRANSLATION_INPUTS,
    outputs: &OUT_ANY,
}];
const ROTATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "transform = fea.PeriodicTransform.rotation(origin, axis, angle)",
    inputs: &ROTATION_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const IDENTITY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IDENTITY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub(super) const TRANSLATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRANSLATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub(super) const ROTATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ROTATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.PeriodicTransform.identity",
    category = "fea",
    summary = "Create an identity periodic transform.",
    keywords = "fea,periodic,transform,identity,typed",
    descriptor(
        crate::builtins::fea::constraints::support::periodic_transform::IDENTITY_DESCRIPTOR
    ),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::support::periodic_transform"
)]
pub async fn periodic_transform_identity_builtin() -> BuiltinResult<Value> {
    ensure_periodic_transform_classes_registered();
    Ok(Value::Object(ObjectInstance::new(
        IDENTITY_TRANSFORM_CLASS.into(),
    )))
}

#[runtime_builtin(
    name = "fea.PeriodicTransform.translation",
    category = "fea",
    summary = "Create a translational periodic transform.",
    keywords = "fea,periodic,transform,translation,typed",
    descriptor(
        crate::builtins::fea::constraints::support::periodic_transform::TRANSLATION_DESCRIPTOR
    ),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::support::periodic_transform"
)]
pub async fn periodic_transform_translation_builtin(translation: Value) -> BuiltinResult<Value> {
    translation_meters("fea.PeriodicTransform.translation", &translation)?;
    transform_object(TRANSLATION_TRANSFORM_CLASS, [("translation", translation)])
}

#[runtime_builtin(
    name = "fea.PeriodicTransform.rotation",
    category = "fea",
    summary = "Create a rotational periodic transform.",
    keywords = "fea,periodic,transform,rotation,axis,typed",
    descriptor(
        crate::builtins::fea::constraints::support::periodic_transform::ROTATION_DESCRIPTOR
    ),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::support::periodic_transform"
)]
pub async fn periodic_transform_rotation_builtin(
    origin: Value,
    axis: Value,
    angle: Value,
) -> BuiltinResult<Value> {
    position_meters("fea.PeriodicTransform.rotation", &origin)?;
    direction_from_value("fea.PeriodicTransform.rotation", &axis)?;
    quantity_scalar(
        "fea.PeriodicTransform.rotation",
        &angle,
        ROTATION_CLASS,
        "radians",
    )?;
    transform_object(
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
        object.properties.get(name).ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("{}.{} is required", object.class_name, name),
            )
        })
    };
    match object.class_name.as_str() {
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
    class_name: &str,
    properties: [(&str, Value); N],
) -> BuiltinResult<Value> {
    ensure_periodic_transform_classes_registered();
    let mut object = ObjectInstance::new(class_name.into());
    for (name, value) in properties {
        object.properties.insert(name.into(), value);
    }
    Ok(Value::Object(object))
}

pub(super) fn ensure_periodic_transform_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(PERIODIC_TRANSFORM_CLASS);
    PORTABLE_SCHEMAS.call_once(|| {
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
            register_immutable_value_class(schema)
                .unwrap_or_else(|error| panic!("failed to register periodic transform: {error}"));
        }
    });
    RUNTIME_CLASSES.ensure(|| {
        register_runtime_class(IDENTITY_TRANSFORM_CLASS, EMPTY_FIELDS, HashMap::new());
        register_runtime_class(
            TRANSLATION_TRANSFORM_CLASS,
            TRANSLATION_FIELDS,
            HashMap::new(),
        );
        register_runtime_class(ROTATION_TRANSFORM_CLASS, ROTATION_FIELDS, HashMap::new());
        register_runtime_class(
            PERIODIC_TRANSFORM_CLASS,
            EMPTY_FIELDS,
            [
                ("identity", "fea.PeriodicTransform.identity"),
                ("translation", "fea.PeriodicTransform.translation"),
                ("rotation", "fea.PeriodicTransform.rotation"),
            ]
            .into_iter()
            .map(|(name, function)| (name.into(), static_method(name, function)))
            .collect(),
        );
    });
}

fn validate_identity(_: &ObjectInstance) -> Result<(), String> {
    Ok(())
}

fn validate_translation(object: &ObjectInstance) -> Result<(), String> {
    validate_nested_classes(object, &[("translation", TRANSLATION_CLASS)])
}

fn validate_rotation(object: &ObjectInstance) -> Result<(), String> {
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
            object.properties.get(*field),
            Some(Value::Object(value)) if value.class_name == *class_name
        ) {
            return Err(format!(
                "{}.{} must be a typed {class_name}",
                object.class_name, field
            ));
        }
    }
    Ok(())
}
