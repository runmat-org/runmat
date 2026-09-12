use std::collections::{HashMap, HashSet};
use std::sync::Once;

use super::super::*;
use super::{register_runtime_class, static_method};
use crate::builtins::fea::values::{quantity_scalar, DISPLACEMENT_CLASS, ROTATION_CLASS};
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};

const STRUCTURAL_DOF_CLASS: &str = "fea.StructuralDof";
const STRUCTURAL_DOF_VALUE_CLASS: &str = "fea.StructuralDofValue";
const TRANSLATION_DOF_CLASS: &str = "fea.StructuralTranslationDof";
const ROTATION_DOF_CLASS: &str = "fea.StructuralRotationDof";
const TRANSLATION_VALUE_CLASS: &str = "fea.StructuralTranslationDofValue";
const ROTATION_VALUE_CLASS: &str = "fea.StructuralRotationDofValue";
const AXIS_FIELDS: &[&str] = &["axis"];
const DOF_VALUE_FIELDS: &[&str] = &["dof", "value"];

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const DOF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dof",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed structural translation or rotation degree of freedom.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimensioned displacement or rotation value matching the DOF family.",
    },
];
const NO_ARG_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dof = fea.StructuralDof.component()",
    inputs: &NO_INPUTS,
    outputs: &OUT_ANY,
}];
const DOF_VALUE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "entry = fea.StructuralDofValue.component(dof, value)",
    inputs: &DOF_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const DOF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NO_ARG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub(super) const DOF_VALUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOF_VALUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

macro_rules! dof_builtin {
    ($function:ident, $name:literal, $class:expr, $axis:expr) => {
        #[runtime_builtin(
                    name = $name,
                    category = "fea",
                    summary = "Create a typed structural degree of freedom.",
                    keywords = "fea,structural,dof,typed",
                    descriptor(crate::builtins::fea::constraints::support::dof::DOF_DESCRIPTOR),
                    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
                    builtin_path = "crate::builtins::fea::constraints::support::dof"
                )]
        pub async fn $function() -> BuiltinResult<Value> {
            ensure_dof_classes_registered();
            let mut object = ObjectInstance::new($class.into());
            object
                .properties
                .insert("axis".into(), Value::Int(IntValue::U8($axis)));
            Ok(Value::Object(object))
        }
    };
}

dof_builtin!(
    structural_dof_ux_builtin,
    "fea.StructuralDof.ux",
    TRANSLATION_DOF_CLASS,
    0
);
dof_builtin!(
    structural_dof_uy_builtin,
    "fea.StructuralDof.uy",
    TRANSLATION_DOF_CLASS,
    1
);
dof_builtin!(
    structural_dof_uz_builtin,
    "fea.StructuralDof.uz",
    TRANSLATION_DOF_CLASS,
    2
);
dof_builtin!(
    structural_dof_rx_builtin,
    "fea.StructuralDof.rx",
    ROTATION_DOF_CLASS,
    0
);
dof_builtin!(
    structural_dof_ry_builtin,
    "fea.StructuralDof.ry",
    ROTATION_DOF_CLASS,
    1
);
dof_builtin!(
    structural_dof_rz_builtin,
    "fea.StructuralDof.rz",
    ROTATION_DOF_CLASS,
    2
);

#[runtime_builtin(
    name = "fea.StructuralDofValue.translation",
    category = "fea",
    summary = "Bind a structural translation DOF to a displacement value.",
    keywords = "fea,structural,dof,displacement,typed",
    descriptor(crate::builtins::fea::constraints::support::dof::DOF_VALUE_DESCRIPTOR),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::support::dof"
)]
pub async fn structural_translation_value_builtin(
    dof: Value,
    value: Value,
) -> BuiltinResult<Value> {
    translation_dof_from_value("fea.StructuralDofValue.translation", &dof)?;
    quantity_scalar(
        "fea.StructuralDofValue.translation",
        &value,
        DISPLACEMENT_CLASS,
        "meters",
    )?;
    dof_value_object(TRANSLATION_VALUE_CLASS, dof, value)
}

#[runtime_builtin(
    name = "fea.StructuralDofValue.rotation",
    category = "fea",
    summary = "Bind a structural rotation DOF to a rotation value.",
    keywords = "fea,structural,dof,rotation,typed",
    descriptor(crate::builtins::fea::constraints::support::dof::DOF_VALUE_DESCRIPTOR),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::support::dof"
)]
pub async fn structural_rotation_value_builtin(dof: Value, value: Value) -> BuiltinResult<Value> {
    rotation_dof_from_value("fea.StructuralDofValue.rotation", &dof)?;
    quantity_scalar(
        "fea.StructuralDofValue.rotation",
        &value,
        ROTATION_CLASS,
        "radians",
    )?;
    dof_value_object(ROTATION_VALUE_CLASS, dof, value)
}

pub(crate) fn structural_dof_values_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<(Vec<TranslationDofValue>, Vec<RotationDofValue>)> {
    let entries = match value {
        Value::Cell(cell) => cell.data.iter().collect::<Vec<_>>(),
        value => vec![value],
    };
    if entries.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "at least one structural DOF value is required",
        ));
    }
    let mut translations = Vec::new();
    let mut rotations = Vec::new();
    for entry in entries {
        let Value::Object(object) = entry else {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "DOF values must be typed fea.StructuralDofValue objects",
            ));
        };
        let dof = object.properties.get("dof").ok_or_else(|| {
            builtin_error(builtin, &ERROR_INPUT, "DOF value is missing its typed DOF")
        })?;
        let value = object.properties.get("value").ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "DOF value is missing its dimensioned value",
            )
        })?;
        match object.class_name.as_str() {
            TRANSLATION_VALUE_CLASS => translations.push(TranslationDofValue {
                dof: translation_dof_from_value(builtin, dof)?,
                value: PrescribedValue::Constant(Displacement::new(quantity_scalar(
                    builtin,
                    value,
                    DISPLACEMENT_CLASS,
                    "meters",
                )?)),
            }),
            ROTATION_VALUE_CLASS => rotations.push(RotationDofValue {
                dof: rotation_dof_from_value(builtin, dof)?,
                value: PrescribedValue::Constant(Rotation::new(quantity_scalar(
                    builtin,
                    value,
                    ROTATION_CLASS,
                    "radians",
                )?)),
            }),
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported structural DOF value class `{other}`"),
                ))
            }
        }
    }
    if translations
        .iter()
        .map(|entry| entry.dof)
        .collect::<HashSet<_>>()
        .len()
        != translations.len()
        || rotations
            .iter()
            .map(|entry| entry.dof)
            .collect::<HashSet<_>>()
            .len()
            != rotations.len()
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "a prescribed structural constraint cannot contain duplicate DOFs",
        ));
    }
    Ok((translations, rotations))
}

fn dof_value_object(class_name: &str, dof: Value, value: Value) -> BuiltinResult<Value> {
    ensure_dof_classes_registered();
    let mut object = ObjectInstance::new(class_name.into());
    object.properties.insert("dof".into(), dof);
    object.properties.insert("value".into(), value);
    Ok(Value::Object(object))
}

pub(super) fn translation_dof_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<StructuralTranslationDof> {
    let axis = dof_axis(builtin, value, TRANSLATION_DOF_CLASS)?;
    Ok(match axis {
        0 => StructuralTranslationDof::Ux,
        1 => StructuralTranslationDof::Uy,
        2 => StructuralTranslationDof::Uz,
        _ => unreachable!("DOF schemas admit only three axes"),
    })
}

fn rotation_dof_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<StructuralRotationDof> {
    let axis = dof_axis(builtin, value, ROTATION_DOF_CLASS)?;
    Ok(match axis {
        0 => StructuralRotationDof::Rx,
        1 => StructuralRotationDof::Ry,
        2 => StructuralRotationDof::Rz,
        _ => unreachable!("DOF schemas admit only three axes"),
    })
}

fn dof_axis(builtin: &'static str, value: &Value, class_name: &str) -> BuiltinResult<u8> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}"),
        ));
    };
    if object.class_name != class_name {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}, got {}", object.class_name),
        ));
    }
    match object.properties.get("axis") {
        Some(Value::Int(IntValue::U8(axis @ 0..=2))) => Ok(*axis),
        _ => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{class_name}.axis must be a uint8 value from 0 through 2"),
        )),
    }
}

pub(super) fn ensure_dof_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(STRUCTURAL_DOF_CLASS);
    PORTABLE_SCHEMAS.call_once(|| {
        for schema in [
            ImmutableValueClassSchema {
                class_name: TRANSLATION_DOF_CLASS,
                type_identity: "runmat.fea.structural-translation-dof",
                schema_version: 1,
                allowed_fields: AXIS_FIELDS,
                required_fields: AXIS_FIELDS,
                validate: validate_axis_object,
            },
            ImmutableValueClassSchema {
                class_name: ROTATION_DOF_CLASS,
                type_identity: "runmat.fea.structural-rotation-dof",
                schema_version: 1,
                allowed_fields: AXIS_FIELDS,
                required_fields: AXIS_FIELDS,
                validate: validate_axis_object,
            },
            ImmutableValueClassSchema {
                class_name: TRANSLATION_VALUE_CLASS,
                type_identity: "runmat.fea.structural-translation-dof-value",
                schema_version: 1,
                allowed_fields: DOF_VALUE_FIELDS,
                required_fields: DOF_VALUE_FIELDS,
                validate: validate_translation_value_object,
            },
            ImmutableValueClassSchema {
                class_name: ROTATION_VALUE_CLASS,
                type_identity: "runmat.fea.structural-rotation-dof-value",
                schema_version: 1,
                allowed_fields: DOF_VALUE_FIELDS,
                required_fields: DOF_VALUE_FIELDS,
                validate: validate_rotation_value_object,
            },
        ] {
            register_immutable_value_class(schema).unwrap_or_else(|error| {
                panic!("failed to register typed structural DOF class: {error}")
            });
        }
    });
    RUNTIME_CLASSES.ensure(|| {
        register_runtime_class(STRUCTURAL_DOF_CLASS, &[], dof_methods());
        register_runtime_class(STRUCTURAL_DOF_VALUE_CLASS, &[], dof_value_methods());
        register_runtime_class(TRANSLATION_DOF_CLASS, AXIS_FIELDS, HashMap::new());
        register_runtime_class(ROTATION_DOF_CLASS, AXIS_FIELDS, HashMap::new());
        register_runtime_class(TRANSLATION_VALUE_CLASS, DOF_VALUE_FIELDS, HashMap::new());
        register_runtime_class(ROTATION_VALUE_CLASS, DOF_VALUE_FIELDS, HashMap::new());
    });
}

fn dof_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    ["ux", "uy", "uz", "rx", "ry", "rz"]
        .into_iter()
        .map(|name| {
            (
                name.into(),
                static_method(name, &format!("fea.StructuralDof.{name}")),
            )
        })
        .collect()
}

fn dof_value_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    ["translation", "rotation"]
        .into_iter()
        .map(|name| {
            (
                name.into(),
                static_method(name, &format!("fea.StructuralDofValue.{name}")),
            )
        })
        .collect()
}

fn validate_axis_object(object: &ObjectInstance) -> Result<(), String> {
    match object.properties.get("axis") {
        Some(Value::Int(IntValue::U8(0..=2))) => Ok(()),
        _ => Err(format!(
            "{}.axis must be a uint8 value from 0 through 2",
            object.class_name
        )),
    }
}

fn validate_translation_value_object(object: &ObjectInstance) -> Result<(), String> {
    validate_dof_value_object(object, TRANSLATION_DOF_CLASS, DISPLACEMENT_CLASS)
}

fn validate_rotation_value_object(object: &ObjectInstance) -> Result<(), String> {
    validate_dof_value_object(object, ROTATION_DOF_CLASS, ROTATION_CLASS)
}

fn validate_dof_value_object(
    object: &ObjectInstance,
    dof_class: &str,
    value_class: &str,
) -> Result<(), String> {
    let nested_class = |name: &str| match object.properties.get(name) {
        Some(Value::Object(value)) => Some(value.class_name.as_str()),
        _ => None,
    };
    if nested_class("dof") != Some(dof_class) || nested_class("value") != Some(value_class) {
        return Err(format!(
            "{} requires a {dof_class} and {value_class}",
            object.class_name
        ));
    }
    Ok(())
}
