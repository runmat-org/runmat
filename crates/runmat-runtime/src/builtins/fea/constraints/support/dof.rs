use std::collections::{HashMap, HashSet};
use std::sync::Once;

use super::super::*;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::{quantity_scalar, DISPLACEMENT_CLASS, ROTATION_CLASS};
use crate::class_registry::declaration::{immutable_properties, method};
use crate::class_registry::RuntimeMethod;
use crate::execution::value_codec::ImmutableValueClassSchema;
use runmat_types::MethodName;

const STRUCTURAL_DOF_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_DOF_CLASS.display_name();
const STRUCTURAL_DOF_VALUE_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_DOF_VALUE_CLASS
        .display_name();
const TRANSLATION_DOF_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_TRANSLATION_DOF_CLASS
        .display_name();
const ROTATION_DOF_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_ROTATION_DOF_CLASS
        .display_name();
const TRANSLATION_VALUE_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_TRANSLATION_DOF_VALUE_CLASS
        .display_name();
const ROTATION_VALUE_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_STRUCTURAL_ROTATION_DOF_VALUE_CLASS
        .display_name();
const AXIS_FIELDS: &[&str] = &["axis"];
const DOF_VALUE_FIELDS: &[&str] = &["dof", "value"];

macro_rules! dof_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $axis:expr) => {
        #[runtime_builtin(
                            name = $name,
                            builtin_path = "crate::builtins::fea::constraints::support::dof",
                            context = true,
                            binding_variant = "default"
                        )]
        pub async fn $function(
            runtime: Option<crate::context::RuntimeContext>,
        ) -> BuiltinResult<Value> {
            crate::builtins::fea::constraints::fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            ensure_dof_classes_registered();
            let axis = Value::Int(IntValue::U8($axis));
            let fields = [("axis", axis)];
            let object = crate::execution::object::construct_object_from_borrowed_fields(
                runtime.as_ref(),
                $class,
                fields.iter().map(|(name, value)| (*name, value)),
            )?;
            Ok(object.into_value_and_lease().0)
        }
    };
}

dof_builtin!(
    structural_dof_ux_builtin,
    FEA_STRUCTURAL_DOF_UX_DESCRIPTOR,
    "fea.StructuralDof.ux",
    TRANSLATION_DOF_CLASS,
    0
);
dof_builtin!(
    structural_dof_uy_builtin,
    FEA_STRUCTURAL_DOF_UY_DESCRIPTOR,
    "fea.StructuralDof.uy",
    TRANSLATION_DOF_CLASS,
    1
);
dof_builtin!(
    structural_dof_uz_builtin,
    FEA_STRUCTURAL_DOF_UZ_DESCRIPTOR,
    "fea.StructuralDof.uz",
    TRANSLATION_DOF_CLASS,
    2
);
dof_builtin!(
    structural_dof_rx_builtin,
    FEA_STRUCTURAL_DOF_RX_DESCRIPTOR,
    "fea.StructuralDof.rx",
    ROTATION_DOF_CLASS,
    0
);
dof_builtin!(
    structural_dof_ry_builtin,
    FEA_STRUCTURAL_DOF_RY_DESCRIPTOR,
    "fea.StructuralDof.ry",
    ROTATION_DOF_CLASS,
    1
);
dof_builtin!(
    structural_dof_rz_builtin,
    FEA_STRUCTURAL_DOF_RZ_DESCRIPTOR,
    "fea.StructuralDof.rz",
    ROTATION_DOF_CLASS,
    2
);

#[runtime_builtin(
    name = "fea.StructuralDofValue.translation",
    builtin_path = "crate::builtins::fea::constraints::support::dof",
    context = true,
    binding_variant = "default"
)]
pub async fn structural_translation_value_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    dof: Value,
    value: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_STRUCTURAL_DOF_VALUE_TRANSLATION_DESCRIPTOR,
        "fea.StructuralDofValue.translation",
    )?;
    translation_dof_from_value("fea.StructuralDofValue.translation", &dof)?;
    quantity_scalar(
        "fea.StructuralDofValue.translation",
        &value,
        DISPLACEMENT_CLASS,
        "meters",
    )?;
    dof_value_object(runtime.as_ref(), TRANSLATION_VALUE_CLASS, dof, value)
}

#[runtime_builtin(
    name = "fea.StructuralDofValue.rotation",
    builtin_path = "crate::builtins::fea::constraints::support::dof",
    context = true,
    binding_variant = "default"
)]
pub async fn structural_rotation_value_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    dof: Value,
    value: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_STRUCTURAL_DOF_VALUE_ROTATION_DESCRIPTOR,
        "fea.StructuralDofValue.rotation",
    )?;
    rotation_dof_from_value("fea.StructuralDofValue.rotation", &dof)?;
    quantity_scalar(
        "fea.StructuralDofValue.rotation",
        &value,
        ROTATION_CLASS,
        "radians",
    )?;
    dof_value_object(runtime.as_ref(), ROTATION_VALUE_CLASS, dof, value)
}

pub(crate) fn structural_dof_values_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<(Vec<TranslationDofValue>, Vec<RotationDofValue>)> {
    let entries = match value {
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
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
        let dof = object.properties().get("dof").ok_or_else(|| {
            builtin_error(builtin, &ERROR_INPUT, "DOF value is missing its typed DOF")
        })?;
        let value = object.properties().get("value").ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "DOF value is missing its dimensioned value",
            )
        })?;
        match object.class_name().display_name() {
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

fn dof_value_object(
    runtime: Option<&crate::context::RuntimeContext>,
    class_name: &str,
    dof: Value,
    value: Value,
) -> BuiltinResult<Value> {
    ensure_dof_classes_registered();
    let fields = [("dof", dof), ("value", value)];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        class_name,
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
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
    if object.class_name().display_name() != class_name {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {class_name}, got {}", object.class_name()),
        ));
    }
    match object.properties().get("axis") {
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
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            STRUCTURAL_DOF_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut schemas = Vec::new();
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
            schemas.push(schema);
        }

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASSES.ensure(|| {
        register_value_class(STRUCTURAL_DOF_CLASS, HashMap::new(), dof_methods());
        register_value_class(
            STRUCTURAL_DOF_VALUE_CLASS,
            HashMap::new(),
            dof_value_methods(),
        );
        register_value_class(
            TRANSLATION_DOF_CLASS,
            immutable_properties(AXIS_FIELDS),
            HashMap::new(),
        );
        register_value_class(
            ROTATION_DOF_CLASS,
            immutable_properties(AXIS_FIELDS),
            HashMap::new(),
        );
        register_value_class(
            TRANSLATION_VALUE_CLASS,
            immutable_properties(DOF_VALUE_FIELDS),
            HashMap::new(),
        );
        register_value_class(
            ROTATION_VALUE_CLASS,
            immutable_properties(DOF_VALUE_FIELDS),
            HashMap::new(),
        );
    });
}

fn dof_methods() -> HashMap<MethodName, RuntimeMethod> {
    ["ux", "uy", "uz", "rx", "ry", "rz"]
        .into_iter()
        .map(|name| method(name, &format!("fea.StructuralDof.{name}"), true))
        .collect()
}

fn dof_value_methods() -> HashMap<MethodName, RuntimeMethod> {
    ["translation", "rotation"]
        .into_iter()
        .map(|name| method(name, &format!("fea.StructuralDofValue.{name}"), true))
        .collect()
}

fn validate_axis_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    match object.properties().get("axis") {
        Some(Value::Int(IntValue::U8(0..=2))) => Ok(()),
        _ => Err(format!(
            "{}.axis must be a uint8 value from 0 through 2",
            object.class_name()
        )),
    }
}

fn validate_translation_value_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    validate_dof_value_object(object, TRANSLATION_DOF_CLASS, DISPLACEMENT_CLASS)
}

fn validate_rotation_value_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    validate_dof_value_object(object, ROTATION_DOF_CLASS, ROTATION_CLASS)
}

fn validate_dof_value_object(
    object: &ObjectInstance,
    dof_class: &str,
    value_class: &str,
) -> Result<(), String> {
    let nested_class = |name: &str| match object.properties().get(name) {
        Some(Value::Object(value)) => Some(value.class_name().display_name()),
        _ => None,
    };
    if nested_class("dof") != Some(dof_class) || nested_class("value") != Some(value_class) {
        return Err(format!(
            "{} requires a {dof_class} and {value_class}",
            object.class_name()
        ));
    }
    Ok(())
}
