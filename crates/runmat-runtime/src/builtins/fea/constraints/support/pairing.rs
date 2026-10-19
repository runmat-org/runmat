use std::collections::HashMap;
use std::sync::Once;

use super::super::*;
use crate::builtins::fea::values::{quantity_scalar, LENGTH_CLASS};
use crate::class_registry::declaration::{
    immutable_properties, register_value_class, static_methods,
};
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};
use runmat_analysis_core::Length;

const PAIRING_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_PERIODIC_PAIRING_CLASS.display_name();
const MATCHING_PAIRING_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_MATCHING_PERIODIC_PAIRING_CLASS
        .display_name();
const GEOMETRIC_PAIRING_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_GEOMETRIC_PERIODIC_PAIRING_CLASS
        .display_name();
const EMPTY_FIELDS: &[&str] = &[];
const GEOMETRIC_FIELDS: &[&str] = &["tolerance"];
#[runtime_builtin(
    name = "fea.PeriodicPairing.matching",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::constraints::support::pairing"
)]
pub async fn periodic_pairing_matching_builtin() -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PERIODIC_PAIRING_MATCHING_DESCRIPTOR,
        "fea.PeriodicPairing.matching",
    )?;
    ensure_pairing_classes_registered();
    Ok(Value::Object(ObjectInstance::new(MATCHING_PAIRING_CLASS)))
}

#[runtime_builtin(
    name = "fea.PeriodicPairing.geometric",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::constraints::support::pairing"
)]
pub async fn periodic_pairing_geometric_builtin(tolerance: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PERIODIC_PAIRING_GEOMETRIC_DESCRIPTOR,
        "fea.PeriodicPairing.geometric",
    )?;
    let meters = quantity_scalar(
        "fea.PeriodicPairing.geometric",
        &tolerance,
        LENGTH_CLASS,
        "meters",
    )?;
    if meters < 0.0 {
        return Err(builtin_error(
            "fea.PeriodicPairing.geometric",
            &ERROR_INPUT,
            "pairing tolerance cannot be negative",
        ));
    }
    ensure_pairing_classes_registered();
    let mut object = ObjectInstance::new(GEOMETRIC_PAIRING_CLASS);
    object.properties.insert("tolerance".into(), tolerance);
    Ok(Value::Object(object))
}

pub(crate) fn pairing_policy_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<PairingPolicy> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "expected a typed fea.PeriodicPairing value",
        ));
    };
    match object.class_name.display_name() {
        MATCHING_PAIRING_CLASS => Ok(PairingPolicy::Matching),
        GEOMETRIC_PAIRING_CLASS => {
            let tolerance = object.properties.get("tolerance").ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "geometric pairing tolerance is required",
                )
            })?;
            Ok(PairingPolicy::GeometricTolerance(Length::new(
                quantity_scalar(builtin, tolerance, LENGTH_CLASS, "meters")?,
            )))
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("unsupported periodic-pairing class `{other}`"),
        )),
    }
}

pub(super) fn ensure_pairing_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            PAIRING_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        for schema in [
            ImmutableValueClassSchema {
                class_name: MATCHING_PAIRING_CLASS,
                type_identity: "runmat.fea.periodic-pairing.matching",
                schema_version: 1,
                allowed_fields: EMPTY_FIELDS,
                required_fields: EMPTY_FIELDS,
                validate: validate_matching,
            },
            ImmutableValueClassSchema {
                class_name: GEOMETRIC_PAIRING_CLASS,
                type_identity: "runmat.fea.periodic-pairing.geometric",
                schema_version: 1,
                allowed_fields: GEOMETRIC_FIELDS,
                required_fields: GEOMETRIC_FIELDS,
                validate: validate_geometric,
            },
        ] {
            register_immutable_value_class(schema)
                .unwrap_or_else(|error| panic!("failed to register periodic pairing: {error}"));
        }
    });
    RUNTIME_CLASSES.ensure(|| {
        register_value_class(
            MATCHING_PAIRING_CLASS,
            immutable_properties(EMPTY_FIELDS),
            HashMap::new(),
        );
        register_value_class(
            GEOMETRIC_PAIRING_CLASS,
            immutable_properties(GEOMETRIC_FIELDS),
            HashMap::new(),
        );
        register_value_class(
            PAIRING_CLASS,
            immutable_properties(EMPTY_FIELDS),
            static_methods([
                ("matching", "fea.PeriodicPairing.matching"),
                ("geometric", "fea.PeriodicPairing.geometric"),
            ]),
        );
    });
}

fn validate_matching(_: &ObjectInstance) -> Result<(), String> {
    Ok(())
}

fn validate_geometric(object: &ObjectInstance) -> Result<(), String> {
    if !matches!(
        object.properties.get("tolerance"),
        Some(Value::Object(value)) if value.class_name.display_name() == LENGTH_CLASS
    ) {
        return Err("geometric pairing requires a typed fea.Length tolerance".into());
    }
    Ok(())
}
