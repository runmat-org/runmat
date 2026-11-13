use std::sync::Once;

use super::*;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::{
    quantity_scalar, RAYLEIGH_MASS_COEFFICIENT_CLASS, RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS,
};
use crate::class_registry::declaration::{immutable_properties, static_methods};
use crate::execution::value_codec::ImmutableValueClassSchema;

pub(super) const MECHANICAL_DAMPING_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_MECHANICAL_DAMPING_CLASS
        .display_name();
const KIND: &str = "kind";
const MASS_COEFFICIENT: &str = "mass_coefficient";
const STIFFNESS_COEFFICIENT: &str = "stiffness_coefficient";
const FIELDS: &[&str] = &[KIND, MASS_COEFFICIENT, STIFFNESS_COEFFICIENT];
const REQUIRED_FIELDS: &[&str] = &[KIND];
#[runtime_builtin(
    name = "fea.MechanicalDamping.undamped",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::damping"
)]
pub async fn mechanical_damping_undamped_builtin() -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_MECHANICAL_DAMPING_UNDAMPED_DESCRIPTOR,
        "fea.MechanicalDamping.undamped",
    )?;
    damping_object("undamped", None, None)
}

#[runtime_builtin(
    name = "fea.MechanicalDamping.rayleigh",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::damping"
)]
pub async fn mechanical_damping_rayleigh_builtin(
    mass_coefficient: Value,
    stiffness_coefficient: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_MECHANICAL_DAMPING_RAYLEIGH_DESCRIPTOR,
        "fea.MechanicalDamping.rayleigh",
    )?;
    let mass = quantity_scalar(
        "fea.MechanicalDamping.rayleigh",
        &mass_coefficient,
        RAYLEIGH_MASS_COEFFICIENT_CLASS,
        "per_second",
    )?;
    let stiffness = quantity_scalar(
        "fea.MechanicalDamping.rayleigh",
        &stiffness_coefficient,
        RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS,
        "seconds",
    )?;
    if mass == 0.0 && stiffness == 0.0 {
        return Err(builtin_error(
            "fea.MechanicalDamping.rayleigh",
            &ERROR_INPUT,
            "at least one Rayleigh coefficient must be positive",
        ));
    }
    damping_object(
        "rayleigh",
        Some(mass_coefficient),
        Some(stiffness_coefficient),
    )
}

pub(super) fn mechanical_damping_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<MechanicalDampingModel> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "Damping must be a typed fea.MechanicalDamping value",
        ));
    };
    if !object
        .class_name()
        .is(runmat_builtins::fea_contract::model_types::classes::FEA_MECHANICAL_DAMPING_CLASS)
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "Damping must be {MECHANICAL_DAMPING_CLASS}, got {}",
                object.class_name()
            ),
        ));
    }
    validate_damping_object(object)
        .map_err(|message| builtin_error(builtin, &ERROR_INPUT, message))?;
    let Value::String(runmat_value::StringElement::Text(kind)) = object
        .properties()
        .get(KIND)
        .expect("validated damping kind")
    else {
        unreachable!("validated damping kind is a string")
    };
    match kind.as_str() {
        "undamped" => Ok(MechanicalDampingModel::Undamped),
        "rayleigh" => Ok(MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(quantity_scalar(
                builtin,
                object
                    .properties()
                    .get(MASS_COEFFICIENT)
                    .expect("validated mass coefficient"),
                RAYLEIGH_MASS_COEFFICIENT_CLASS,
                "per_second",
            )?),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(
                quantity_scalar(
                    builtin,
                    object
                        .properties()
                        .get(STIFFNESS_COEFFICIENT)
                        .expect("validated stiffness coefficient"),
                    RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS,
                    "seconds",
                )?,
            ),
        }),
        _ => unreachable!("validated damping kind is supported"),
    }
}

fn damping_object(
    kind: &str,
    mass_coefficient: Option<Value>,
    stiffness_coefficient: Option<Value>,
) -> BuiltinResult<Value> {
    ensure_damping_class_registered();
    let mut object = crate::builtins::fea::object_api::ordinary_builder(
        "fea.MechanicalDamping",
        MECHANICAL_DAMPING_CLASS,
    )?;
    object.insert(KIND.into(), Value::String(kind.into()));
    if let Some(value) = mass_coefficient {
        object.insert(MASS_COEFFICIENT.into(), value);
    }
    if let Some(value) = stiffness_coefficient {
        object.insert(STIFFNESS_COEFFICIENT.into(), value);
    }
    Ok(Value::Object(object.finish()))
}

pub(super) fn ensure_damping_class_registered() {
    static PORTABLE_SCHEMA: Once = Once::new();
    static RUNTIME_CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            MECHANICAL_DAMPING_CLASS,
        ));
    PORTABLE_SCHEMA.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let schemas = vec![ImmutableValueClassSchema {
            class_name: MECHANICAL_DAMPING_CLASS,
            type_identity: "runmat.fea.mechanical-damping",
            schema_version: 1,
            allowed_fields: FIELDS,
            required_fields: REQUIRED_FIELDS,
            validate: validate_damping_object,
        }];

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASS.ensure(|| {
        register_value_class(
            MECHANICAL_DAMPING_CLASS,
            immutable_properties(FIELDS),
            static_methods([
                ("undamped", "fea.MechanicalDamping.undamped"),
                ("rayleigh", "fea.MechanicalDamping.rayleigh"),
            ]),
        );
    });
}

fn validate_damping_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    let Some(Value::String(runmat_value::StringElement::Text(kind))) =
        object.properties().get(KIND)
    else {
        return Err("fea.MechanicalDamping requires a string `kind`".into());
    };
    match kind.as_str() {
        "undamped" => {
            if object.properties().contains_key(MASS_COEFFICIENT)
                || object.properties().contains_key(STIFFNESS_COEFFICIENT)
            {
                return Err("undamped mechanical damping cannot contain coefficients".into());
            }
        }
        "rayleigh" => {
            let mass = object.properties().get(MASS_COEFFICIENT);
            let stiffness = object.properties().get(STIFFNESS_COEFFICIENT);
            if !matches!(mass, Some(Value::Object(value)) if value.class_name().display_name() == RAYLEIGH_MASS_COEFFICIENT_CLASS)
            {
                return Err("Rayleigh damping requires a typed fea.RayleighMassCoefficient".into());
            }
            if !matches!(stiffness, Some(Value::Object(value)) if value.class_name().display_name() == RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS)
            {
                return Err(
                    "Rayleigh damping requires a typed fea.RayleighStiffnessCoefficient".into(),
                );
            }
            let mass = quantity_scalar(
                "fea.MechanicalDamping",
                mass.expect("validated mass coefficient"),
                RAYLEIGH_MASS_COEFFICIENT_CLASS,
                "per_second",
            )
            .map_err(|error| error.message().to_string())?;
            let stiffness = quantity_scalar(
                "fea.MechanicalDamping",
                stiffness.expect("validated stiffness coefficient"),
                RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS,
                "seconds",
            )
            .map_err(|error| error.message().to_string())?;
            if mass == 0.0 && stiffness == 0.0 {
                return Err("at least one Rayleigh coefficient must be positive".into());
            }
        }
        _ => return Err(format!("unsupported mechanical damping kind `{kind}`")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::values::{
        rayleigh_mass_coefficient_builtin, rayleigh_stiffness_coefficient_builtin,
    };
    use crate::execution::value_codec::{decode_inline_value, encode_inline_value};
    use futures::executor::block_on;

    #[test]
    fn typed_damping_laws_are_portable_and_reject_a_zero_rayleigh_pair() {
        let undamped = block_on(mechanical_damping_undamped_builtin()).expect("undamped law");
        assert_eq!(
            decode_inline_value(&encode_inline_value(&undamped).expect("encode undamped"))
                .expect("decode undamped"),
            undamped
        );

        let zero_mass = block_on(rayleigh_mass_coefficient_builtin(Value::Num(0.0)))
            .expect("zero mass coefficient");
        let zero_stiffness = block_on(rayleigh_stiffness_coefficient_builtin(Value::Num(0.0)))
            .expect("zero stiffness coefficient");
        assert!(block_on(mechanical_damping_rayleigh_builtin(
            zero_mass,
            zero_stiffness
        ))
        .is_err());

        let mass =
            block_on(rayleigh_mass_coefficient_builtin(Value::Num(0.1))).expect("mass coefficient");
        let stiffness = block_on(rayleigh_stiffness_coefficient_builtin(Value::Num(0.002)))
            .expect("stiffness coefficient");
        let rayleigh =
            block_on(mechanical_damping_rayleigh_builtin(mass, stiffness)).expect("Rayleigh law");
        assert_eq!(
            decode_inline_value(&encode_inline_value(&rayleigh).expect("encode Rayleigh"))
                .expect("decode Rayleigh"),
            rayleigh
        );
    }
}
