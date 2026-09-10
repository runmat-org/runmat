use std::collections::HashMap;
use std::sync::Once;

use super::*;
use crate::builtins::fea::values::{
    quantity_scalar, RAYLEIGH_MASS_COEFFICIENT_CLASS, RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS,
};
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};

pub(super) const MECHANICAL_DAMPING_CLASS: &str = "fea.MechanicalDamping";
const KIND: &str = "kind";
const MASS_COEFFICIENT: &str = "mass_coefficient";
const STIFFNESS_COEFFICIENT: &str = "stiffness_coefficient";
const FIELDS: &[&str] = &[KIND, MASS_COEFFICIENT, STIFFNESS_COEFFICIENT];
const REQUIRED_FIELDS: &[&str] = &[KIND];
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const RAYLEIGH_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "massCoefficient",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.RayleighMassCoefficient multiplying the mass matrix.",
    },
    BuiltinParamDescriptor {
        name: "stiffnessCoefficient",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.RayleighStiffnessCoefficient multiplying the stiffness matrix.",
    },
];
const UNDAMPED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "damping = fea.MechanicalDamping.undamped()",
    inputs: &NO_INPUTS,
    outputs: &OUT_ANY,
}];
const RAYLEIGH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "damping = fea.MechanicalDamping.rayleigh(massCoefficient, stiffnessCoefficient)",
    inputs: &RAYLEIGH_INPUTS,
    outputs: &OUT_ANY,
}];

pub(super) const UNDAMPED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNDAMPED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub(super) const RAYLEIGH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RAYLEIGH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.MechanicalDamping.undamped",
    category = "fea",
    summary = "Create an explicitly undamped mechanical material law.",
    keywords = "fea,material,damping,undamped,typed",
    descriptor(crate::builtins::fea::damping::UNDAMPED_DESCRIPTOR),
    integer_audit(FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::damping"
)]
pub async fn mechanical_damping_undamped_builtin() -> BuiltinResult<Value> {
    damping_object("undamped", None, None)
}

#[runtime_builtin(
    name = "fea.MechanicalDamping.rayleigh",
    category = "fea",
    summary = "Create a Rayleigh damping law from typed mass and stiffness coefficients.",
    keywords = "fea,material,damping,rayleigh,mass,stiffness,typed",
    descriptor(crate::builtins::fea::damping::RAYLEIGH_DESCRIPTOR),
    integer_audit(FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::damping"
)]
pub async fn mechanical_damping_rayleigh_builtin(
    mass_coefficient: Value,
    stiffness_coefficient: Value,
) -> BuiltinResult<Value> {
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
    if object.class_name != MECHANICAL_DAMPING_CLASS {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "Damping must be {MECHANICAL_DAMPING_CLASS}, got {}",
                object.class_name
            ),
        ));
    }
    validate_damping_object(object)
        .map_err(|message| builtin_error(builtin, &ERROR_INPUT, message))?;
    let Value::String(kind) = object.properties.get(KIND).expect("validated damping kind") else {
        unreachable!("validated damping kind is a string")
    };
    match kind.as_str() {
        "undamped" => Ok(MechanicalDampingModel::Undamped),
        "rayleigh" => Ok(MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(quantity_scalar(
                builtin,
                object
                    .properties
                    .get(MASS_COEFFICIENT)
                    .expect("validated mass coefficient"),
                RAYLEIGH_MASS_COEFFICIENT_CLASS,
                "per_second",
            )?),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(
                quantity_scalar(
                    builtin,
                    object
                        .properties
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
    let mut object = ObjectInstance::new(MECHANICAL_DAMPING_CLASS.into());
    object
        .properties
        .insert(KIND.into(), Value::String(kind.into()));
    if let Some(value) = mass_coefficient {
        object.properties.insert(MASS_COEFFICIENT.into(), value);
    }
    if let Some(value) = stiffness_coefficient {
        object
            .properties
            .insert(STIFFNESS_COEFFICIENT.into(), value);
    }
    Ok(Value::Object(object))
}

pub(super) fn ensure_damping_class_registered() {
    static PORTABLE_SCHEMA: Once = Once::new();
    static RUNTIME_CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(MECHANICAL_DAMPING_CLASS);
    PORTABLE_SCHEMA.call_once(|| {
        register_immutable_value_class(ImmutableValueClassSchema {
            class_name: MECHANICAL_DAMPING_CLASS,
            type_identity: "runmat.fea.mechanical-damping",
            schema_version: 1,
            allowed_fields: FIELDS,
            required_fields: REQUIRED_FIELDS,
            validate: validate_damping_object,
        })
        .unwrap_or_else(|error| panic!("failed to register mechanical damping class: {error}"));
    });
    RUNTIME_CLASS.ensure(|| {
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: MECHANICAL_DAMPING_CLASS.into(),
            parent: None,
            properties: FIELDS
                .iter()
                .map(|field| {
                    (
                        (*field).into(),
                        crate::class_registry::RuntimeProperty {
                            name: (*field).into(),
                            is_static: false,
                            is_constant: true,
                            is_dependent: false,
                            get_access: MemberAccess::Public,
                            set_access: MemberAccess::Private,
                            default_value: None,
                        },
                    )
                })
                .collect(),
            methods: [
                ("undamped", "fea.MechanicalDamping.undamped"),
                ("rayleigh", "fea.MechanicalDamping.rayleigh"),
            ]
            .into_iter()
            .map(|(name, function_name)| {
                (
                    name.into(),
                    crate::class_registry::RuntimeMethod {
                        name: name.into(),
                        is_static: true,
                        is_abstract: false,
                        is_sealed: false,
                        access: MemberAccess::Public,
                        function_name: function_name.into(),
                        implicit_class_argument: None,
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
        });
    });
}

fn validate_damping_object(object: &ObjectInstance) -> Result<(), String> {
    let Some(Value::String(kind)) = object.properties.get(KIND) else {
        return Err("fea.MechanicalDamping requires a string `kind`".into());
    };
    match kind.as_str() {
        "undamped" => {
            if object.properties.contains_key(MASS_COEFFICIENT)
                || object.properties.contains_key(STIFFNESS_COEFFICIENT)
            {
                return Err("undamped mechanical damping cannot contain coefficients".into());
            }
        }
        "rayleigh" => {
            let mass = object.properties.get(MASS_COEFFICIENT);
            let stiffness = object.properties.get(STIFFNESS_COEFFICIENT);
            if !matches!(mass, Some(Value::Object(value)) if value.class_name == RAYLEIGH_MASS_COEFFICIENT_CLASS)
            {
                return Err("Rayleigh damping requires a typed fea.RayleighMassCoefficient".into());
            }
            if !matches!(stiffness, Some(Value::Object(value)) if value.class_name == RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS)
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
