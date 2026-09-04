use std::collections::HashMap;
use std::sync::Once;

use super::super::*;
use super::{register_runtime_class, static_method};
use crate::builtins::fea::values::finite_numeric_vector3;
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};

const DIRECTION_CLASS: &str = "fea.Direction";
const DIRECTION_FIELDS: &[&str] = &["components"];
const DIRECTION_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "components",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite, nonzero 1-by-3 or 3-by-1 direction vector.",
}];
const DIRECTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "direction = fea.Direction.vector(components)",
    inputs: &DIRECTION_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const DIRECTION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIRECTION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const DIRECTION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "components",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The exact three-component integer vector is validated before explicit conversion to canonical normalized binary64 components.",
    }];
pub(super) const DIRECTION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "three-component direction",
        inputs: &DIRECTION_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Direction normalization uses binary64 arithmetic; resident values are rejected without provider execution.",
    }];

#[runtime_builtin(
    name = "fea.Direction.vector",
    category = "fea",
    summary = "Create a normalized typed direction from three components.",
    keywords = "fea,direction,axis,frame,typed",
    descriptor(crate::builtins::fea::constraints::support::direction::DIRECTION_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::fea::constraints::support::direction::DIRECTION_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::fea::constraints::support::direction"
)]
pub async fn direction_vector_builtin(components: Value) -> BuiltinResult<Value> {
    let components = finite_numeric_vector3("fea.Direction.vector", &components, "components")?;
    let norm = components
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON {
        return Err(builtin_error(
            "fea.Direction.vector",
            &ERROR_INPUT,
            "direction components must form a nonzero vector",
        ));
    }
    direction_to_value(components.map(|value| value / norm))
}

pub(crate) fn direction_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<[f64; 3]> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {DIRECTION_CLASS}"),
        ));
    };
    if object.class_name != DIRECTION_CLASS {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {DIRECTION_CLASS}, got {}", object.class_name),
        ));
    }
    let components = object.properties.get("components").ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{DIRECTION_CLASS} is missing `components`"),
        )
    })?;
    finite_numeric_vector3(builtin, components, "direction")
}

fn direction_to_value(components: [f64; 3]) -> BuiltinResult<Value> {
    ensure_direction_class_registered();
    let tensor = Tensor::new(components.to_vec(), vec![1, 3]).map_err(|error| {
        builtin_error(
            "fea.Direction.vector",
            &ERROR_INTERNAL,
            format!("failed to store direction: {error}"),
        )
    })?;
    let mut object = ObjectInstance::new(DIRECTION_CLASS.into());
    object
        .properties
        .insert("components".into(), Value::Tensor(tensor));
    Ok(Value::Object(object))
}

pub(super) fn ensure_direction_class_registered() {
    static PORTABLE_SCHEMA: Once = Once::new();
    static RUNTIME_CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(DIRECTION_CLASS);
    PORTABLE_SCHEMA.call_once(|| {
        register_immutable_value_class(ImmutableValueClassSchema {
            class_name: DIRECTION_CLASS,
            type_identity: "runmat.fea.direction",
            schema_version: 1,
            allowed_fields: DIRECTION_FIELDS,
            required_fields: DIRECTION_FIELDS,
            validate: validate_direction_object,
        })
        .unwrap_or_else(|error| panic!("failed to register typed direction class: {error}"));
    });
    RUNTIME_CLASS.ensure(|| {
        register_runtime_class(
            DIRECTION_CLASS,
            DIRECTION_FIELDS,
            HashMap::from([(
                "vector".into(),
                static_method("vector", "fea.Direction.vector"),
            )]),
        );
    });
}

fn validate_direction_object(object: &ObjectInstance) -> Result<(), String> {
    let Some(value) = object.properties.get("components") else {
        return Err("fea.Direction.components is required".into());
    };
    let components = finite_numeric_vector3("fea.Direction", value, "components")
        .map_err(|error| error.to_string())?;
    let norm = components
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if (norm - 1.0).abs() > 1.0e-12 {
        return Err("fea.Direction.components must be normalized".into());
    }
    Ok(())
}
