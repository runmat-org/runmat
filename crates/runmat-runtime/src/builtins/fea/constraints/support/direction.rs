use std::sync::Once;

use super::super::*;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::finite_numeric_vector3;
use crate::class_registry::declaration::{immutable_properties, static_methods};
use crate::execution::value_codec::ImmutableValueClassSchema;

const DIRECTION_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_DIRECTION_CLASS.display_name();
const DIRECTION_FIELDS: &[&str] = &["components"];
#[runtime_builtin(
    name = "fea.Direction.vector",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::fea::constraints::support::direction"
)]
pub async fn direction_vector_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    components: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_DIRECTION_VECTOR_DESCRIPTOR,
        "fea.Direction.vector",
    )?;
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
    direction_to_value(runtime.as_ref(), components.map(|value| value / norm))
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
    if !object
        .class_name()
        .is(runmat_builtins::fea_contract::model_types::classes::FEA_DIRECTION_CLASS)
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {DIRECTION_CLASS}, got {}", object.class_name()),
        ));
    }
    let components = object.properties().get("components").ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{DIRECTION_CLASS} is missing `components`"),
        )
    })?;
    finite_numeric_vector3(builtin, components, "direction")
}

fn direction_to_value(
    runtime: Option<&crate::context::RuntimeContext>,
    components: [f64; 3],
) -> BuiltinResult<Value> {
    ensure_direction_class_registered();
    let tensor = Tensor::new(components.to_vec(), vec![1, 3]).map_err(|error| {
        builtin_error(
            "fea.Direction.vector",
            &ERROR_INTERNAL,
            format!("failed to store direction: {error}"),
        )
    })?;
    let fields = [("components", Value::Tensor(tensor))];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        DIRECTION_CLASS,
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

pub(super) fn ensure_direction_class_registered() {
    static PORTABLE_SCHEMA: Once = Once::new();
    static RUNTIME_CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            DIRECTION_CLASS,
        ));
    PORTABLE_SCHEMA.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let schemas = vec![ImmutableValueClassSchema {
            class_name: DIRECTION_CLASS,
            type_identity: "runmat.fea.direction",
            schema_version: 1,
            allowed_fields: DIRECTION_FIELDS,
            required_fields: DIRECTION_FIELDS,
            validate: validate_direction_object,
        }];

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASS.ensure(|| {
        register_value_class(
            DIRECTION_CLASS,
            immutable_properties(DIRECTION_FIELDS),
            static_methods([("vector", "fea.Direction.vector")]),
        );
    });
}

fn validate_direction_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    let Some(value) = object.properties().get("components") else {
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
