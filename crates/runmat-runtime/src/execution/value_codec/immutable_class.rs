use std::collections::{BTreeSet, HashMap};
use std::sync::{OnceLock, RwLock};

use runmat_value::ObjectInstance;

use super::ValueCodecError;

pub type ImmutableValueClassValidator = fn(&ObjectInstance) -> Result<(), String>;

/// Portable schema for an immutable RunMat value class.
///
/// The object's `Value` properties are its sole authority. Registration only
/// describes which properties may cross an execution boundary and validates
/// the fully reconstructed object. It does not attach a second opaque payload.
#[derive(Clone, Copy, Debug)]
pub struct ImmutableValueClassSchema {
    pub class_name: &'static str,
    pub type_identity: &'static str,
    pub schema_version: u32,
    pub allowed_fields: &'static [&'static str],
    pub required_fields: &'static [&'static str],
    pub validate: ImmutableValueClassValidator,
}

#[derive(Clone)]
pub(super) struct RegisteredImmutableValueClass {
    class_name: String,
    type_identity: String,
    schema_version: u32,
    allowed_fields: BTreeSet<String>,
    required_fields: BTreeSet<String>,
    validate: ImmutableValueClassValidator,
}

impl RegisteredImmutableValueClass {
    pub(super) fn class_name(&self) -> &str {
        &self.class_name
    }

    pub(super) fn type_identity(&self) -> &str {
        &self.type_identity
    }

    pub(super) fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub(super) fn validate_object(
        &self,
        object: &ObjectInstance,
        path: &str,
    ) -> Result<(), ValueCodecError> {
        if object.class_name != self.class_name {
            return Err(ValueCodecError::invalid(
                path,
                format!(
                    "registered immutable class `{}` cannot encode object of class `{}`",
                    self.class_name, object.class_name
                ),
            ));
        }
        if object.dynamic_properties.is_some() {
            return Err(ValueCodecError::invalid(
                path,
                "registered immutable value classes cannot have dynamic properties",
            ));
        }
        for field in object.properties.keys() {
            if !self.allowed_fields.contains(field) {
                return Err(ValueCodecError::invalid(
                    path,
                    format!(
                        "immutable class `{}` contains unknown field `{field}`",
                        self.class_name
                    ),
                ));
            }
        }
        for field in &self.required_fields {
            if !object.properties.contains_key(field) {
                return Err(ValueCodecError::invalid(
                    path,
                    format!(
                        "immutable class `{}` is missing required field `{field}`",
                        self.class_name
                    ),
                ));
            }
        }
        (self.validate)(object).map_err(|message| ValueCodecError::invalid(path, message))
    }
}

#[derive(Default)]
struct ImmutableValueClassRegistry {
    by_class_name: HashMap<String, RegisteredImmutableValueClass>,
    class_name_by_type_identity: HashMap<String, String>,
}

fn registry() -> &'static RwLock<ImmutableValueClassRegistry> {
    static REGISTRY: OnceLock<RwLock<ImmutableValueClassRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(ImmutableValueClassRegistry::default()))
}

/// Register one immutable value-class schema for portable execution values.
///
/// Class names and portable type identities are one-to-one. A second
/// registration is rejected, including an attempted schema-version replacement;
/// schema changes must be coordinated explicitly rather than depending on
/// registration order.
pub fn register_immutable_value_class(
    schema: ImmutableValueClassSchema,
) -> Result<(), ValueCodecError> {
    let registered = validate_schema(schema)?;
    let mut registry = registry()
        .write()
        .map_err(|_| ValueCodecError::invalid("$registry", "value-class registry is poisoned"))?;
    if registry.by_class_name.contains_key(&registered.class_name) {
        return Err(ValueCodecError::invalid(
            "$registry",
            format!(
                "immutable value class `{}` is already registered",
                registered.class_name
            ),
        ));
    }
    if registry
        .class_name_by_type_identity
        .contains_key(&registered.type_identity)
    {
        return Err(ValueCodecError::invalid(
            "$registry",
            format!(
                "immutable value-class type identity `{}` is already registered",
                registered.type_identity
            ),
        ));
    }
    registry.class_name_by_type_identity.insert(
        registered.type_identity.clone(),
        registered.class_name.clone(),
    );
    registry
        .by_class_name
        .insert(registered.class_name.clone(), registered);
    Ok(())
}

pub(super) fn schema_for_class(
    class_name: &str,
) -> Result<Option<RegisteredImmutableValueClass>, ValueCodecError> {
    registry()
        .read()
        .map(|registry| registry.by_class_name.get(class_name).cloned())
        .map_err(|_| ValueCodecError::invalid("$registry", "value-class registry is poisoned"))
}

pub(super) fn schema_for_type_identity(
    type_identity: &str,
) -> Result<Option<RegisteredImmutableValueClass>, ValueCodecError> {
    registry()
        .read()
        .map_err(|_| ValueCodecError::invalid("$registry", "value-class registry is poisoned"))
        .map(|registry| {
            registry
                .class_name_by_type_identity
                .get(type_identity)
                .and_then(|class_name| registry.by_class_name.get(class_name))
                .cloned()
        })
}

fn validate_schema(
    schema: ImmutableValueClassSchema,
) -> Result<RegisteredImmutableValueClass, ValueCodecError> {
    if schema.class_name.trim().is_empty() {
        return Err(ValueCodecError::invalid(
            "$registry",
            "immutable value-class name must not be empty",
        ));
    }
    if schema.type_identity.trim().is_empty() {
        return Err(ValueCodecError::invalid(
            "$registry",
            "immutable value-class type identity must not be empty",
        ));
    }
    if schema.schema_version == 0 {
        return Err(ValueCodecError::invalid(
            "$registry",
            "immutable value-class schema version must be non-zero",
        ));
    }
    let allowed_fields = collect_fields(schema.allowed_fields, "allowed")?;
    let required_fields = collect_fields(schema.required_fields, "required")?;
    if let Some(field) = required_fields
        .iter()
        .find(|field| !allowed_fields.contains(*field))
    {
        return Err(ValueCodecError::invalid(
            "$registry",
            format!("required field `{field}` is not present in allowed fields"),
        ));
    }
    Ok(RegisteredImmutableValueClass {
        class_name: schema.class_name.to_owned(),
        type_identity: schema.type_identity.to_owned(),
        schema_version: schema.schema_version,
        allowed_fields,
        required_fields,
        validate: schema.validate,
    })
}

fn collect_fields(fields: &[&str], category: &str) -> Result<BTreeSet<String>, ValueCodecError> {
    let mut collected = BTreeSet::new();
    for field in fields {
        if field.trim().is_empty() {
            return Err(ValueCodecError::invalid(
                "$registry",
                format!("immutable value-class {category} field must not be empty"),
            ));
        }
        if !collected.insert((*field).to_owned()) {
            return Err(ValueCodecError::invalid(
                "$registry",
                format!("duplicate immutable value-class {category} field `{field}`"),
            ));
        }
    }
    Ok(collected)
}
