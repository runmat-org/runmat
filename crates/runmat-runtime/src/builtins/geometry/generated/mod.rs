mod args;
mod codec;
pub(crate) mod combine;
pub(crate) mod features;
pub(crate) mod primitives;
pub(crate) mod regions;

use std::collections::HashMap;
use std::sync::Once;

use runmat_types::{MemberAccess, MemberName};

use crate::class_registry::declaration::{immutable_property, register_value_class};
use crate::class_registry::RuntimeProperty;
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};

// Class identities are owned by the catalog; these names are derived from it.
pub const GEOMETRY_FEATURE_CLASS: &str = runmat_builtins::GEOMETRY_FEATURE_CLASS.display_name();
pub const GENERATED_GEOMETRY_CLASS: &str = runmat_builtins::GEOMETRY_GENERATED_CLASS.display_name();
pub const GEOMETRY_REGION_OUTPUT_CLASS: &str =
    runmat_builtins::GEOMETRY_REGION_OUTPUT_CLASS.display_name();
const FEATURE_TYPE_IDENTITY: &str = "runmat.geometry.feature";
pub(crate) const GENERATED_TYPE_IDENTITY: &str = "runmat.geometry.generated-product";
const REGION_OUTPUT_TYPE_IDENTITY: &str = "runmat.geometry.region-output";
pub(crate) const CONTRACT_PROPERTY: &str = "contract";
const CONTRACT_FIELDS: &[&str] = &[CONTRACT_PROPERTY];

pub(crate) fn ensure_classes_registered() {
    static PORTABLE: Once = Once::new();
    PORTABLE.call_once(|| {
        for (class_name, type_identity, validator) in [
            (
                GEOMETRY_FEATURE_CLASS,
                FEATURE_TYPE_IDENTITY,
                codec::validate_feature_object
                    as crate::execution::value_codec::ImmutableValueClassValidator,
            ),
            (
                GENERATED_GEOMETRY_CLASS,
                GENERATED_TYPE_IDENTITY,
                codec::validate_generated_object,
            ),
            (
                GEOMETRY_REGION_OUTPUT_CLASS,
                REGION_OUTPUT_TYPE_IDENTITY,
                codec::validate_region_output_object,
            ),
        ] {
            register_immutable_value_class(ImmutableValueClassSchema {
                class_name,
                type_identity,
                schema_version: 1,
                allowed_fields: CONTRACT_FIELDS,
                required_fields: CONTRACT_FIELDS,
                validate: validator,
            })
            .unwrap_or_else(|error| panic!("failed to register {class_name}: {error}"));
        }
    });
    static FEATURE: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_builtins::GEOMETRY_FEATURE_CLASS);
    static GENERATED: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_builtins::GEOMETRY_GENERATED_CLASS);
    static REGION: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(
            runmat_builtins::GEOMETRY_REGION_OUTPUT_CLASS,
        );
    FEATURE.ensure(|| {
        register_value_class(
            GEOMETRY_FEATURE_CLASS,
            contract_properties(),
            HashMap::new(),
        )
    });
    GENERATED.ensure(|| {
        register_value_class(
            GENERATED_GEOMETRY_CLASS,
            contract_properties(),
            HashMap::new(),
        )
    });
    REGION.ensure(|| {
        register_value_class(
            GEOMETRY_REGION_OUTPUT_CLASS,
            contract_properties(),
            HashMap::new(),
        )
    });
}

/// Contract payload fields are internal to the class; they are never read as
/// public properties.
fn contract_properties() -> HashMap<MemberName, RuntimeProperty> {
    CONTRACT_FIELDS
        .iter()
        .map(|field| immutable_property(field, MemberAccess::Private))
        .collect()
}

#[cfg(test)]
mod tests;

pub(crate) fn invalid(builtin: &'static str, message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(format!("{builtin}: {}", message.into()))
        .with_builtin(builtin)
        .with_identifier(format!(
            "RunMat:geometry:{}:InvalidInput",
            builtin.trim_start_matches("geometry.")
        ))
        .build()
}
