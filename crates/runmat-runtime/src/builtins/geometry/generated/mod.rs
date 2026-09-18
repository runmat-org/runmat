mod codec;
mod combine;
mod metadata;
mod primitives;
mod regions;

use std::sync::Once;

use runmat_builtins::{
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, ResolveContext, Type,
};
use runmat_types::MemberAccess;

use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};

pub const GEOMETRY_FEATURE_CLASS: &str = "geometry.Feature";
pub const GENERATED_GEOMETRY_CLASS: &str = "geometry.Generated";
pub const GEOMETRY_REGION_OUTPUT_CLASS: &str = "geometry.RegionOutput";
const FEATURE_TYPE_IDENTITY: &str = "runmat.geometry.feature";
pub(crate) const GENERATED_TYPE_IDENTITY: &str = "runmat.geometry.generated-product";
const REGION_OUTPUT_TYPE_IDENTITY: &str = "runmat.geometry.region-output";
pub(crate) const CONTRACT_PROPERTY: &str = "contract";
const CONTRACT_FIELDS: &[&str] = &[CONTRACT_PROPERTY];

pub(crate) const GENERATED_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Generated geometry stores SI coordinates and transforms in canonical binary64 form; exactly representable integer literals may cross that boundary, but no operation performs integer-domain computation or returns integer data.",
    };

pub(crate) fn feature_type(_args: &[Type], _context: &ResolveContext) -> Type {
    object_type(GEOMETRY_FEATURE_CLASS)
}

pub(crate) fn generated_type(_args: &[Type], _context: &ResolveContext) -> Type {
    object_type(GENERATED_GEOMETRY_CLASS)
}

pub(crate) fn region_output_type(_args: &[Type], _context: &ResolveContext) -> Type {
    object_type(GEOMETRY_REGION_OUTPUT_CLASS)
}

fn object_type(class_name: &str) -> Type {
    Type::Object {
        class_name: Some(class_name.into()),
        shape: Some(vec![Some(1), Some(1)]),
    }
}

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
        crate::class_registry::ClassRegistration::new(GEOMETRY_FEATURE_CLASS);
    static GENERATED: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(GENERATED_GEOMETRY_CLASS);
    static REGION: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(GEOMETRY_REGION_OUTPUT_CLASS);
    FEATURE.ensure(|| register_runtime_class(GEOMETRY_FEATURE_CLASS));
    GENERATED.ensure(|| register_runtime_class(GENERATED_GEOMETRY_CLASS));
    REGION.ensure(|| register_runtime_class(GEOMETRY_REGION_OUTPUT_CLASS));
}

fn register_runtime_class(class_name: &'static str) {
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: class_name.into(),
        properties: CONTRACT_FIELDS
            .iter()
            .map(|name| {
                (
                    (*name).into(),
                    crate::class_registry::RuntimeProperty {
                        name: (*name).into(),
                        is_static: false,
                        is_constant: true,
                        is_dependent: false,
                        get_access: MemberAccess::Private,
                        set_access: MemberAccess::Private,
                        default_value: None,
                    },
                )
            })
            .collect(),
        methods: Default::default(),
        parent: None,
    });
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
