//! Scalar native-object target admission without consulting a runtime registry.

use runmat_types::{ResidencyFact, StaticClassIdentity, ValueFact, ValueKindFact};

/// Reject only proven incompatibility with the allowed canonical handle
/// classes. Unknown class/handle semantics remain for runtime admission; this
/// predicate neither issues a graphics receiver nor licenses alias resolution.
pub(crate) fn known_invalid_scalar_object_handle(
    value: &ValueFact,
    allowed: &[StaticClassIdentity],
) -> bool {
    if value.shape.element_count().is_some_and(|count| count != 1)
        || matches!(
            value.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        )
    {
        return true;
    }
    match &value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Object(object) => {
            object.handle_semantics == Some(false)
                || object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|class| !allowed.iter().any(|allowed| class.is(*allowed)))
        }
        _ => true,
    }
}
