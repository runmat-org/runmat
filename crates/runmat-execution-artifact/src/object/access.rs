use runmat_execution::identity::ValueId;
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
use runmat_execution::value::{artifact_object_value_id, ValueLimits, ValueRef, ValueRefKind};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{ArtifactError, ArtifactResult, LogicalObject, ObjectDescriptor};

/// Authorization and encryption identity used to project immutable logical objects into portable
/// execution value references. Content identity remains independent of this access context.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactAccess {
    pub authorization_scope: String,
    pub encryption_context: Digest,
}

impl ArtifactAccess {
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.authorization_scope.is_empty()
            || self.authorization_scope.len() > 1024
            || self.authorization_scope.chars().any(char::is_control)
            || self
                .encryption_context
                .bytes()
                .iter()
                .all(|byte| *byte == 0)
        {
            return Err(ArtifactError::Identity(
                "artifact authorization or encryption context is invalid".into(),
            ));
        }
        Ok(())
    }

    pub fn value_id(&self, logical_digest: Digest) -> ValueId {
        artifact_object_value_id(
            logical_digest,
            self.encryption_context,
            &self.authorization_scope,
        )
    }

    pub fn reference(
        &self,
        object: &LogicalObject,
        kind: ValueRefKind,
        value_schema: impl Into<String>,
    ) -> ArtifactResult<ValueRef> {
        object.validate()?;
        self.reference_descriptor(&object.descriptor, kind, value_schema)
    }

    /// Requested string bytes in a newly constructed reference. Inline/boxed
    /// ValueRef slots are the enclosing publication owner's separate inventory.
    pub fn reference_text_bytes(
        &self,
        media_type: &str,
        value_schema: &str,
    ) -> ArtifactResult<u64> {
        media_type
            .len()
            .checked_add(value_schema.len())
            .and_then(|bytes| bytes.checked_add(self.authorization_scope.len()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| ArtifactError::Limit("artifact reference text storage overflow".into()))
    }

    /// Project an immutable descriptor into an access-bound reference. This
    /// constructs a content claim; providers still verify the referenced bytes.
    pub fn reference_descriptor(
        &self,
        descriptor: &ObjectDescriptor,
        kind: ValueRefKind,
        value_schema: impl Into<String>,
    ) -> ArtifactResult<ValueRef> {
        self.validate()?;
        descriptor.validate()?;
        let reference = ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
            id: self.value_id(descriptor.digest),
            logical_digest: descriptor.digest,
            encoded_length: descriptor.encoded_length,
            media_type: descriptor.media_type.clone(),
            value_schema: value_schema.into(),
            encryption_context: self.encryption_context,
            kind,
            authorization_scope: self.authorization_scope.clone(),
            resident_fence: None,
        };
        reference
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        Ok(reference)
    }

    pub fn validate_reference(
        &self,
        reference: &ValueRef,
        kind: ValueRefKind,
        media_type: &str,
        value_schema: &str,
    ) -> ArtifactResult<()> {
        self.validate()?;
        reference
            .validate(ValueLimits::default())
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if reference.id != self.value_id(reference.logical_digest)
            || reference.kind != kind
            || reference.media_type != media_type
            || reference.value_schema != value_schema
            || reference.authorization_scope != self.authorization_scope
            || reference.encryption_context != self.encryption_context
            || reference.resident_fence.is_some()
        {
            return Err(ArtifactError::Identity(
                "value reference is outside its artifact access authority".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use runmat_execution::value::ValueRefKind;

    use super::*;
    use crate::ObjectNamespace;

    #[test]
    fn access_changes_physical_value_identity_but_not_logical_content() {
        let object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "results/value",
            "application/vnd.runmat.test",
            b"content".to_vec(),
        )
        .unwrap();
        let first = ArtifactAccess {
            authorization_scope: "study-a".into(),
            encryption_context: Digest::sha256("key-a"),
        };
        let second = ArtifactAccess {
            authorization_scope: "study-b".into(),
            encryption_context: Digest::sha256("key-b"),
        };
        let first_ref = first
            .reference(&object, ValueRefKind::ResultObject, "runmat.test.v1")
            .unwrap();
        let second_ref = second
            .reference(&object, ValueRefKind::ResultObject, "runmat.test.v1")
            .unwrap();
        for reference in [&first_ref, &second_ref] {
            assert_eq!(reference.schema_version, VALUE_PAYLOAD_SCHEMA_VERSION);
            reference.validate(ValueLimits::default()).unwrap();
        }
        assert_eq!(first_ref.logical_digest, second_ref.logical_digest);
        assert_ne!(first_ref.id, second_ref.id);
        first
            .validate_reference(
                &first_ref,
                ValueRefKind::ResultObject,
                "application/vnd.runmat.test",
                "runmat.test.v1",
            )
            .unwrap();
        assert!(second
            .validate_reference(
                &first_ref,
                ValueRefKind::ResultObject,
                "application/vnd.runmat.test",
                "runmat.test.v1",
            )
            .is_err());
    }
    #[test]
    fn borrowed_reference_validation_matches_payload_limits_without_copying() {
        use runmat_execution::value::ValuePayload;
        let access = ArtifactAccess {
            authorization_scope: "study".into(),
            encryption_context: Digest::sha256("key"),
        };
        let object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "result/value",
            "application/test",
            vec![1],
        )
        .unwrap();
        let reference = access
            .reference(&object, ValueRefKind::ResultObject, "schema.v1")
            .unwrap();
        assert_eq!(
            access
                .reference_text_bytes(&reference.media_type, &reference.value_schema)
                .unwrap(),
            (reference.media_type.capacity()
                + reference.value_schema.capacity()
                + reference.authorization_scope.capacity()) as u64
        );
        for case in 0..5 {
            let mut reference = reference.clone();
            let mut limits = ValueLimits::default();
            match case {
                0 => {}
                1 => limits.max_nodes = 0,
                2 => limits.max_text_bytes = 1,
                3 => reference.schema_version += 1,
                _ => reference.kind = ValueRefKind::ResidentObject,
            }
            assert_eq!(
                reference.validate(limits),
                ValuePayload::Object(Box::new(reference.clone())).validate(limits)
            );
        }
    }
}
