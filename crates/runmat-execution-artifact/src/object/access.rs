use runmat_execution::identity::ValueId;
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{ArtifactError, ArtifactResult, LogicalObject};

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
        ValueId::derive(&[
            b"runmat-artifact-object/v1",
            logical_digest.bytes(),
            self.encryption_context.bytes(),
            self.authorization_scope.as_bytes(),
        ])
    }

    pub fn reference(
        &self,
        object: &LogicalObject,
        kind: ValueRefKind,
        value_schema: impl Into<String>,
    ) -> ArtifactResult<ValueRef> {
        self.validate()?;
        object.validate()?;
        let reference = ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
            id: self.value_id(object.descriptor.digest),
            logical_digest: object.descriptor.digest,
            encoded_length: object.descriptor.encoded_length,
            media_type: object.descriptor.media_type.clone(),
            value_schema: value_schema.into(),
            encryption_context: self.encryption_context,
            kind,
            authorization_scope: self.authorization_scope.clone(),
            resident_fence: None,
        };
        ValuePayload::Object(Box::new(reference.clone()))
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
        ValuePayload::Object(Box::new(reference.clone()))
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
}
