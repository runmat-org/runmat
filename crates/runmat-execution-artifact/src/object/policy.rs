use crate::{ArtifactError, ArtifactResult, LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_execution::ForeignPolicyRef;
use serde::{de::DeserializeOwned, Serialize};
use std::num::NonZeroU64;

/// The policy bytes are the existing Config schema's canonical encoding. This
/// crate deliberately does not interpret that schema or depend on Config.
pub const MAX_FOREIGN_POLICY_BYTES: u64 = 64 * 1024;
const FOREIGN_POLICY_MEDIA_TYPE: &str = "application/vnd.runmat.foreign-policy+json";

pub fn foreign_policy_object(bytes: Vec<u8>) -> ArtifactResult<LogicalObject> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_FOREIGN_POLICY_BYTES {
        return Err(ArtifactError::Limit(
            "foreign policy bytes exceed the bound".into(),
        ));
    }
    let digest = runmat_execution::Digest::sha256(&bytes);
    validate_canonical_policy_bytes(&bytes)?;
    LogicalObject::new(
        ObjectNamespace::ForeignPolicy,
        format!("foreign-policy/{digest}"),
        FOREIGN_POLICY_MEDIA_TYPE,
        bytes,
    )
}

/// Canonicalize a producer-supplied JSON value without interpreting its
/// Config-owned fields. Core can serialize ForeignConfig to a Value first.
pub fn foreign_policy_object_from_json_value(
    value: serde_json::Value,
) -> ArtifactResult<LogicalObject> {
    let bytes = serde_json::to_vec(&sort_json(value))
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    foreign_policy_object(bytes)
}

/// Capture a Config-owned schema without interpreting or duplicating its fields.
pub fn capture_typed_foreign_policy<T: Serialize>(config: &T) -> ArtifactResult<LogicalObject> {
    let value =
        serde_json::to_value(config).map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    foreign_policy_object_from_json_value(value)
}

/// A receiver must prove that its typed schema decodes the entire canonical
/// object losslessly; serde defaults or ignored fields cannot change policy.
pub fn decode_typed_foreign_policy<T: Serialize + DeserializeOwned>(
    object: &LogicalObject,
) -> ArtifactResult<T> {
    validate_foreign_policy_object(object)?;
    let decoded: T = serde_json::from_slice(&object.bytes).map_err(|error| {
        ArtifactError::Invalid(format!("foreign policy schema is invalid: {error}"))
    })?;
    if capture_typed_foreign_policy(&decoded)? != *object {
        return Err(ArtifactError::Invalid(
            "foreign policy has noncanonical typed fields".into(),
        ));
    }
    Ok(decoded)
}

pub fn validate_foreign_policy_object(object: &LogicalObject) -> ArtifactResult<()> {
    validate_foreign_policy_descriptor(&object.descriptor)?;
    object.validate()?;
    validate_canonical_policy_bytes(&object.bytes)
}

fn validate_canonical_policy_bytes(bytes: &[u8]) -> ArtifactResult<()> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_FOREIGN_POLICY_BYTES {
        return Err(ArtifactError::Limit(
            "foreign policy bytes exceed the bound".into(),
        ));
    }
    // This checks only the generic JSON byte form. Config owns the existing
    // ForeignConfig schema and its typed, lossless decode/re-encode check.
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| ArtifactError::Invalid(format!("foreign policy is not JSON: {error}")))?;
    let canonical = serde_json::to_vec(&sort_json(value))
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    if canonical != bytes {
        return Err(ArtifactError::Invalid(
            "foreign policy JSON is not canonical".into(),
        ));
    }
    Ok(())
}

/// Project only a fully admitted Artifact descriptor into the lower-level
/// metadata carried by VM/Runtime. No policy bytes or config fields move.
pub fn foreign_policy_ref(descriptor: &ObjectDescriptor) -> ArtifactResult<ForeignPolicyRef> {
    validate_foreign_policy_descriptor(descriptor)?;
    let length = NonZeroU64::new(descriptor.encoded_length)
        .ok_or_else(|| ArtifactError::Invalid("foreign policy length is zero".into()))?;
    Ok(ForeignPolicyRef::new(descriptor.digest, length))
}

/// Reconstruct the one canonical descriptor from an admitted compact ref.
/// This is the sole metadata-to-Artifact conversion; unbounded or substituted
/// namespace/name/media combinations cannot enter a recipe through it.
pub fn foreign_policy_descriptor(policy: ForeignPolicyRef) -> ArtifactResult<ObjectDescriptor> {
    let descriptor = ObjectDescriptor {
        namespace: ObjectNamespace::ForeignPolicy,
        logical_name: format!("foreign-policy/{}", policy.digest()),
        digest: policy.digest(),
        encoded_length: policy.encoded_length().get(),
        media_type: FOREIGN_POLICY_MEDIA_TYPE.into(),
    };
    validate_foreign_policy_descriptor(&descriptor)?;
    Ok(descriptor)
}

pub(super) fn sort_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries = map.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = serde_json::Map::new();
            for (key, value) in entries {
                sorted.insert(key, sort_json(value));
            }
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(sort_json).collect())
        }
        value => value,
    }
}

pub fn validate_foreign_policy_descriptor(descriptor: &ObjectDescriptor) -> ArtifactResult<()> {
    descriptor.validate()?;
    if descriptor.namespace != ObjectNamespace::ForeignPolicy
        || descriptor.logical_name != format!("foreign-policy/{}", descriptor.digest)
        || descriptor.media_type != FOREIGN_POLICY_MEDIA_TYPE
        || descriptor.encoded_length == 0
        || descriptor.encoded_length > MAX_FOREIGN_POLICY_BYTES
    {
        return Err(ArtifactError::Invalid(
            "foreign policy descriptor is not canonical".into(),
        ));
    }
    Ok(())
}
