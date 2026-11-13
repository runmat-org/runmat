//! Digest-bound project-only runtime policy. Config owns the complete typed schema.
use crate::{ArtifactError, ArtifactResult, LogicalObject, ObjectDescriptor, ObjectNamespace};
use serde::{de::DeserializeOwned, Serialize};

pub const PROJECT_RUNTIME_POLICY_SCHEMA_VERSION: u16 = 1;
pub const MAX_PROJECT_RUNTIME_POLICY_BYTES: u64 = 64 * 1024;
pub const PROJECT_RUNTIME_POLICY_MEDIA_TYPE: &str =
    "application/vnd.runmat.project-runtime-policy+json";

pub fn capture_typed_project_runtime_policy<T: Serialize>(
    policy: &T,
) -> ArtifactResult<LogicalObject> {
    let value =
        serde_json::to_value(policy).map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    let bytes = serde_json::to_vec(&super::policy::sort_json(value))
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    validate_bytes(&bytes)?;
    let digest = runmat_execution::Digest::sha256(&bytes);
    LogicalObject::new(
        ObjectNamespace::ProjectRuntimePolicy,
        format!("project-runtime-policy/{digest}"),
        PROJECT_RUNTIME_POLICY_MEDIA_TYPE,
        bytes,
    )
}

/// Typed receivers must interpret every field without defaults/ignored fields changing authority.
pub fn decode_typed_project_runtime_policy<T: Serialize + DeserializeOwned>(
    object: &LogicalObject,
) -> ArtifactResult<T> {
    validate_project_runtime_policy_object(object)?;
    let decoded = serde_json::from_slice(&object.bytes).map_err(|error| {
        ArtifactError::Invalid(format!("project runtime policy schema is invalid: {error}"))
    })?;
    if capture_typed_project_runtime_policy(&decoded)? != *object {
        return Err(ArtifactError::Invalid(
            "project runtime policy has noncanonical typed fields".into(),
        ));
    }
    Ok(decoded)
}

pub fn validate_project_runtime_policy_descriptor(
    descriptor: &ObjectDescriptor,
) -> ArtifactResult<()> {
    descriptor.validate()?;
    if descriptor.namespace != ObjectNamespace::ProjectRuntimePolicy
        || descriptor.logical_name != format!("project-runtime-policy/{}", descriptor.digest)
        || descriptor.media_type != PROJECT_RUNTIME_POLICY_MEDIA_TYPE
        || descriptor.encoded_length == 0
        || descriptor.encoded_length > MAX_PROJECT_RUNTIME_POLICY_BYTES
    {
        return Err(ArtifactError::Invalid(
            "project runtime policy descriptor is not canonical".into(),
        ));
    }
    Ok(())
}

pub fn validate_project_runtime_policy_object(object: &LogicalObject) -> ArtifactResult<()> {
    validate_project_runtime_policy_descriptor(&object.descriptor)?;
    object.validate()?;
    validate_bytes(&object.bytes)
}

fn validate_bytes(bytes: &[u8]) -> ArtifactResult<()> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_PROJECT_RUNTIME_POLICY_BYTES {
        return Err(ArtifactError::Limit(
            "project runtime policy bytes exceed the bound".into(),
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        ArtifactError::Invalid(format!("project runtime policy is not JSON: {error}"))
    })?;
    let fields = value
        .as_object()
        .ok_or_else(|| ArtifactError::Invalid("project runtime policy must be an object".into()))?;
    if fields
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        != Some(u64::from(PROJECT_RUNTIME_POLICY_SCHEMA_VERSION))
    {
        return Err(ArtifactError::Invalid(
            "unsupported project runtime policy schema".into(),
        ));
    }
    // Collector policy and credentials are process/admin authority, never project-run authority.
    // Config's deny-unknown-fields lossless decode enforces the remaining complete schema.
    if [
        "logging",
        "telemetry",
        "headers",
        "otel",
        "openTelemetryHeaders",
    ]
    .iter()
    .any(|field| fields.contains_key(*field))
    {
        return Err(ArtifactError::Invalid(
            "project runtime policy must not carry collector, telemetry, or header settings".into(),
        ));
    }
    let canonical = serde_json::to_vec(&super::policy::sort_json(value))
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    if canonical != bytes {
        return Err(ArtifactError::Invalid(
            "project runtime policy JSON is not canonical".into(),
        ));
    }
    Ok(())
}
