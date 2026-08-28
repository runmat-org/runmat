use crate::TestDomainError;

use super::{
    MaterializationRecord, MaterializationRequest, MaterializationResponse, MaterializationStatus,
};

/// Host port for metadata-only evaluation.
///
/// Implementations must execute in a fresh, killable worker with no ambient
/// test workspace, must expose only the expression and immutable source
/// revision carried by the request, and must enforce the request's time and
/// resource policy in addition to the wire-size limits validated here. Test
/// procedures are never submitted through this port.
pub trait IsolatedMetadataMaterializer {
    fn materialize(
        &mut self,
        request: &MaterializationRequest,
    ) -> Result<MaterializationResponse, String>;
}

pub fn materialize_metadata<M: IsolatedMetadataMaterializer>(
    requests: &[MaterializationRequest],
    materializer: &mut M,
) -> (
    Vec<MaterializationResponse>,
    Vec<MaterializationRecord>,
    Vec<TestDomainError>,
) {
    let mut responses = Vec::new();
    let mut records = Vec::with_capacity(requests.len());
    let mut errors = Vec::new();
    for request in requests {
        match materializer.materialize(request) {
            Ok(response) => match validate_response(request, &response) {
                Ok(()) => {
                    records.push(MaterializationRecord {
                        request_id: request.id.clone(),
                        status: MaterializationStatus::Completed,
                        value_count: response.values.len() as u32,
                        diagnostic_count: response.diagnostics.len() as u32,
                    });
                    responses.push(response);
                }
                Err(error) => {
                    records.push(MaterializationRecord {
                        request_id: request.id.clone(),
                        status: MaterializationStatus::Rejected,
                        value_count: 0,
                        diagnostic_count: 0,
                    });
                    errors.push(error);
                }
            },
            Err(message) => {
                records.push(MaterializationRecord {
                    request_id: request.id.clone(),
                    status: MaterializationStatus::Failed,
                    value_count: 0,
                    diagnostic_count: 0,
                });
                errors.push(TestDomainError::InvalidField {
                    field: "materialization",
                    reason: message,
                });
            }
        }
    }
    (responses, records, errors)
}

pub fn validate_response(
    request: &MaterializationRequest,
    response: &MaterializationResponse,
) -> Result<(), TestDomainError> {
    if request.limits.max_duration_ms == 0
        || request.limits.max_memory_bytes == 0
        || request.limits.max_steps == 0
        || request.limits.max_encoded_bytes == 0
    {
        return Err(TestDomainError::InvalidField {
            field: "materialization.limits",
            reason: "execution and payload limits must be non-zero".into(),
        });
    }
    if response.request_id != request.id {
        return Err(TestDomainError::InvalidField {
            field: "materialization.request_id",
            reason: "response does not match request".into(),
        });
    }
    if response.program_revision != request.program_revision {
        return Err(TestDomainError::InvalidField {
            field: "materialization.program_revision",
            reason: "source or graph revision changed during materialization".into(),
        });
    }
    if response.values.len() > request.limits.max_values as usize {
        return Err(TestDomainError::ProtocolCollectionTooLarge {
            field: "materialization.values",
            actual: response.values.len(),
            limit: request.limits.max_values as usize,
        });
    }
    if response.diagnostics.len() > request.limits.max_diagnostics as usize {
        return Err(TestDomainError::ProtocolCollectionTooLarge {
            field: "materialization.diagnostics",
            actual: response.diagnostics.len(),
            limit: request.limits.max_diagnostics as usize,
        });
    }
    let encoded = serde_json::to_vec(response).map_err(|error| TestDomainError::InvalidField {
        field: "materialization.response",
        reason: error.to_string(),
    })?;
    if encoded.len() > request.limits.max_encoded_bytes as usize {
        return Err(TestDomainError::ProtocolPayloadTooLarge {
            actual: encoded.len(),
            limit: request.limits.max_encoded_bytes as usize,
        });
    }
    for value in &response.values {
        if value.name.is_empty() || value.normalized_identity.is_empty() {
            return Err(TestDomainError::InvalidField {
                field: "materialization.value",
                reason: "name and normalized identity must be non-empty".into(),
            });
        }
    }
    Ok(())
}
