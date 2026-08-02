use serde::Serialize;

use crate::error::TestDomainError;
use crate::version::PROTOCOL_VERSION;

use super::{ProtocolHandshake, ProtocolLimits, WorkerRequest, WorkerResponse};

pub fn negotiate(
    local: &ProtocolHandshake,
    remote: &ProtocolHandshake,
) -> Result<ProtocolLimits, TestDomainError> {
    for version in [local.protocol_version, remote.protocol_version] {
        if version != PROTOCOL_VERSION {
            return Err(TestDomainError::IncompatibleProtocol {
                actual: version,
                supported: PROTOCOL_VERSION,
            });
        }
    }
    Ok(ProtocolLimits {
        max_message_bytes: local
            .limits
            .max_message_bytes
            .min(remote.limits.max_message_bytes),
        max_tests_per_plan: local
            .limits
            .max_tests_per_plan
            .min(remote.limits.max_tests_per_plan),
        max_commands_per_invocation: local
            .limits
            .max_commands_per_invocation
            .min(remote.limits.max_commands_per_invocation),
        max_output_bytes_per_attempt: local
            .limits
            .max_output_bytes_per_attempt
            .min(remote.limits.max_output_bytes_per_attempt),
        max_diagnostics_per_attempt: local
            .limits
            .max_diagnostics_per_attempt
            .min(remote.limits.max_diagnostics_per_attempt),
        max_artifacts_per_attempt: local
            .limits
            .max_artifacts_per_attempt
            .min(remote.limits.max_artifacts_per_attempt),
        max_coverage_sites_per_attempt: local
            .limits
            .max_coverage_sites_per_attempt
            .min(remote.limits.max_coverage_sites_per_attempt),
    })
}

pub fn encode_request(
    request: &WorkerRequest,
    limits: ProtocolLimits,
) -> Result<Vec<u8>, TestDomainError> {
    validate_request(request, limits)?;
    encode(request, limits)
}

pub fn encode_response(
    response: &WorkerResponse,
    limits: ProtocolLimits,
) -> Result<Vec<u8>, TestDomainError> {
    validate_response(response, limits)?;
    encode(response, limits)
}

pub fn decode_request(
    bytes: &[u8],
    limits: ProtocolLimits,
) -> Result<WorkerRequest, TestDomainError> {
    check_payload(bytes.len(), limits)?;
    let request = serde_json::from_slice(bytes).map_err(|error| TestDomainError::InvalidField {
        field: "worker_request",
        reason: error.to_string(),
    })?;
    validate_request(&request, limits)?;
    Ok(request)
}

pub fn decode_response(
    bytes: &[u8],
    limits: ProtocolLimits,
) -> Result<WorkerResponse, TestDomainError> {
    check_payload(bytes.len(), limits)?;
    let response =
        serde_json::from_slice(bytes).map_err(|error| TestDomainError::InvalidField {
            field: "worker_response",
            reason: error.to_string(),
        })?;
    validate_response(&response, limits)?;
    Ok(response)
}

fn encode<T: Serialize>(value: &T, limits: ProtocolLimits) -> Result<Vec<u8>, TestDomainError> {
    let encoded = serde_json::to_vec(value).map_err(|error| TestDomainError::InvalidField {
        field: "worker_message",
        reason: error.to_string(),
    })?;
    check_payload(encoded.len(), limits)?;
    Ok(encoded)
}

fn check_payload(actual: usize, limits: ProtocolLimits) -> Result<(), TestDomainError> {
    if actual > limits.max_message_bytes as usize {
        return Err(TestDomainError::ProtocolPayloadTooLarge {
            actual,
            limit: limits.max_message_bytes as usize,
        });
    }
    Ok(())
}

fn check_attempt(
    result: &crate::result::AttemptResult,
    limits: ProtocolLimits,
) -> Result<(), TestDomainError> {
    for (field, actual, limit) in [
        (
            "attempt.diagnostics",
            result.diagnostics.len(),
            limits.max_diagnostics_per_attempt as usize,
        ),
        (
            "attempt.artifacts",
            result.artifacts.len(),
            limits.max_artifacts_per_attempt as usize,
        ),
    ] {
        if actual > limit {
            return Err(TestDomainError::ProtocolCollectionTooLarge {
                field,
                actual,
                limit,
            });
        }
    }
    if result.output.len() > limits.max_output_bytes_per_attempt as usize {
        return Err(TestDomainError::ProtocolPayloadTooLarge {
            actual: result.output.len(),
            limit: limits.max_output_bytes_per_attempt as usize,
        });
    }
    Ok(())
}

fn validate_request(
    request: &WorkerRequest,
    limits: ProtocolLimits,
) -> Result<(), TestDomainError> {
    if let WorkerRequest::InstallPlan { plan, snapshot } = request {
        let count = plan.tests().count();
        if count > limits.max_tests_per_plan as usize {
            return Err(TestDomainError::ProtocolCollectionTooLarge {
                field: "plan.tests",
                actual: count,
                limit: limits.max_tests_per_plan as usize,
            });
        }
        snapshot.validate()?;
        if snapshot.program_revision != plan.program_revision {
            return Err(TestDomainError::InvalidField {
                field: "plan.program_revision",
                reason: "installed plan and frozen source snapshot revisions differ".into(),
            });
        }
    }
    Ok(())
}

fn validate_response(
    response: &WorkerResponse,
    limits: ProtocolLimits,
) -> Result<(), TestDomainError> {
    match response {
        WorkerResponse::Completed { result, coverage } => {
            check_attempt(result, limits)?;
            let actual = coverage
                .iter()
                .map(|fragment| fragment.sites.len())
                .sum::<usize>();
            let limit = limits.max_coverage_sites_per_attempt as usize;
            if actual > limit {
                return Err(TestDomainError::ProtocolCollectionTooLarge {
                    field: "attempt.coverage.sites",
                    actual,
                    limit,
                });
            }
            Ok(())
        }
        WorkerResponse::Event { event } => match &event.payload {
            crate::event::TestEventPayload::TestFinished { result } => {
                check_attempt(result, limits)
            }
            _ => Ok(()),
        },
        _ => Ok(()),
    }
}
