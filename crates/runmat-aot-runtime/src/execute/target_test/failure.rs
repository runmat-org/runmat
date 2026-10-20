//! Typed receiver failure causes and source-chain classification.

use runmat_execution_artifact::target_test::NativeTargetTestFailure;
use runmat_runtime::context::host_allocation::HostAllocationError;
use runmat_runtime::context::host_effect::HostEffectAllocationError;
use runmat_runtime::execution::value_codec::ValueCodecError;
use runmat_runtime::execution::ExecutionServiceError;
use runmat_runtime::RuntimeError;

pub(in crate::execute) fn classify_failure(error: &RuntimeError) -> NativeTargetTestFailure {
    let mut cursor: &(dyn std::error::Error + 'static) = error;
    let mut protocol = false;
    let mut input_decode = false;
    let mut cancelled = false;
    let mut process_failure = false;
    let mut timed_out = false;
    let mut assertion = false;
    let mut out_of_memory = false;
    loop {
        input_decode |= cursor.is::<TargetTestInputDecodeFailure>();
        timed_out |= cursor.is::<TargetTestTimeout>();
        assertion |= cursor.is::<runmat_runtime::runtime_error::RuntimeAssertionFailure>();
        if let Some(service) = cursor.downcast_ref::<ExecutionServiceError>() {
            match service {
                ExecutionServiceError::Cancelled => cancelled = true,
                ExecutionServiceError::ForeignScope
                | ExecutionServiceError::UnknownHandle
                | ExecutionServiceError::WorkerLost(_)
                | ExecutionServiceError::Infrastructure(_) => process_failure = true,
                ExecutionServiceError::InvalidOutputContract if input_decode => protocol = true,
                _ => {}
            }
        }
        if let Some(value) = cursor.downcast_ref::<ValueCodecError>() {
            protocol |= input_decode
                && matches!(
                    value,
                    ValueCodecError::Unsupported { .. }
                        | ValueCodecError::Invalid { .. }
                        | ValueCodecError::TransientSequenceNotPortable { .. }
                );
        }
        if let Some(root) = cursor.downcast_ref::<runmat_gc::GcError>() {
            match root {
                runmat_gc::GcError::OutOfMemory(_) => out_of_memory = true,
                runmat_gc::GcError::InvalidPointer(_)
                | runmat_gc::GcError::CollectionFailed(_)
                | runmat_gc::GcError::RootRegistrationFailed(_)
                | runmat_gc::GcError::ConfigError(_)
                | runmat_gc::GcError::SyncError(_) => process_failure = true,
            }
        }
        if let Some(allocation) = cursor.downcast_ref::<HostAllocationError>() {
            match allocation {
                HostAllocationError::Cancelled
                | HostAllocationError::Resource(HostEffectAllocationError::Cancelled) => {
                    cancelled = true;
                }
                HostAllocationError::MissingManagedOwner
                | HostAllocationError::StaleAuthority
                | HostAllocationError::Resource(
                    HostEffectAllocationError::Unavailable
                    | HostEffectAllocationError::Revoked
                    | HostEffectAllocationError::InvalidInvocation,
                ) => process_failure = true,
                _ => {}
            }
            out_of_memory |= allocation.is_capacity_denial();
        }
        if let Some(allocation) = cursor.downcast_ref::<HostEffectAllocationError>() {
            match allocation {
                HostEffectAllocationError::Cancelled => cancelled = true,
                HostEffectAllocationError::Unavailable
                | HostEffectAllocationError::Revoked
                | HostEffectAllocationError::InvalidInvocation => process_failure = true,
                HostEffectAllocationError::ExceedsCapacity
                | HostEffectAllocationError::ExceedsStageEnvelope => out_of_memory = true,
                _ => {}
            }
        }
        let Some(source) = cursor.source() else { break };
        cursor = source;
    }
    if cancelled {
        NativeTargetTestFailure::Cancelled
    } else if process_failure {
        NativeTargetTestFailure::ProcessFailure
    } else if out_of_memory {
        NativeTargetTestFailure::ConfiguredOutOfMemory
    } else if timed_out {
        NativeTargetTestFailure::Timeout
    } else if assertion {
        NativeTargetTestFailure::Assertion
    } else if protocol {
        NativeTargetTestFailure::ProtocolFailure
    } else {
        NativeTargetTestFailure::LanguageException
    }
}

#[derive(Debug)]
pub(super) struct TargetTestInputDecodeFailure(
    pub runmat_runtime::execution::value_codec::CooperativeValueCodecError,
);
impl std::fmt::Display for TargetTestInputDecodeFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "target-test input decoding failed: {}", self.0)
    }
}
impl std::error::Error for TargetTestInputDecodeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[derive(Debug)]
pub(super) struct TargetTestTimeout;
impl std::fmt::Display for TargetTestTimeout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("target-test wall time exceeded")
    }
}
impl std::error::Error for TargetTestTimeout {}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::execution::value_codec::CooperativeValueCodecError;

    fn decoded_failure(cause: CooperativeValueCodecError) -> NativeTargetTestFailure {
        let error = runmat_runtime::build_runtime_error("test input decoding failed")
            .with_source(TargetTestInputDecodeFailure(cause))
            .build();
        classify_failure(&error)
    }

    #[test]
    fn typed_decode_causes_keep_control_and_input_failure_domains_distinct() {
        assert_eq!(
            decoded_failure(ExecutionServiceError::Cancelled.into()),
            NativeTargetTestFailure::Cancelled
        );
        assert_eq!(
            decoded_failure(ExecutionServiceError::Infrastructure("lost service".into()).into()),
            NativeTargetTestFailure::ProcessFailure
        );
        assert_eq!(
            decoded_failure(
                ValueCodecError::Allocation {
                    path: "$".into(),
                    source: HostAllocationError::Resource(
                        HostEffectAllocationError::ExceedsCapacity
                    ),
                }
                .into()
            ),
            NativeTargetTestFailure::ConfiguredOutOfMemory
        );
        assert_eq!(
            decoded_failure(
                ValueCodecError::Invalid {
                    path: "$".into(),
                    message: "invalid input".into(),
                }
                .into()
            ),
            NativeTargetTestFailure::ProtocolFailure
        );
        for (source, expected) in [
            (
                runmat_gc::GcError::OutOfMemory("root allocation".into()),
                NativeTargetTestFailure::ConfiguredOutOfMemory,
            ),
            (
                runmat_gc::GcError::RootRegistrationFailed("root service unavailable".into()),
                NativeTargetTestFailure::ProcessFailure,
            ),
        ] {
            assert_eq!(
                decoded_failure(
                    ValueCodecError::RootRegistration {
                        path: "$".into(),
                        source
                    }
                    .into()
                ),
                expected
            );
        }
        assert_eq!(
            classify_failure(
                &runmat_runtime::build_runtime_error("ordinary language failure").build()
            ),
            NativeTargetTestFailure::LanguageException
        );
    }
}
