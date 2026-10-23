use runmat_execution::resource::ResourceRequest;
use runmat_execution::{Digest, ProgramInvocationContext};
use runmat_test::identity::{SuiteId, TestId};
use runmat_test::result::TerminalDisposition;
use serde::{Deserialize, Serialize};

use crate::program::ProgramExecutionInputs;
use crate::{ArtifactError, ArtifactResult};

use super::binding::NativePairedBuildBinding;
use super::{canonical_bytes, decode_canonical};

pub const NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION: u16 = 4;
const MAX_DIAGNOSTIC_BYTES: usize = 64 * 1024;

/// Inputs and placement bounds for one compiled test entry. The digest covers
/// the complete canonical request, including validated portable values.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTargetTestRequest {
    pub schema_version: u16,
    pub build_binding: Digest,
    pub suite: SuiteId,
    pub test: TestId,
    pub attempt: u32,
    pub inputs: ProgramExecutionInputs,
    pub resources: ResourceRequest,
    /// Per-attempt deterministic time inputs, covered by this request digest.
    /// Hosted ambient products must not supply a virtual clock policy.
    pub clock_inputs: Option<runmat_execution::host::NativeClockInputs>,
    /// Per-attempt local service inputs, sealed by this request digest.
    pub local_service_inputs: Option<runmat_execution::host::NativeLocalServiceInputs>,
}

impl NativeTargetTestRequest {
    pub fn validate_for(&self, binding: &NativePairedBuildBinding) -> ArtifactResult<()> {
        binding.validate()?;
        if self.schema_version != NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION
            || self.build_binding != binding.digest()?
            || binding.test(&self.suite, &self.test).is_none()
            || self.inputs.context != ProgramInvocationContext::Direct
        {
            return Err(invalid(
                "target-test request does not select a bound direct test entry",
            ));
        }
        self.inputs.validate()?;
        runmat_execution::host::validate_native_clock_inputs(
            Some(&binding.selected_product),
            self.clock_inputs.as_ref(),
        )
        .map_err(crate::error::contract_error)?;
        runmat_execution::host::validate_native_local_inputs(
            Some(&binding.selected_product),
            self.local_service_inputs.as_ref(),
        )
        .map_err(crate::error::contract_error)?;
        self.resources
            .validate()
            .map_err(crate::error::contract_error)?;
        Ok(())
    }

    pub fn digest_for(&self, binding: &NativePairedBuildBinding) -> ArtifactResult<Digest> {
        self.validate_for(binding)?;
        Ok(Digest::sha256(
            [
                b"runmat-native-target-test-request-v3\0".as_slice(),
                &canonical_bytes(self)?,
            ]
            .concat(),
        ))
    }

    pub fn to_canonical_bytes(
        &self,
        binding: &NativePairedBuildBinding,
    ) -> ArtifactResult<Vec<u8>> {
        self.validate_for(binding)?;
        canonical_bytes(self)
    }

    pub fn from_canonical_bytes(
        bytes: &[u8],
        binding: &NativePairedBuildBinding,
    ) -> ArtifactResult<Self> {
        let request: Self = decode_canonical(bytes)?;
        request.validate_for(binding)?;
        Ok(request)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTargetTestFailure {
    Assertion,
    LanguageException,
    ConfiguredOutOfMemory,
    Cancelled,
    Timeout,
    ProcessFailure,
    ProtocolFailure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeTargetTestStatus {
    Passed,
    Failed(NativeTargetTestFailure),
}

/// Receiver-observed managed-allocation high-water marks. These are not RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTargetTestUsage {
    pub configured_memory_bytes: u64,
    pub peak_managed_reserved_bytes: u64,
    pub peak_managed_claimed_bytes: u64,
    pub elapsed_millis: u64,
}

/// A typed receiver record. Decoding and structural validation do not prove
/// that the receiver ran, or authenticate the process that emitted this value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTargetTestResult {
    pub schema_version: u16,
    pub request: Digest,
    pub suite: SuiteId,
    pub test: TestId,
    pub attempt: u32,
    pub status: NativeTargetTestStatus,
    /// Absent when the receiver could not report its managed ledger (for
    /// example, a process crash). A host must not fabricate zero peaks.
    pub usage: Option<NativeTargetTestUsage>,
    pub diagnostic: String,
}

impl NativeTargetTestResult {
    pub fn validate_for(
        &self,
        request: &NativeTargetTestRequest,
        binding: &NativePairedBuildBinding,
    ) -> ArtifactResult<()> {
        request.validate_for(binding)?;
        if self.schema_version != NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION
            || self.request != request.digest_for(binding)?
            || self.suite != request.suite
            || self.test != request.test
            || self.attempt != request.attempt
            || self.diagnostic.len() > MAX_DIAGNOSTIC_BYTES
        {
            return Err(invalid(
                "target-test result is inconsistent with its request",
            ));
        }
        let receiver_terminal = matches!(
            self.status,
            NativeTargetTestStatus::Passed
                | NativeTargetTestStatus::Failed(NativeTargetTestFailure::Assertion)
                | NativeTargetTestStatus::Failed(NativeTargetTestFailure::LanguageException)
                | NativeTargetTestStatus::Failed(NativeTargetTestFailure::ConfiguredOutOfMemory)
                | NativeTargetTestStatus::Failed(NativeTargetTestFailure::Cancelled)
        );
        if receiver_terminal && self.usage.is_none() {
            return Err(invalid("receiver-completed test has no managed usage"));
        }
        if let Some(usage) = self.usage {
            if usage.configured_memory_bytes != request.resources.memory_bytes
                || usage.peak_managed_reserved_bytes > usage.configured_memory_bytes
                || usage.peak_managed_claimed_bytes > usage.configured_memory_bytes
            {
                return Err(invalid("target-test managed usage exceeds its request"));
            }
        }
        Ok(())
    }

    pub fn terminal_disposition(&self) -> TerminalDisposition {
        match self.status {
            NativeTargetTestStatus::Passed => TerminalDisposition::Passed,
            NativeTargetTestStatus::Failed(NativeTargetTestFailure::Timeout) => {
                TerminalDisposition::TimedOut
            }
            NativeTargetTestStatus::Failed(NativeTargetTestFailure::Cancelled) => {
                TerminalDisposition::Cancelled
            }
            NativeTargetTestStatus::Failed(NativeTargetTestFailure::ProcessFailure) => {
                TerminalDisposition::Crashed
            }
            NativeTargetTestStatus::Failed(_) => TerminalDisposition::Failed,
        }
    }

    pub fn to_canonical_bytes(
        &self,
        request: &NativeTargetTestRequest,
        binding: &NativePairedBuildBinding,
    ) -> ArtifactResult<Vec<u8>> {
        self.validate_for(request, binding)?;
        canonical_bytes(self)
    }

    pub fn from_canonical_bytes(
        bytes: &[u8],
        request: &NativeTargetTestRequest,
        binding: &NativePairedBuildBinding,
    ) -> ArtifactResult<Self> {
        let result: Self = decode_canonical(bytes)?;
        result.validate_for(request, binding)?;
        Ok(result)
    }
}

fn invalid(message: &str) -> ArtifactError {
    ArtifactError::Invalid(message.into())
}
