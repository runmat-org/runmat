use runmat_aot::PersistedNativePair;
use runmat_execution::Digest;
use runmat_execution_artifact::target_test::{
    NativeTargetTestRequest, NativeTargetTestResult, NativeTargetTestStatus,
};
use serde::{Deserialize, Serialize};

use super::application::NativePairedApplicationReport;
use crate::{NativeRunnerError, NativeRunnerResult};

pub const NATIVE_PAIRED_QUALIFICATION_SCHEMA_VERSION: u16 = 1;
const MAX_QUALIFICATION_RECEIPT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedNativeApplication {
    pub exit_code: Option<i32>,
    pub exit_signal: Option<i32>,
    pub stdout_digest: Digest,
    pub stdout_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedNativeTest {
    pub request: NativeTargetTestRequest,
    pub result: NativeTargetTestResult,
    pub expected: NativeTargetTestStatus,
}

/// Canonical record of an already completed paired execution. Decoding this
/// receipt grants no permission to reopen or execute either physical image.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePairedQualificationReceipt {
    pub schema_version: u16,
    pub product: PersistedNativePair,
    pub application: QualifiedNativeApplication,
    pub tests: Vec<QualifiedNativeTest>,
}

impl NativePairedQualificationReceipt {
    pub fn new(
        product: PersistedNativePair,
        application: &NativePairedApplicationReport,
        mut tests: Vec<QualifiedNativeTest>,
    ) -> NativeRunnerResult<Self> {
        tests.sort_by(|left, right| {
            (&left.request.suite, &left.request.test)
                .cmp(&(&right.request.suite, &right.request.test))
        });
        let receipt = Self {
            schema_version: NATIVE_PAIRED_QUALIFICATION_SCHEMA_VERSION,
            product,
            application: QualifiedNativeApplication {
                exit_code: application.exit.code,
                exit_signal: application.exit.signal,
                stdout_digest: Digest::sha256(&application.stdout),
                stdout_bytes: application.stdout.len() as u64,
            },
            tests,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> NativeRunnerResult<()> {
        self.product
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        let binding = &self.product.manifest.binding;
        if self.schema_version != NATIVE_PAIRED_QUALIFICATION_SCHEMA_VERSION
            || self.application.exit_code != Some(0)
            || self.application.exit_signal.is_some()
            || self.tests.len() != binding.tests.len()
        {
            return Err(invalid("invalid paired qualification completion"));
        }
        for (qualified, bound) in self.tests.iter().zip(&binding.tests) {
            qualified
                .request
                .validate_for(binding)
                .map_err(|error| invalid(error.to_string()))?;
            qualified
                .result
                .validate_for(&qualified.request, binding)
                .map_err(|error| invalid(error.to_string()))?;
            if qualified.request.suite != bound.suite
                || qualified.request.test != bound.test
                || qualified.result.status != qualified.expected
            {
                return Err(invalid(
                    "qualified test differs from bound entry or expected control",
                ));
            }
        }
        Ok(())
    }

    pub fn to_canonical_bytes(&self) -> NativeRunnerResult<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| invalid(error.to_string()))?;
        if bytes.len() > MAX_QUALIFICATION_RECEIPT_BYTES {
            return Err(invalid(
                "paired qualification receipt exceeds its wire bound",
            ));
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> NativeRunnerResult<Self> {
        if bytes.len() > MAX_QUALIFICATION_RECEIPT_BYTES {
            return Err(invalid(
                "paired qualification receipt exceeds its wire bound",
            ));
        }
        let receipt: Self =
            serde_json::from_slice(bytes).map_err(|error| invalid(error.to_string()))?;
        if receipt.to_canonical_bytes()?.as_slice() != bytes {
            return Err(invalid(
                "paired qualification receipt is not canonically encoded",
            ));
        }
        Ok(receipt)
    }
}

fn invalid(message: impl Into<String>) -> NativeRunnerError {
    NativeRunnerError::Configuration(message.into())
}
