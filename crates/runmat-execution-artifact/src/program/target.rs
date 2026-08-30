use serde::{Deserialize, Serialize};

use super::ExecutableForm;
use crate::{ArtifactError, ArtifactResult};
pub use runmat_execution::host::NativeTargetIdentity;

pub const PROGRAM_TARGET_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramTargetCohort {
    Portable,
    Native,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramTarget {
    pub schema_version: u16,
    pub profile: String,
    pub cohort: ProgramTargetCohort,
    pub native: Option<NativeTargetIdentity>,
}

impl ProgramTarget {
    pub fn portable(profile: impl Into<String>) -> Self {
        Self {
            schema_version: PROGRAM_TARGET_SCHEMA_VERSION,
            profile: profile.into(),
            cohort: ProgramTargetCohort::Portable,
            native: None,
        }
    }

    pub fn native(profile: impl Into<String>, target: NativeTargetIdentity) -> Self {
        Self {
            schema_version: PROGRAM_TARGET_SCHEMA_VERSION,
            profile: profile.into(),
            cohort: ProgramTargetCohort::Native,
            native: Some(target),
        }
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_TARGET_SCHEMA_VERSION || !valid_token(&self.profile, 256)
        {
            return Err(ArtifactError::Invalid(
                "program artifact target is not canonical".into(),
            ));
        }
        match (self.cohort, self.native.as_ref()) {
            (ProgramTargetCohort::Portable, None) => Ok(()),
            (ProgramTargetCohort::Native, Some(native)) => native
                .validate()
                .map_err(|error| ArtifactError::Invalid(error.to_string())),
            _ => Err(ArtifactError::Invalid(
                "program target cohort has inconsistent native identity".into(),
            )),
        }
    }

    pub fn validate_form(&self, form: ExecutableForm) -> ArtifactResult<()> {
        self.validate()?;
        let compatible = match form {
            ExecutableForm::NativeObjectV1 => self.cohort == ProgramTargetCohort::Native,
            ExecutableForm::InterpreterBytecodeV1
            | ExecutableForm::InterpreterScriptV1
            | ExecutableForm::TestAttemptV1
            | ExecutableForm::MeshingWorkload
            | ExecutableForm::ExecutableUnitV3 => self.cohort == ProgramTargetCohort::Portable,
        };
        if compatible {
            Ok(())
        } else {
            Err(ArtifactError::Invalid(
                "program executable form is incompatible with its target cohort".into(),
            ))
        }
    }

    pub fn validate_for_portable_host(&self) -> ArtifactResult<()> {
        self.validate()?;
        if self.cohort == ProgramTargetCohort::Portable {
            Ok(())
        } else {
            Err(ArtifactError::Invalid(
                "native program artifact is incompatible with a portable execution host".into(),
            ))
        }
    }

    pub fn validate_for_native_host(&self, host: &NativeTargetIdentity) -> ArtifactResult<()> {
        self.validate()?;
        host.validate()
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        match self.cohort {
            ProgramTargetCohort::Portable => Ok(()),
            ProgramTargetCohort::Native if self.native.as_ref() == Some(host) => Ok(()),
            ProgramTargetCohort::Native => Err(ArtifactError::Invalid(
                "native program artifact does not match this execution host".into(),
            )),
        }
    }

    pub fn canonical_bytes(&self) -> ArtifactResult<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| ArtifactError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> ArtifactResult<Self> {
        let target: Self = serde_json::from_slice(bytes)
            .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
        target.validate()?;
        if target.canonical_bytes()? != bytes {
            return Err(ArtifactError::Invalid(
                "program target encoding is not canonical".into(),
            ));
        }
        Ok(target)
    }
}

fn valid_token(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value.is_ascii()
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native() -> NativeTargetIdentity {
        NativeTargetIdentity {
            architecture: runmat_execution::host::NativeArchitecture::Aarch64,
            operating_system: runmat_execution::host::NativeOperatingSystem::Macos,
            pointer_width: 64,
            abi: runmat_execution::host::NativeAbi::new("runmat-native-abi-v1").unwrap(),
            object_format: runmat_execution::host::NativeObjectFormat::MachO,
        }
    }

    #[test]
    fn target_cohorts_reject_incompatible_forms_and_hosts() {
        let portable = ProgramTarget::portable("portable-executable-unit-v3");
        portable
            .validate_form(ExecutableForm::ExecutableUnitV3)
            .unwrap();
        assert!(portable
            .validate_form(ExecutableForm::NativeObjectV1)
            .is_err());

        let target = native();
        let native_program = ProgramTarget::native("native-object-v1", target.clone());
        native_program
            .validate_form(ExecutableForm::NativeObjectV1)
            .unwrap();
        assert!(native_program.validate_for_portable_host().is_err());
        native_program.validate_for_native_host(&target).unwrap();

        let mut different_host = target.clone();
        different_host.architecture = runmat_execution::host::NativeArchitecture::X86_64;
        assert!(native_program
            .validate_for_native_host(&different_host)
            .is_err());
    }

    #[test]
    fn target_cohorts_require_exactly_one_consistent_native_identity() {
        let mut portable_with_native = ProgramTarget::portable("portable-test");
        portable_with_native.native = Some(native());
        assert!(portable_with_native.validate().is_err());

        let mut native_without_identity = ProgramTarget::native("native-test", native());
        native_without_identity.native = None;
        assert!(native_without_identity.validate().is_err());
    }
}
