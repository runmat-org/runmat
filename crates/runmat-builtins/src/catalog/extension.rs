use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum MatlabReleaseHalf {
    A,
    B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct MatlabRelease {
    pub year: u16,
    pub half: MatlabReleaseHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExtensionMode {
    /// The form is rejected in the MATLAB profile with a compatibility error.
    RunMatOnly,
    /// Both profiles accept the form, selecting their documented behavior.
    /// This is not an admission gate and has no rejection identifier.
    ProfileDependentBehavior,
    /// Public form admitted by the pinned MATLAB profile once its release is
    /// reached; earlier pins may still admit it in RunMat extension mode.
    AvailableSince(MatlabRelease),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinExtensionDescriptor {
    pub id: &'static str,
    pub mode: BuiltinExtensionMode,
    pub description: &'static str,
    pub error_identifier: Option<&'static str>,
}

impl BuiltinExtensionDescriptor {
    pub fn validate_error_contract(&self) -> Result<(), &'static str> {
        match (self.mode, self.error_identifier) {
            (
                BuiltinExtensionMode::RunMatOnly | BuiltinExtensionMode::AvailableSince(_),
                Some(identifier),
            ) if identifier.starts_with("RunMat:compatibility:") => Ok(()),
            (BuiltinExtensionMode::ProfileDependentBehavior, None) => Ok(()),
            (BuiltinExtensionMode::RunMatOnly | BuiltinExtensionMode::AvailableSince(_), _) => {
                Err("rejected extension forms require a RunMat:compatibility: error identifier")
            }
            (BuiltinExtensionMode::ProfileDependentBehavior, Some(_)) => {
                Err("profile-dependent behavior must not declare a rejection identifier")
            }
        }
    }
}

pub const SPARSE_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sparse-integer-storage",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sparse integer storage is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SparseIntegerExtension"),
};

pub const SPARSE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPARSE_INTEGER_EXTENSION];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_error_contract_distinguishes_rejection_from_profile_selection() {
        for (mode, error_identifier, valid) in [
            (
                BuiltinExtensionMode::RunMatOnly,
                Some("RunMat:compatibility:Fixture"),
                true,
            ),
            (BuiltinExtensionMode::RunMatOnly, None, false),
            (
                BuiltinExtensionMode::AvailableSince(MatlabRelease {
                    year: 2026,
                    half: MatlabReleaseHalf::B,
                }),
                Some("RunMat:compatibility:Fixture"),
                true,
            ),
            (
                BuiltinExtensionMode::AvailableSince(MatlabRelease {
                    year: 2026,
                    half: MatlabReleaseHalf::B,
                }),
                None,
                false,
            ),
            (
                BuiltinExtensionMode::RunMatOnly,
                Some("RunMat:Fixture"),
                false,
            ),
            (BuiltinExtensionMode::ProfileDependentBehavior, None, true),
            (
                BuiltinExtensionMode::ProfileDependentBehavior,
                Some("RunMat:compatibility:Fixture"),
                false,
            ),
        ] {
            let descriptor = BuiltinExtensionDescriptor {
                id: "fixture",
                mode,
                description: "Schema fixture",
                error_identifier,
            };
            assert_eq!(descriptor.validate_error_contract().is_ok(), valid);
        }
    }
}
