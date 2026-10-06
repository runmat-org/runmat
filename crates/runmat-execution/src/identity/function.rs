use serde::{Deserialize, Serialize};

use super::{Digest, ProgramRevision};
use crate::ContractError;
use runmat_types::ProgramFunctionId;

/// A function's immutable program identity, independent of session-local IDs.
/// The program digest is derived from the validated complete ProgramRevision.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionBindingOrigin {
    pub program: Digest,
    pub function: ProgramFunctionId,
}

impl FunctionBindingOrigin {
    pub fn from_revision(
        revision: &ProgramRevision,
        function: ProgramFunctionId,
    ) -> Result<Self, ContractError> {
        Ok(Self {
            program: revision.identity_digest()?,
            function,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LanguageCompatibilityMode, ProgramEnvironment};

    #[test]
    fn same_revision_and_function_have_same_origin() {
        let revision = ProgramRevision::new(
            Digest::sha256("graph"),
            Digest::sha256("source"),
            ProgramEnvironment {
                semantic_schema: 1,
                compiler_schema: 1,
                runtime_fingerprint: Digest::sha256("runtime"),
                catalog_fingerprint: Digest::sha256("catalog"),
                compatibility_mode: LanguageCompatibilityMode::RunMat,
            },
        )
        .expect("valid revision");
        let first =
            FunctionBindingOrigin::from_revision(&revision, ProgramFunctionId(3)).expect("origin");
        assert_eq!(
            first,
            FunctionBindingOrigin::from_revision(&revision, ProgramFunctionId(3))
                .expect("same origin")
        );
        assert_ne!(
            first,
            FunctionBindingOrigin::from_revision(&revision, ProgramFunctionId(4))
                .expect("different function")
        );
    }
}
