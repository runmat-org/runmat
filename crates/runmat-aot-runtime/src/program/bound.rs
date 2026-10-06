//! Exact portable function-handle admission for the linked AOT program.
//!
//! A handle's numeric function field is only a cache. The immutable program
//! origin selects the native body; local leases belong to their prior owner.

use std::collections::HashMap;

use runmat_execution::{Digest, FunctionBindingOrigin, ProgramRevision};
use runmat_types::ProgramFunctionId;
use runmat_value::{BoundFunctionAuthority, BoundFunctionBinding};

use super::native_error;

pub(super) struct AdmittedBoundFunctions {
    program: Digest,
    by_origin: HashMap<FunctionBindingOrigin, ProgramFunctionId>,
}

impl AdmittedBoundFunctions {
    pub(super) fn from_program(
        manifest: &runmat_native_codegen::aot::AotProgramManifest,
    ) -> Result<Self, Box<runmat_runtime::RuntimeError>> {
        Self::from_revision(
            &manifest.executable.program,
            manifest.functions.iter().map(|function| function.function),
        )
    }

    fn from_revision(
        revision: &ProgramRevision,
        functions: impl IntoIterator<Item = ProgramFunctionId>,
    ) -> Result<Self, Box<runmat_runtime::RuntimeError>> {
        let program = revision
            .identity_digest()
            .map_err(|error| Box::new(native_error(error)))?;
        let mut by_origin = HashMap::new();
        for function in functions {
            let origin = FunctionBindingOrigin::from_revision(revision, function)
                .map_err(|error| Box::new(native_error(error)))?;
            if by_origin.insert(origin, function).is_some() {
                return Err(Box::new(native_error(
                    "standalone callable inventory repeats a function origin",
                )));
            }
        }
        Ok(Self { program, by_origin })
    }

    /// `None` means this handle belongs to a different execution owner, which
    /// may still be reached through the previously installed bound invoker.
    pub(super) fn resolve(
        &self,
        binding: &BoundFunctionBinding,
    ) -> Result<Option<ProgramFunctionId>, runmat_runtime::RuntimeError> {
        let BoundFunctionAuthority::Portable { origin } = &binding.authority else {
            return Ok(None);
        };
        if origin.program != self.program {
            return Ok(None);
        }
        let function = self.by_origin.get(origin).ok_or_else(|| {
            unavailable("portable function origin is absent from the linked AOT program")
        })?;
        if usize::try_from(function.0).ok() != Some(binding.function) {
            return Err(unavailable(
                "portable function cache disagrees with its linked AOT origin",
            ));
        }
        Ok(Some(*function))
    }
}

pub(super) fn unavailable(message: &str) -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error("UnavailableBoundFunction", message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{LanguageCompatibilityMode, ProgramEnvironment};

    fn revision(source: &str) -> ProgramRevision {
        ProgramRevision::new(
            Digest::sha256(format!("{source}-graph")),
            Digest::sha256(format!("{source}-source")),
            ProgramEnvironment {
                semantic_schema: 1,
                compiler_schema: 1,
                runtime_fingerprint: Digest::sha256("aot-runtime"),
                catalog_fingerprint: Digest::sha256("aot-catalog"),
                compatibility_mode: LanguageCompatibilityMode::RunMat,
            },
        )
        .expect("valid program revision")
    }

    #[test]
    fn portable_origin_selects_only_the_exact_linked_function() {
        let own = revision("own");
        let other = revision("other");
        let admitted = AdmittedBoundFunctions::from_revision(
            &own,
            [ProgramFunctionId(4), ProgramFunctionId(7)],
        )
        .expect("admit native inventory");
        let origin = FunctionBindingOrigin::from_revision(&own, ProgramFunctionId(4)).unwrap();
        assert_eq!(
            admitted
                .resolve(&BoundFunctionBinding::portable(4, origin))
                .unwrap(),
            Some(ProgramFunctionId(4))
        );
        assert_eq!(
            admitted
                .resolve(&BoundFunctionBinding::portable(
                    4,
                    FunctionBindingOrigin::from_revision(&other, ProgramFunctionId(4)).unwrap(),
                ))
                .unwrap(),
            None,
            "same numeric ID from another program cannot select this body"
        );
        for invalid in [
            BoundFunctionBinding::portable(7, origin),
            BoundFunctionBinding::portable(
                8,
                FunctionBindingOrigin::from_revision(&own, ProgramFunctionId(8)).unwrap(),
            ),
        ] {
            let error = admitted
                .resolve(&invalid)
                .expect_err("wrong own-program authority");
            assert_eq!(error.identifier(), Some("RunMat:UnavailableBoundFunction"));
        }
        assert_eq!(
            admitted
                .resolve(&BoundFunctionBinding::unavailable(4))
                .unwrap(),
            None,
            "an unowned local numeric ID never selects a native body"
        );
    }

    #[test]
    fn duplicate_origin_is_rejected_before_invoker_installation() {
        let error = AdmittedBoundFunctions::from_revision(
            &revision("duplicate"),
            [ProgramFunctionId(4), ProgramFunctionId(4)],
        )
        .err()
        .expect("duplicate function origin must fail");
        assert_eq!(error.identifier(), Some("RunMat:AotNativeCall"));
    }
}
