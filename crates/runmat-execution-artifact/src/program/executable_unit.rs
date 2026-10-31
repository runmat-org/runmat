//! Artifact construction for the original complete portable compiler product.
use super::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
};
use crate::{ArtifactError, ArtifactResult};
use runmat_execution::{ExecutableUnitEnvelope, OutputContract};

impl ProgramArtifact {
    pub fn materialize_executable_unit(
        envelope: &ExecutableUnitEnvelope,
        outputs: OutputContract,
        target: ProgramTarget,
    ) -> ArtifactResult<(ProgramBuildRecipe, Self)> {
        let program = envelope
            .canonical_bytes()
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if envelope.manifest.identity.entrypoint_function.is_none() {
            return Err(ArtifactError::Invalid(
                "executable unit has no entrypoint".into(),
            ));
        }
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: envelope.manifest.identity.program.clone(),
            // The issued entrypoint label and typed function identity are
            // distinct. A function's numeric ID cannot replace its label.
            entrypoint: envelope.manifest.identity.entrypoint.clone(),
            outputs,
            execution_mode: "portable-v4".into(),
            target,
            interop: envelope.manifest.interop.clone(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let artifact = Self::materialize(&recipe, ExecutableForm::ExecutableUnitV4, program)?;
        Ok((recipe, artifact))
    }
}
