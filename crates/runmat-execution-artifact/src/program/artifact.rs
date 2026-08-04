use minicbor::Encoder;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{ProgramArtifactId, ProgramBuildRecipe, ProgramRecipeId};
use crate::{ArtifactError, ArtifactResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ExecutableForm {
    InterpreterBytecodeV1 = 0,
    InterpreterScriptV1 = 1,
    TestAttemptV1 = 2,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramArtifact {
    pub schema_version: u16,
    pub id: ProgramArtifactId,
    pub recipe_id: ProgramRecipeId,
    pub target_profile: String,
    pub form: ExecutableForm,
    pub executable_bytes: Vec<u8>,
}

impl ProgramArtifact {
    pub fn materialize(
        recipe: &ProgramBuildRecipe,
        form: ExecutableForm,
        executable_bytes: Vec<u8>,
    ) -> ArtifactResult<Self> {
        let recipe_id = recipe.id()?;
        if executable_bytes.is_empty() {
            return Err(ArtifactError::Invalid(
                "program artifact executable is empty".into(),
            ));
        }
        let id = derive_id(recipe_id, &recipe.target_profile, form, &executable_bytes)?;
        let artifact = Self {
            schema_version: 1,
            id,
            recipe_id,
            target_profile: recipe.target_profile.clone(),
            form,
            executable_bytes,
        };
        artifact.validate_against(recipe)?;
        Ok(artifact)
    }

    pub fn validate_against(&self, recipe: &ProgramBuildRecipe) -> ArtifactResult<()> {
        if self.schema_version != 1
            || self.recipe_id != recipe.id()?
            || self.target_profile != recipe.target_profile
            || self.id
                != derive_id(
                    self.recipe_id,
                    &self.target_profile,
                    self.form,
                    &self.executable_bytes,
                )?
            || recipe
                .expected_artifact_id
                .is_some_and(|expected| expected != self.id)
        {
            return Err(ArtifactError::Identity(
                "program artifact does not converge with its exact recipe".into(),
            ));
        }
        Ok(())
    }
}

fn derive_id(
    recipe_id: ProgramRecipeId,
    target_profile: &str,
    form: ExecutableForm,
    executable_bytes: &[u8],
) -> ArtifactResult<ProgramArtifactId> {
    let mut bytes = b"runmat-program-artifact-v1\0".to_vec();
    let mut encoder = Encoder::new(&mut bytes);
    encoder
        .array(4)
        .and_then(|encoder| encoder.bytes(recipe_id.0.bytes()))
        .and_then(|encoder| encoder.str(target_profile))
        .and_then(|encoder| encoder.u8(form as u8))
        .and_then(|encoder| encoder.bytes(executable_bytes))
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    Ok(ProgramArtifactId(Digest::sha256(bytes)))
}
