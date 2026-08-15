use minicbor::Encoder;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{ProgramArtifactId, ProgramBuildRecipe, ProgramRecipeId, ProgramTarget};
use crate::{ArtifactError, ArtifactResult};

pub const PROGRAM_ARTIFACT_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ExecutableForm {
    InterpreterBytecodeV1 = 0,
    InterpreterScriptV1 = 1,
    TestAttemptV1 = 2,
    ExecutableUnitV3 = 3,
    NativeObjectV1 = 4,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramArtifact {
    pub schema_version: u16,
    pub id: ProgramArtifactId,
    pub recipe_id: ProgramRecipeId,
    pub target: ProgramTarget,
    pub form: ExecutableForm,
    pub executable_bytes: Vec<u8>,
}

impl ProgramArtifact {
    pub fn native_object(&self) -> ArtifactResult<Option<super::NativeObjectPayload>> {
        if self.form != ExecutableForm::NativeObjectV1 {
            return Ok(None);
        }
        super::NativeObjectPayload::from_canonical_bytes(&self.executable_bytes).map(Some)
    }

    pub fn executable_unit(
        &self,
    ) -> ArtifactResult<Option<runmat_execution::ExecutableUnitEnvelope>> {
        if self.form != ExecutableForm::ExecutableUnitV3 {
            return Ok(None);
        }
        runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&self.executable_bytes)
            .map(Some)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))
    }

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
        recipe.target.validate_form(form)?;
        let id = derive_id(recipe_id, &recipe.target, form, &executable_bytes)?;
        let artifact = Self {
            schema_version: PROGRAM_ARTIFACT_SCHEMA_VERSION,
            id,
            recipe_id,
            target: recipe.target.clone(),
            form,
            executable_bytes,
        };
        artifact.validate_against(recipe)?;
        Ok(artifact)
    }

    pub fn validate_against(&self, recipe: &ProgramBuildRecipe) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_ARTIFACT_SCHEMA_VERSION
            || self.recipe_id != recipe.id()?
            || self.target != recipe.target
            || self.id
                != derive_id(
                    self.recipe_id,
                    &self.target,
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
        self.target.validate_form(self.form)?;
        if let Some(envelope) = self.executable_unit()? {
            if envelope.manifest.identity.program != recipe.program_revision {
                return Err(ArtifactError::Identity(
                    "executable unit does not match its exact program revision".into(),
                ));
            }
        }
        if self.form == ExecutableForm::NativeObjectV1 {
            let payload = self
                .native_object()?
                .expect("native-object form returns its validated payload");
            let native = self.target.native.as_ref().ok_or_else(|| {
                ArtifactError::Invalid("native object artifact has no native target".into())
            })?;
            if payload.object_format != native.object_format {
                return Err(ArtifactError::Identity(
                    "native object format differs from its target identity".into(),
                ));
            }
        }
        Ok(())
    }
}

fn derive_id(
    recipe_id: ProgramRecipeId,
    target: &ProgramTarget,
    form: ExecutableForm,
    executable_bytes: &[u8],
) -> ArtifactResult<ProgramArtifactId> {
    let mut bytes = b"runmat-program-artifact-v2\0".to_vec();
    let mut encoder = Encoder::new(&mut bytes);
    encoder
        .array(4)
        .and_then(|encoder| encoder.bytes(recipe_id.0.bytes()))
        .and_then(|encoder| {
            encoder.bytes(
                &target
                    .canonical_bytes()
                    .map_err(|_| minicbor::encode::Error::message("invalid program target"))?,
            )
        })
        .and_then(|encoder| encoder.u8(form as u8))
        .and_then(|encoder| encoder.bytes(executable_bytes))
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    Ok(ProgramArtifactId(Digest::sha256(bytes)))
}
