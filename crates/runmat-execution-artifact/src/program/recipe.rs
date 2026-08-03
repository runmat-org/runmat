use std::collections::BTreeSet;

use minicbor::Encoder;
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use serde::{Deserialize, Serialize};

use super::ProgramRecipeId;
use crate::{ArtifactError, ArtifactResult, ObjectDescriptor};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramBuildRecipe {
    pub schema_version: u16,
    pub program_revision: ProgramRevision,
    pub entrypoint: String,
    pub outputs: OutputContract,
    pub execution_mode: String,
    pub target_profile: String,
    pub features: BTreeSet<String>,
    pub compile_options: BTreeSet<String>,
    pub source_objects: Vec<ObjectDescriptor>,
    pub expected_artifact_id: Option<super::ProgramArtifactId>,
}

impl ProgramBuildRecipe {
    pub fn validate(&self) -> ArtifactResult<()> {
        self.program_revision
            .validate()
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        if self.schema_version != 1
            || !valid_token(&self.entrypoint, 512)
            || !valid_token(&self.execution_mode, 64)
            || !valid_token(&self.target_profile, 256)
            || self.features.iter().any(|value| !valid_token(value, 128))
            || self
                .compile_options
                .iter()
                .any(|value| !valid_token(value, 256))
            || self
                .source_objects
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(ArtifactError::Invalid(
                "program build recipe is not canonical".into(),
            ));
        }
        for source in &self.source_objects {
            source.validate()?;
        }
        Ok(())
    }

    pub fn id(&self) -> ArtifactResult<ProgramRecipeId> {
        self.validate()?;
        let revision = self
            .program_revision
            .canonical_bytes()
            .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
        let mut bytes = b"runmat-program-build-recipe-v1\0".to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .array(9)
            .and_then(|encoder| encoder.bytes(&revision))
            .and_then(|encoder| encoder.str(&self.entrypoint))
            .and_then(|encoder| encoder.u16(self.outputs.requested_outputs))
            .and_then(|encoder| encoder.str(&self.execution_mode))
            .and_then(|encoder| encoder.str(&self.target_profile))
            .and_then(|encoder| encoder.array(self.features.len() as u64))
            .map_err(encoding)?;
        for feature in &self.features {
            encoder.str(feature).map_err(encoding)?;
        }
        encoder
            .array(self.compile_options.len() as u64)
            .map_err(encoding)?;
        for option in &self.compile_options {
            encoder.str(option).map_err(encoding)?;
        }
        encoder
            .array(self.source_objects.len() as u64)
            .map_err(encoding)?;
        for source in &self.source_objects {
            encoder
                .array(5)
                .and_then(|encoder| encoder.u8(source.namespace as u8))
                .and_then(|encoder| encoder.str(&source.logical_name))
                .and_then(|encoder| encoder.bytes(source.digest.bytes()))
                .and_then(|encoder| encoder.u64(source.encoded_length))
                .and_then(|encoder| encoder.str(&source.media_type))
                .map_err(encoding)?;
        }
        match self.expected_artifact_id {
            Some(id) => encoder.bytes(id.0.bytes()).map_err(encoding)?,
            None => encoder.null().map_err(encoding)?,
        };
        Ok(ProgramRecipeId(Digest::sha256(bytes)))
    }
}

fn valid_token(value: &str, max: usize) -> bool {
    !value.is_empty()
        && value.len() <= max
        && value.is_ascii()
        && !value.chars().any(char::is_control)
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}
