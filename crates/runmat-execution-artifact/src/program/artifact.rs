use minicbor::Encoder;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    InterpreterRegionSourcesV1, ProgramArtifactId, ProgramBuildRecipe, ProgramRecipeId,
    ProgramTarget,
};
use crate::{error::contract_error, ArtifactError, ArtifactResult};

pub const PROGRAM_ARTIFACT_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ExecutableForm {
    InterpreterBytecodeV1 = 0,
    InterpreterScriptV1 = 1,
    TestAttemptV1 = 2,
    /// Dependency-bearing executable unit; never aliases the retired V3 discriminant.
    ExecutableUnitV4 = 9,
    NativeObjectV1 = 4,
    MeshingWorkload = 5,
    InterpreterBytecodeV2 = 6,
    InterpreterScriptV2 = 7,
    /// Versioned region payload containing the exact source map and script.
    InterpreterRegionSourcesV1 = 8,
    AnalysisWorkload = 10,
    GeometryWorkload = 11,
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

pub(super) struct ValidatedArtifactContext {
    pub(super) source_map: Option<runmat_execution::ExecutableSourceMapV2>,
    pub(super) unit_admission: Option<runmat_execution::ExecutableUnitAdmission>,
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
        if self.form != ExecutableForm::ExecutableUnitV4 {
            return Ok(None);
        }
        runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&self.executable_bytes)
            .map(Some)
            .map_err(contract_error)
    }

    pub fn executable_unit_admission(
        &self,
    ) -> ArtifactResult<Option<runmat_execution::ExecutableUnitAdmission>> {
        if self.form != ExecutableForm::ExecutableUnitV4 {
            return Ok(None);
        }
        runmat_execution::ExecutableUnitEnvelope::admission(&self.executable_bytes)
            .map(Some)
            .map_err(contract_error)
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
        self.validate_context(recipe).map(|_| ())
    }

    pub(super) fn validate_context(
        &self,
        recipe: &ProgramBuildRecipe,
    ) -> ArtifactResult<ValidatedArtifactContext> {
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
        let mut context = ValidatedArtifactContext {
            source_map: None,
            unit_admission: self.executable_unit_admission()?,
        };
        if self.form == ExecutableForm::InterpreterRegionSourcesV1 {
            let region = InterpreterRegionSourcesV1::decode(
                &self.executable_bytes,
                &recipe.program_revision,
            )?;
            context.source_map = Some(region.source_map().clone());
        }
        if let Some(admission) = context.unit_admission.as_ref() {
            if admission.identity.program != recipe.program_revision
                || admission.identity.entrypoint != recipe.entrypoint
                || admission.interop != recipe.interop
            {
                return Err(ArtifactError::Identity(
                    "executable unit does not match its exact program revision, entrypoint, and interop manifest"
                        .into(),
                ));
            }
            let semantic_accelerators =
                runmat_execution::resource::accelerator_requirements_for_capabilities(
                    &admission.capabilities,
                )
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            if !runmat_execution::resource::accelerator_requests_satisfy_requirements(
                &recipe.accelerators,
                &semantic_accelerators,
            ) {
                return Err(ArtifactError::Identity(
                    "program recipe omits accelerator requirements declared by its executable"
                        .into(),
                ));
            }
            // The transport envelope remains a bounded, opaque byte container
            // here. Validate its canonical bytes and component digests without
            // interpreting MIR, analysis, bytecode, or registry payloads.
            let unit = self
                .executable_unit()?
                .expect("executable-unit form returns its validated envelope");
            context.source_map = match unit.manifest.revisions.source_map_schema {
                1 => None, // Explicit legacy source-less admission.
                2 => {
                    let component = unit
                        .component(runmat_execution::ExecutableComponentKind::SourceMap)
                        .ok_or_else(|| {
                            ArtifactError::Invalid("executable source map is missing".into())
                        })?;
                    Some(
                        runmat_execution::ExecutableSourceMapV2::decode(&component.bytes)
                            .map_err(contract_error)?,
                    )
                }
                version => {
                    return Err(ArtifactError::Invalid(format!(
                        "unsupported executable source-map schema {version}"
                    )))
                }
            };
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
        Ok(context)
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
