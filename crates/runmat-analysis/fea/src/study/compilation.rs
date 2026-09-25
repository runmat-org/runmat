use std::collections::BTreeMap;

use runmat_analysis_core::{
    realize_analysis_model, validate_authored_outputs, AnalysisModel,
    AnalysisModelRealizationError, AnalysisModelRealizationInput, AuthoredMaterialRef,
    AuthoredOutput, AuthoredStudyInputBinding, FormulationRequirement, MaterialModel,
    PhysicsFamily, RealizedStudy,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_geometry_core::{GeometryDefinitionSource, GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::properties::{
    compile_property_map, compile_structural_property_map, realize_formulation_properties,
    CompiledPropertyMap, PropertyCompilationError, PropertyRealizationError,
};

const STUDY_IDENTITY_PREFIX: &[u8] = b"runmat-compiled-analysis-study\0";
const STUDY_IDENTITY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 65_536, 64);

/// Exact compiled study definition shared by local and remote execution.
///
/// This contract contains no worker, retry, accelerator, or placement policy. Those decisions may
/// change without changing the admitted model, mesh, observations, or constitutive ownership.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledAnalysisStudy {
    pub module_path: String,
    pub study_id: String,
    pub formulation_requirement: FormulationRequirement,
    pub study_inputs: Vec<AuthoredStudyInputBinding>,
    pub outputs: Vec<AuthoredOutput>,
    pub model: AnalysisModel,
    pub properties: CompiledPropertyMap,
    pub solver_mesh_digest: StableDigest,
}

impl CompiledAnalysisStudy {
    fn validate_contract(&self) -> Result<(), AnalysisStudyCompilationError> {
        if self.module_path.trim().is_empty() || self.study_id.trim().is_empty() {
            return Err(AnalysisStudyCompilationError::InvalidStudyIdentity);
        }
        self.properties
            .validate_integrity()
            .map_err(AnalysisStudyCompilationError::Identity)?;
        if requires_structural_solid_capabilities(&self.model) {
            self.properties.validate_structural_solid_capabilities()?;
        }
        validate_study_inputs(&self.study_inputs)?;
        validate_authored_outputs(&self.outputs, &self.model.sequence.steps)
            .map_err(|error| AnalysisStudyCompilationError::InvalidOutputs(error.to_string()))?;
        Ok(())
    }

    pub fn validate_against(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> Result<(), AnalysisStudyCompilationError> {
        mesh.validate_canonical()
            .map_err(|error| AnalysisStudyCompilationError::InvalidMesh(error.to_string()))?;
        self.validate_contract()?;
        if self.solver_mesh_digest != mesh.canonical_digest
            || self.properties.identity.mesh_digest != Some(mesh.canonical_digest)
            || self.properties.identity.geometry_id != self.model.geometry_id
            || self.properties.identity.geometry_revision != self.model.geometry_revision
        {
            return Err(AnalysisStudyCompilationError::IdentityMismatch);
        }
        let expected = compile_study_properties(&self.model, mesh)?;
        if expected != self.properties {
            return Err(AnalysisStudyCompilationError::IdentityMismatch);
        }
        Ok(())
    }

    /// Identity of the compiled analysis definition and exact topology, independent of placement.
    pub fn definition_digest(&self) -> Result<Digest, AnalysisStudyCompilationError> {
        self.validate_contract()?;
        let bytes = encode_contract(
            STUDY_IDENTITY_PREFIX,
            "compiled-analysis-study",
            &(
                &self.module_path,
                &self.study_id,
                self.formulation_requirement,
                &self.study_inputs,
                &self.outputs,
                &self.model,
                &self.properties.identity,
                self.solver_mesh_digest,
            ),
            STUDY_IDENTITY_LIMITS,
        )
        .map_err(|error| AnalysisStudyCompilationError::Identity(error.to_string()))?;
        Ok(Digest::sha256(bytes))
    }
}

pub struct AnalysisStudyCompilationInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
}

/// Compile a source-independent realized study after geometry and meshing have completed.
///
/// Exact region binding cannot happen earlier: constructed, derived, and generated geometry only
/// acquire persistent entities during materialization, and formulation elements only acquire
/// canonical solver-node identities during meshing.
pub fn compile_analysis_study(
    input: AnalysisStudyCompilationInput<'_>,
) -> Result<CompiledAnalysisStudy, AnalysisStudyCompilationError> {
    let AnalysisStudyCompilationInput {
        study,
        geometry,
        regions,
        materials,
        mesh,
    } = input;
    geometry
        .validate()
        .map_err(|error| AnalysisStudyCompilationError::InvalidGeometry(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisStudyCompilationError::InvalidMesh(error.to_string()))?;
    validate_realized_geometry_source(study, geometry)?;
    validate_geometry_mesh_identity(geometry, mesh)?;
    if study.module_path.trim().is_empty() || study.study_id.trim().is_empty() {
        return Err(AnalysisStudyCompilationError::InvalidStudyIdentity);
    }
    let geometry_revision = u32::try_from(geometry.revision.revision)
        .map_err(|_| AnalysisStudyCompilationError::GeometryRevisionOverflow)?;
    let realized = realize_analysis_model(AnalysisModelRealizationInput {
        study,
        geometry_id: &study.geometry.definition.geometry_id,
        geometry_revision,
        units: geometry.source.source_units,
        regions,
        materials,
    })?;
    let model = realize_formulation_properties(realized, mesh)?;
    let properties = compile_study_properties(&model, mesh)?;
    let compiled = CompiledAnalysisStudy {
        module_path: study.module_path.clone(),
        study_id: study.study_id.clone(),
        formulation_requirement: study.formulation_requirement,
        study_inputs: study.study_inputs.clone(),
        outputs: study.outputs.clone(),
        model,
        properties,
        solver_mesh_digest: mesh.canonical_digest,
    };
    compiled.validate_against(mesh)?;
    Ok(compiled)
}

fn compile_study_properties(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<CompiledPropertyMap, PropertyCompilationError> {
    if requires_structural_solid_capabilities(model) {
        compile_structural_property_map(model, Some(mesh))
    } else {
        compile_property_map(model, Some(mesh))
    }
}

fn requires_structural_solid_capabilities(model: &AnalysisModel) -> bool {
    model
        .sequence
        .steps
        .iter()
        .any(|step| step.family == PhysicsFamily::Structural)
}

fn validate_study_inputs(
    inputs: &[AuthoredStudyInputBinding],
) -> Result<(), AnalysisStudyCompilationError> {
    let mut ids = std::collections::BTreeSet::new();
    for input in inputs {
        if input.input_id.trim().is_empty() || !ids.insert(input.input_id.as_str()) {
            return Err(AnalysisStudyCompilationError::InvalidStudyInputs);
        }
        input
            .value
            .validate()
            .map_err(|_| AnalysisStudyCompilationError::InvalidStudyInputs)?;
    }
    Ok(())
}

fn validate_geometry_mesh_identity(
    geometry: &GeometryDocument,
    mesh: &SolverMeshArtifact,
) -> Result<(), AnalysisStudyCompilationError> {
    if geometry.source.content_digest.bytes() != mesh.geometry.source_digest.bytes()
        || geometry.revision.revision != mesh.geometry.geometry_revision
        || geometry.revision.persistent_mapping_version != mesh.geometry.persistent_mapping_version
    {
        return Err(AnalysisStudyCompilationError::GeometryMeshMismatch);
    }
    Ok(())
}

fn validate_realized_geometry_source(
    study: &RealizedStudy,
    geometry: &GeometryDocument,
) -> Result<(), AnalysisStudyCompilationError> {
    if let GeometryDefinitionSource::Imported(imported) = &study.geometry.source {
        if imported.content_digest != geometry.source.content_digest
            || imported.format != geometry.source.format
            || imported.units != geometry.source.source_units
        {
            return Err(AnalysisStudyCompilationError::ImportedGeometrySourceMismatch);
        }
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum AnalysisStudyCompilationError {
    #[error("realized study identity is empty")]
    InvalidStudyIdentity,
    #[error("geometry document is invalid: {0}")]
    InvalidGeometry(String),
    #[error("solver mesh is invalid: {0}")]
    InvalidMesh(String),
    #[error("geometry document and solver mesh identities differ")]
    GeometryMeshMismatch,
    #[error("imported geometry source identity differs from the realized study")]
    ImportedGeometrySourceMismatch,
    #[error("geometry revision exceeds the analysis model contract")]
    GeometryRevisionOverflow,
    #[error("compiled study identity differs from its model or mesh")]
    IdentityMismatch,
    #[error("compiled study inputs are invalid or have duplicate identifiers")]
    InvalidStudyInputs,
    #[error("compiled study outputs are invalid: {0}")]
    InvalidOutputs(String),
    #[error("compiled study identity encoding failed: {0}")]
    Identity(String),
    #[error(transparent)]
    Model(#[from] AnalysisModelRealizationError),
    #[error(transparent)]
    Formulation(#[from] PropertyRealizationError),
    #[error(transparent)]
    Properties(#[from] PropertyCompilationError),
}

#[cfg(test)]
mod tests;
