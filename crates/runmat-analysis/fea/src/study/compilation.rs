use std::collections::BTreeMap;

use runmat_analysis_core::{
    realize_analysis_model, validate_authored_outputs, AnalysisModel,
    AnalysisModelRealizationError, AnalysisModelRealizationInput, AuthoredMaterialRef,
    AuthoredOutput, AuthoredStudyInputBinding, CoupledPhysicsFamily, MaterialModel, PhysicsFamily,
    RealizedStudy,
};
use runmat_canonical_codec::CanonicalLimits;
use runmat_execution::Digest;
use runmat_geometry_core::{GeometryDefinitionSource, GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::properties::{
    compile_property_map, realize_formulation_properties, CompiledPropertyMap,
    PropertyCompilationError, PropertyRealizationError,
};

mod storage;

const STUDY_IDENTITY_PREFIX: &[u8] = b"runmat-compiled-analysis-study\0";
const STUDY_IDENTITY_LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(
    CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 65_536, 64),
);

/// Exact compiled study definition shared by local and remote execution.
///
/// This contract contains no worker, retry, accelerator, or placement policy. Those decisions may
/// change without changing the admitted model, mesh, observations, or constitutive ownership.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledAnalysisStudy {
    pub module_path: String,
    pub study_id: String,
    pub study_inputs: Vec<AuthoredStudyInputBinding>,
    pub outputs: Vec<AuthoredOutput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assessments: Vec<runmat_analysis_core::AuthoredAssessment>,
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
        if requires_thermal_solid_capabilities(&self.model) {
            self.properties.validate_thermal_solid_capabilities()?;
        }
        validate_study_inputs(&self.study_inputs)?;
        validate_authored_outputs(&self.outputs, &self.model.sequence.steps)
            .map_err(|error| AnalysisStudyCompilationError::InvalidOutputs(error.to_string()))?;
        runmat_analysis_core::validate_assessments(
            &self.assessments,
            &runmat_analysis_core::authored_output_types(&self.outputs, &self.model.sequence.steps)
                .map_err(|e| AnalysisStudyCompilationError::InvalidOutputs(e.to_string()))?,
        )
        .map_err(AnalysisStudyCompilationError::InvalidOutputs)?;
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
        // Streamed: the compiled study (mesh-sized property identity) is never materialized.
        let baseline = crate::contracts::canonical_digest::canonical_contract_digest(
            STUDY_IDENTITY_PREFIX,
            "compiled-analysis-study",
            &(
                &self.module_path,
                &self.study_id,
                &self.study_inputs,
                &self.outputs,
                &self.model,
                &self.properties.identity,
                self.solver_mesh_digest,
            ),
            STUDY_IDENTITY_LIMITS,
        )
        .map_err(|error| AnalysisStudyCompilationError::Identity(error.to_string()))?;
        if self.assessments.is_empty() {
            return Ok(baseline);
        }
        crate::contracts::canonical_digest::canonical_contract_digest(
            STUDY_IDENTITY_PREFIX,
            "assessed-analysis-study",
            &(baseline, &self.assessments),
            STUDY_IDENTITY_LIMITS,
        )
        .map_err(|error| AnalysisStudyCompilationError::Identity(error.to_string()))
    }

    /// This study with its definition digest, derived once for every holder of the pair.
    pub fn definition(&self) -> Result<AnalysisStudyDefinition<'_>, AnalysisStudyCompilationError> {
        Ok(AnalysisStudyDefinition {
            digest: self.definition_digest()?,
            compiled: self,
        })
    }
}

/// A compiled study and its definition digest. Only [`CompiledAnalysisStudy::definition`]
/// pairs them, so the digest is the study's own and holders never re-derive it.
#[derive(Clone, Copy, Debug)]
pub struct AnalysisStudyDefinition<'a> {
    compiled: &'a CompiledAnalysisStudy,
    digest: Digest,
}

impl<'a> AnalysisStudyDefinition<'a> {
    pub fn compiled(&self) -> &'a CompiledAnalysisStudy {
        self.compiled
    }

    pub fn digest(&self) -> Digest {
        self.digest
    }

    /// Whether this definition is the one of `compiled` itself (the same study object).
    pub fn is_of(&self, compiled: &CompiledAnalysisStudy) -> bool {
        std::ptr::eq(self.compiled, compiled)
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
        units: runmat_types::physical::CoordinateUnits::canonical(),
        regions,
        materials,
    })?;
    let mut model = realize_formulation_properties(realized, mesh)?;
    heat_rate::distribute_heat_rate_sources(&mut model, mesh)?;
    let properties = compile_study_properties(&model, mesh)?;
    let compiled = CompiledAnalysisStudy {
        module_path: study.module_path.clone(),
        study_id: study.study_id.clone(),
        study_inputs: study.study_inputs.clone(),
        assessments: study.assessments.clone(),
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
    let properties = compile_property_map(model, Some(mesh))?;
    if requires_structural_solid_capabilities(model) {
        properties.validate_structural_solid_capabilities()?;
    }
    if requires_thermal_solid_capabilities(model) {
        properties.validate_thermal_solid_capabilities()?;
    }
    Ok(properties)
}

fn requires_thermal_solid_capabilities(model: &AnalysisModel) -> bool {
    model.sequence.steps.iter().any(|step| {
        matches!(
            step.family,
            PhysicsFamily::Thermal | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
        )
    })
}

fn requires_structural_solid_capabilities(model: &AnalysisModel) -> bool {
    model.sequence.steps.iter().any(|step| {
        matches!(
            step.family,
            PhysicsFamily::Structural
                | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
        )
    })
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
        let file_evidence_matches = match &geometry.source.unit_evidence {
            runmat_types::physical::LengthUnitEvidence::ExplicitFallback {
                declaration, ..
            } => imported.authored_units == Some(declaration.unit),
            _ => imported.unit_evidence == geometry.source.unit_evidence,
        };
        if !file_evidence_matches
            || imported.content_digest != geometry.source.content_digest
            || imported.format != geometry.source.format
            || matches!(&geometry.source.unit_evidence,runmat_types::physical::LengthUnitEvidence::ExplicitFallback{declaration,..} if imported.authored_units.is_some_and(|unit|unit!=declaration.unit))
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
    #[error("heat generation `{excitation_id}`: {reason}")]
    HeatRate {
        excitation_id: String,
        reason: String,
    },
}

mod heat_rate;

#[cfg(test)]
mod tests;
