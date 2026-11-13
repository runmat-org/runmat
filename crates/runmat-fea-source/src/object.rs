//! Targeted canonical facts. Labels and source locations never substitute for
//! the checked domain values used by authoring and execution.
use crate::{
    FeaDeclaredMaterial, FeaModelMemberKind, FeaQueryError, FeaSourceCheckResult, FeaSourceIdentity,
};
use runmat_analysis_core::{
    AuthoredBoundary, AuthoredConnector, AuthoredConstraint, AuthoredExcitation,
    AuthoredExcitationCase, AuthoredInitialCondition, AuthoredInterface, AuthoredLayupDefinition,
    AuthoredOrientationDefinition, AuthoredParameterizedSignal, AuthoredSectionDefinition,
    PhysicalSignal, RandomExcitationSpectrum, ResponseSpectrum,
};
use runmat_analysis_core::{AuthoredExecutionPolicy, AuthoredModel, AuthoredStudy, AuthoredSweep};
use runmat_analysis_language::FeaSymbolKind;
use runmat_geometry_core::GeometryDefinition;
use runmat_meshing_core::MeshDefinition;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "definition",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum FeaObjectDefinition {
    Material(FeaDeclaredMaterial),
    Geometry(GeometryDefinition),
    Mesh(MeshDefinition),
    Model(AuthoredModel),
    Study(AuthoredStudy),
    Output(runmat_analysis_core::AuthoredOutput),
    Assessment(runmat_analysis_core::AuthoredAssessment),
    Sweep(AuthoredSweep),
    Execution(AuthoredExecutionPolicy),
    Section(AuthoredSectionDefinition),
    Layup(AuthoredLayupDefinition),
    Orientation(AuthoredOrientationDefinition),
    Constraint(AuthoredConstraint),
    Boundary(AuthoredBoundary),
    Connector(AuthoredConnector),
    Interface(AuthoredInterface),
    Excitation(AuthoredExcitation),
    ExcitationCase(AuthoredExcitationCase),
    InitialCondition(AuthoredInitialCondition),
    ResponseSpectrum(ResponseSpectrum),
    RandomExcitation(RandomExcitationSpectrum),
    Signal(PhysicalSignal),
    ParameterizedSignal(AuthoredParameterizedSignal),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaObjectDetails {
    pub identity: FeaSourceIdentity,
    pub definition: FeaObjectDefinition,
}

impl FeaSourceCheckResult {
    pub(crate) fn definition(
        &self,
        identity: &FeaSourceIdentity,
    ) -> Result<FeaObjectDetails, FeaQueryError> {
        if let FeaSourceIdentity::StudyMember {
            module_path,
            study_id,
            member_kind,
            member_id,
        } = identity
        {
            let study = self
                .program
                .studies
                .iter()
                .find(|study| {
                    study.module_path == module_path.as_str() && &study.study_id == study_id
                })
                .ok_or(FeaQueryError::NotFound)?;
            let definition = match member_kind {
                crate::FeaStudyMemberKind::Output => study
                    .outputs
                    .iter()
                    .find(|value| &value.output_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Output),
                crate::FeaStudyMemberKind::Assessment => study
                    .assessments
                    .iter()
                    .find(|value| &value.assessment_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Assessment),
            }
            .ok_or(FeaQueryError::NotFound)?;
            return Ok(FeaObjectDetails {
                identity: identity.clone(),
                definition,
            });
        }
        if let FeaSourceIdentity::ModelMember {
            module_path,
            model_id,
            member_kind,
            member_id,
        } = identity
        {
            let model = self
                .program
                .models
                .iter()
                .find(|value| {
                    value.module_path == module_path.as_str() && &value.model_id == model_id
                })
                .ok_or(FeaQueryError::NotFound)?;
            let definition = match member_kind {
                FeaModelMemberKind::Section => model
                    .sections
                    .iter()
                    .find(|v| v.id() == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Section),
                FeaModelMemberKind::Layup => model
                    .layups
                    .iter()
                    .find(|v| &v.layup_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Layup),
                FeaModelMemberKind::Orientation => model
                    .orientations
                    .iter()
                    .find(|v| v.id() == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Orientation),
                FeaModelMemberKind::Constraint => model
                    .constraints
                    .iter()
                    .find(|v| &v.constraint_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Constraint),
                FeaModelMemberKind::Load => model
                    .boundaries
                    .iter()
                    .find(|v| &v.boundary_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Boundary),
                FeaModelMemberKind::LoadCase => model
                    .excitation_cases
                    .iter()
                    .find(|v| &v.case_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::ExcitationCase),
                FeaModelMemberKind::Interface => model
                    .interfaces
                    .iter()
                    .find(|v| &v.interface_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Interface),
                FeaModelMemberKind::Connector => model
                    .connectors
                    .iter()
                    .find(|v| &v.connector_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Connector),
                FeaModelMemberKind::Excitation => model
                    .excitations
                    .iter()
                    .find(|v| &v.excitation_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Excitation),
                FeaModelMemberKind::ResponseSpectrum => model
                    .response_spectra
                    .iter()
                    .find(|v| &v.spectrum_id.0 == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::ResponseSpectrum),
                FeaModelMemberKind::RandomExcitation => model
                    .random_excitations
                    .iter()
                    .find(|v| &v.excitation_id.0 == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::RandomExcitation),
                FeaModelMemberKind::InitialCondition => model
                    .initial_conditions
                    .iter()
                    .find(|v| &v.initial_condition_id == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::InitialCondition),
                FeaModelMemberKind::Signal => model
                    .signals
                    .iter()
                    .find(|v| &v.signal_id().0 == member_id)
                    .cloned()
                    .map(FeaObjectDefinition::Signal)
                    .or_else(|| {
                        model
                            .parameterized_signals
                            .iter()
                            .find(|v| &v.signal_id.0 == member_id)
                            .cloned()
                            .map(FeaObjectDefinition::ParameterizedSignal)
                    }),
            }
            .ok_or(FeaQueryError::NotFound)?;
            return Ok(FeaObjectDetails {
                identity: identity.clone(),
                definition,
            });
        }
        let FeaSourceIdentity::Declaration {
            module_path,
            symbol,
        } = identity
        else {
            return Err(FeaQueryError::DefinitionUnavailable);
        };
        let module = module_path.as_str();
        let name = &symbol.name;
        let definition = match symbol.kind {
            FeaSymbolKind::Material => self
                .program
                .materials
                .iter()
                .find(|v| &v.module_path == module_path && &v.material.material_id == name)
                .cloned()
                .map(FeaObjectDefinition::Material),
            FeaSymbolKind::Geometry => self
                .program
                .geometries
                .iter()
                .find(|v| v.module_path == module && &v.geometry_id == name)
                .cloned()
                .map(FeaObjectDefinition::Geometry),
            FeaSymbolKind::Mesh => self
                .program
                .meshes
                .iter()
                .find(|v| v.module_path == module && &v.mesh_id == name)
                .cloned()
                .map(FeaObjectDefinition::Mesh),
            FeaSymbolKind::Model => self
                .program
                .models
                .iter()
                .find(|v| v.module_path == module && &v.model_id == name)
                .cloned()
                .map(FeaObjectDefinition::Model),
            FeaSymbolKind::Study => self
                .program
                .studies
                .iter()
                .find(|v| v.module_path == module && &v.study_id == name)
                .cloned()
                .map(FeaObjectDefinition::Study),
            FeaSymbolKind::Sweep => self
                .program
                .sweeps
                .iter()
                .find(|v| v.module_path == module && &v.sweep_id == name)
                .cloned()
                .map(FeaObjectDefinition::Sweep),
            FeaSymbolKind::Execution => self
                .program
                .execution_policies
                .iter()
                .find(|v| v.module_path == module && &v.policy_id == name)
                .cloned()
                .map(FeaObjectDefinition::Execution),
            // A table is lowered into the boundaries that use it; it has no standalone object.
            FeaSymbolKind::Function | FeaSymbolKind::Table => {
                return Err(FeaQueryError::DefinitionUnavailable)
            }
        }
        .ok_or(FeaQueryError::NotFound)?;
        Ok(FeaObjectDetails {
            identity: identity.clone(),
            definition,
        })
    }
}
