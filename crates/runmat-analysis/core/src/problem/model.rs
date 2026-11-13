use serde::{Deserialize, Serialize};

use super::{
    boundary::AnalysisBoundary,
    connector::AnalysisConnector,
    constraint::AnalysisConstraint,
    coordinate_frame::CoordinateFrame,
    domains::{
        CfdDomain, ElectroThermalDomain, ElectromagneticDomain, ThermalDomain,
        ThermoMechanicalDomain,
    },
    excitation::{ExcitationAction, ExcitationCase},
    initial_condition::InitialCondition,
    interfaces::AnalysisInterface,
    material_assignment::MaterialAssignment,
    materials::MaterialModel,
    random_vibration::RandomExcitationSpectrum,
    response_spectrum::ResponseSpectrum,
    signal::PhysicalSignal,
    structure::StructuralModel,
};
mod solver_items;
pub use solver_items::{SolverMaterialAssignments, SolverMaterials, SolverSteps};

use crate::{
    AnalysisSequence, IndependentDomain, PhysicsFamily, ProcedureStep, ProcedureStepId,
    RestartPolicy, SequenceBranch, SequenceDependency, SolutionSpace, StateInput, StateTransition,
    StepObjectActivation,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisModelId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisModel {
    pub model_id: AnalysisModelId,
    pub geometry_id: String,
    pub geometry_revision: u32,
    pub units:
        runmat_types::physical::CoordinateUnits<runmat_types::physical::CanonicalCoordinates>,
    pub frame: CoordinateFrame,
    pub materials: Vec<MaterialModel>,
    #[serde(default)]
    pub material_assignments: Vec<MaterialAssignment>,
    #[serde(default)]
    pub structural: Option<StructuralModel>,
    #[serde(default)]
    pub thermal: Option<ThermalDomain>,
    #[serde(default)]
    pub thermo_mechanical: Option<ThermoMechanicalDomain>,
    #[serde(default)]
    pub electro_thermal: Option<ElectroThermalDomain>,
    #[serde(default)]
    pub electromagnetic: Option<ElectromagneticDomain>,
    #[serde(default)]
    pub cfd: Option<CfdDomain>,
    #[serde(default)]
    pub interfaces: Vec<AnalysisInterface>,
    #[serde(default)]
    pub constraints: Vec<AnalysisConstraint>,
    #[serde(default)]
    pub boundaries: Vec<AnalysisBoundary>,
    #[serde(default)]
    pub connectors: Vec<AnalysisConnector>,
    pub signals: Vec<PhysicalSignal>,
    pub excitations: Vec<ExcitationAction>,
    pub excitation_cases: Vec<ExcitationCase>,
    pub initial_conditions: Vec<InitialCondition>,
    pub response_spectra: Vec<ResponseSpectrum>,
    #[serde(default)]
    pub random_excitations: Vec<RandomExcitationSpectrum>,
    pub sequence: AnalysisSequence,
}

/// Canonical solver semantics of an authored step.
///
/// Observation, accuracy, and state-publication declarations intentionally do not appear here:
/// they consume a solver state but cannot change the equations that produce it.
#[derive(Serialize)]
pub struct SolverProcedureStep<'a> {
    pub step_id: &'a ProcedureStepId,
    pub family: PhysicsFamily,
    pub procedure: crate::AnalysisProcedure,
    pub solution_space: SolutionSpace,
    pub independent_domain: &'a IndependentDomain,
    pub activations: &'a [StepObjectActivation],
    pub excitation_case_selection: &'a crate::ExcitationCaseSelection,
    pub state_inputs: &'a [StateInput],
    pub restart: &'a RestartPolicy,
}

/// Canonical solver semantics of an authored procedure graph.
#[derive(Serialize)]
pub struct SolverAnalysisSequence<'a> {
    pub sequence_id: &'a str,
    pub steps: SolverSteps<'a>,
    pub dependencies: &'a [SequenceDependency],
    pub transitions: &'a [StateTransition],
    pub branches: &'a [SequenceBranch],
}

/// Canonical solver semantics of an analysis model, excluding downstream result policies.
#[derive(Serialize)]
pub struct SolverAnalysisModel<'a> {
    pub model_id: &'a AnalysisModelId,
    pub geometry_id: &'a str,
    pub geometry_revision: u32,
    pub units:
        runmat_types::physical::CoordinateUnits<runmat_types::physical::CanonicalCoordinates>,
    pub frame: &'a CoordinateFrame,
    pub materials: SolverMaterials<'a>,
    pub material_assignments: SolverMaterialAssignments<'a>,
    pub structural: &'a Option<StructuralModel>,
    pub thermal: &'a Option<ThermalDomain>,
    pub thermo_mechanical: &'a Option<ThermoMechanicalDomain>,
    pub electro_thermal: &'a Option<ElectroThermalDomain>,
    pub electromagnetic: &'a Option<ElectromagneticDomain>,
    pub cfd: &'a Option<CfdDomain>,
    pub interfaces: &'a [AnalysisInterface],
    pub constraints: &'a [AnalysisConstraint],
    pub boundaries: &'a [AnalysisBoundary],
    pub connectors: &'a [AnalysisConnector],
    pub signals: &'a [PhysicalSignal],
    pub excitations: &'a [ExcitationAction],
    pub excitation_cases: &'a [ExcitationCase],
    pub initial_conditions: &'a [InitialCondition],
    pub response_spectra: &'a [ResponseSpectrum],
    pub random_excitations: &'a [RandomExcitationSpectrum],
    pub sequence: SolverAnalysisSequence<'a>,
}

/// Numerical material ownership without importer/reconciliation audit evidence.
#[derive(Serialize)]
pub struct SolverMaterialAssignment<'a> {
    pub target: &'a crate::PropertyRegionTarget,
    pub assigned_material_id: &'a str,
}

impl AnalysisModel {
    pub fn solver_identity_contract(&self) -> SolverAnalysisModel<'_> {
        SolverAnalysisModel {
            model_id: &self.model_id,
            geometry_id: &self.geometry_id,
            geometry_revision: self.geometry_revision,
            units: self.units,
            frame: &self.frame,
            materials: SolverMaterials(&self.materials),
            material_assignments: SolverMaterialAssignments(&self.material_assignments),
            structural: &self.structural,
            thermal: &self.thermal,
            thermo_mechanical: &self.thermo_mechanical,
            electro_thermal: &self.electro_thermal,
            electromagnetic: &self.electromagnetic,
            cfd: &self.cfd,
            interfaces: &self.interfaces,
            constraints: &self.constraints,
            boundaries: &self.boundaries,
            connectors: &self.connectors,
            signals: &self.signals,
            excitations: &self.excitations,
            excitation_cases: &self.excitation_cases,
            initial_conditions: &self.initial_conditions,
            response_spectra: &self.response_spectra,
            random_excitations: &self.random_excitations,
            sequence: self.sequence.solver_identity_contract(),
        }
    }
}

impl AnalysisSequence {
    pub fn solver_identity_contract(&self) -> SolverAnalysisSequence<'_> {
        SolverAnalysisSequence {
            sequence_id: &self.sequence_id,
            steps: SolverSteps(&self.steps),
            dependencies: &self.dependencies,
            transitions: &self.transitions,
            branches: &self.branches,
        }
    }
}

impl crate::ProcedureStep {
    pub fn solver_identity_contract(&self) -> SolverProcedureStep<'_> {
        SolverProcedureStep {
            excitation_case_selection: &self.excitation_case_selection,
            step_id: &self.step_id,
            family: self.family,
            procedure: self.procedure,
            solution_space: self.solution_space,
            independent_domain: &self.independent_domain,
            activations: &self.activations,
            state_inputs: &self.state_inputs,
            restart: &self.restart,
        }
    }
}
