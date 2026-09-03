use runmat_geometry_core::UnitSystem;
use serde::{Deserialize, Serialize};

use super::{
    boundary::AnalysisBoundary,
    connector::AnalysisConnector,
    constraint::AnalysisConstraint,
    coordinate_frame::CoordinateFrame,
    domains::{CfdDomain, ElectroThermalDomain, ElectromagneticDomain, ThermoMechanicalDomain},
    excitation::{ExcitationAction, ExcitationCase},
    initial_condition::InitialCondition,
    interfaces::AnalysisInterface,
    loads::LoadCase,
    material_assignment::MaterialAssignment,
    materials::MaterialModel,
    response_spectrum::ResponseSpectrum,
    signal::PhysicalSignal,
    structure::StructuralModel,
};
use crate::AnalysisSequence;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisModelId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisModel {
    pub model_id: AnalysisModelId,
    pub geometry_id: String,
    pub geometry_revision: u32,
    pub units: UnitSystem,
    pub frame: CoordinateFrame,
    pub materials: Vec<MaterialModel>,
    #[serde(default)]
    pub material_assignments: Vec<MaterialAssignment>,
    #[serde(default)]
    pub structural: Option<StructuralModel>,
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
    pub loads: Vec<LoadCase>,
    pub sequence: AnalysisSequence,
}
