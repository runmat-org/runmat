use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicsFamily {
    Structural,
    Thermal,
    Electromagnetic,
    Acoustic,
    Fluid,
    Coupled(CoupledPhysicsFamily),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoupledPhysicsFamily {
    Thermomechanical,
    Electrothermal,
    ConjugateHeatTransfer,
    FluidStructureInteraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisProcedure {
    Static,
    ModalExtraction,
    PrestressedModalExtraction,
    EigenvalueBuckling,
    TransientResponse,
    HarmonicResponse,
    PrestressedHarmonicResponse,
    ModalResponseSpectrum,
    RandomResponse,
    Nonlinear,
    Adaptive,
}

impl AnalysisProcedure {
    pub const ALL: [Self; 11] = [
        Self::Static,
        Self::ModalExtraction,
        Self::PrestressedModalExtraction,
        Self::EigenvalueBuckling,
        Self::TransientResponse,
        Self::HarmonicResponse,
        Self::PrestressedHarmonicResponse,
        Self::ModalResponseSpectrum,
        Self::RandomResponse,
        Self::Nonlinear,
        Self::Adaptive,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolutionSpace {
    Direct,
    ModalReduced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulationMaturity {
    Production,
    Experimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionFallbackPolicy {
    Forbid,
    AllowExperimental,
}
