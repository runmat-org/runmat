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
    /// Per-length transmission-line parameters of a 2D cross-section (quasi-static TEM).
    LineParameters,
}

impl AnalysisProcedure {
    pub const ALL: [Self; 12] = [
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
        Self::LineParameters,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolutionSpace {
    Direct,
    ModalReduced,
}
