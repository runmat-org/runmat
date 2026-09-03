use serde::{Deserialize, Serialize};

use super::{coordinate_frame::CoordinateFrame, signal::SignalId};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResponseSpectrumId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSpectrum {
    pub spectrum_id: ResponseSpectrumId,
    pub abscissa: SpectrumAbscissa,
    pub ordinates: Vec<f64>,
    pub damping: SpectrumDamping,
    pub response: SpectrumResponse,
    pub envelope: SpectrumEnvelope,
    pub interpolation: SpectrumInterpolation,
    pub extrapolation: SpectrumExtrapolation,
    pub direction: SpectrumDirection,
    pub modal_combination: ModalCombination,
    pub directional_combination: DirectionalCombination,
    pub truncation: ModalTruncationPolicy,
    pub construction: SpectrumConstruction,
    pub evidence: SpectrumEvidence,
    pub provenance_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumAbscissa {
    FrequencyHertz(Vec<f64>),
    PeriodSeconds(Vec<f64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumDamping {
    Ratio(f64),
    QualityFactor(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumResponse {
    RelativeDisplacement,
    RelativeVelocity,
    RelativeAcceleration,
    AbsoluteAcceleration,
    PseudoVelocity,
    PseudoAcceleration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumEnvelope {
    Positive,
    Negative,
    Maximax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumInterpolation {
    LinearFrequency,
    LinearPeriod,
    LogLog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumExtrapolation {
    Reject,
    HoldEndpoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectrumDirection {
    pub frame: CoordinateFrame,
    pub vector: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalCombination {
    Srss,
    CompleteQuadraticCombination,
    TenPercent,
    AbsoluteSum,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionalCombination {
    Srss,
    AbsoluteSum,
    PercentageRule { primary: f64, secondary: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectrumConstruction {
    AuthoredDesign,
    DerivedFromAccelerationHistory {
        source_signal_id: SignalId,
        method: ShockResponseSpectrumMethod,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShockResponseSpectrumMethod {
    ExactPiecewiseLinear,
    RecursiveFilter,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalTruncationPolicy {
    pub minimum_cumulative_mass_ratio: f64,
    pub include_residual_mode: bool,
    pub include_rigid_response: bool,
    pub include_missing_mass: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectrumEvidence {
    pub sampling_adequacy_ratio: f64,
    pub max_interpolation_error: Option<f64>,
}
