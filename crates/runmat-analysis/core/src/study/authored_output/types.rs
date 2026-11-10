use serde::{Deserialize, Serialize};

use crate::{
    AuthoredRegionSource, CompressionPolicy, FieldLocation, ObservationPurpose,
    ObservationQuantity, ObservationValueShape, PartialAccessPolicy, Position, RetentionPolicy,
    SamplingPolicy, SeriesWeighting, SignalPrecision,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredOutput {
    pub output_id: String,
    pub expression: AuthoredOutputExpression,
    pub policy: AuthoredOutputPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredOutputPolicy {
    pub precision: Option<SignalPrecision>,
    pub compression: CompressionPolicy,
    pub retention: RetentionPolicy,
    pub partial_access: PartialAccessPolicy,
    pub purpose: ObservationPurpose,
}

impl Default for AuthoredOutputPolicy {
    fn default() -> Self {
        Self {
            precision: None,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
            purpose: ObservationPurpose::Publication,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredOutputExpression {
    AnalysisField {
        step_id: String,
        quantity: ObservationQuantity,
        location: FieldLocation,
    },
    OutputReference {
        output_id: String,
    },
    Projection {
        projection: AuthoredFieldProjection,
        source: Box<AuthoredOutputExpression>,
    },
    Recovery {
        source: Box<AuthoredOutputExpression>,
        output_location: FieldLocation,
        recovery: crate::RecoveryPolicy,
    },
    SpatialReduction {
        reduction: AuthoredSpatialReduction,
        source: Box<AuthoredOutputExpression>,
        target: AuthoredRegionSource,
        ties: AuthoredExtremaTiePolicy,
    },
    /// One analysis field reduced over two face regions into one scalar per sample:
    /// `pressure_drop(analysis, from, to)` and `transmission_loss(analysis, incident, transmitted)`.
    PortReduction {
        reduction: AuthoredPortReduction,
        source: Box<AuthoredOutputExpression>,
        first: AuthoredRegionSource,
        second: AuthoredRegionSource,
    },
    SpatialSelection {
        source: Box<AuthoredOutputExpression>,
        target: AuthoredRegionSource,
    },
    /// A per-length line parameter of an `electromagnetic line_parameters` step:
    /// `analysis.<step>.<name>.<conductor>` for a per-line value, `analysis.<step>.<name>` for a
    /// pair or matrix value.
    LineParameter {
        step_id: String,
        quantity: ObservationQuantity,
        /// The signal conductor a per-line value reads; `None` for a pair or matrix value.
        conductor: Option<String>,
        /// Number of signal conductors of the section: the extent of the matrices.
        #[serde(with = "runmat_types::wire_integer")]
        signal_conductors: usize,
    },
    Interpolation {
        source: Box<AuthoredOutputExpression>,
        positions: Vec<Position>,
    },
    Sampling {
        source: Box<AuthoredOutputExpression>,
        sampling: SamplingPolicy,
        domain: AuthoredSeriesDomain,
    },
    SeriesReduction {
        reduction: AuthoredSeriesReduction,
        source: Box<AuthoredOutputExpression>,
        domain: AuthoredSeriesDomain,
        ties: AuthoredExtremaTiePolicy,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredFieldProjection {
    Magnitude,
    Component(AuthoredVectorComponent),
    Real,
    Imaginary,
    Phase,
    /// Sound pressure level of a complex acoustic pressure, dB re 20 uPa (RMS).
    Level,
    Principal(AuthoredPrincipalSelection),
    VonMises,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredPrincipalSelection {
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredVectorComponent {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredSpatialReduction {
    Integral,
    Mean,
    RootMeanSquare,
    EuclideanNorm,
    Minimum,
    Maximum,
    Resultant,
    Moment {
        about: Position,
    },
    /// ∫ v·n dA of a vector field through a face region.
    NormalFlux,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredPortReduction {
    /// Area-weighted mean over the first region minus that over the second (pressure drop).
    MeanDifference,
    /// 10·log10 of the incident port power over the transmitted port power, dB.
    TransmissionLoss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredExtremaTiePolicy {
    FirstCanonical,
    AllLocations,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredSeriesDomain {
    Time,
    Frequency,
    LoadFactor,
    StudyParameter { input_id: String },
    Mode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredSeriesReduction {
    Minimum,
    Maximum,
    AbsoluteMaximum,
    Mean { weighting: SeriesWeighting },
    RootMeanSquare { weighting: SeriesWeighting },
    Variance { weighting: SeriesWeighting },
    Envelope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredOutputType {
    Field {
        value: AuthoredOutputValue,
        shape: ObservationValueShape,
        domain: AuthoredFieldDomain,
        location: FieldLocation,
    },
    SpatialExtremum {
        value: AuthoredOutputValue,
    },
    Reduced {
        value: AuthoredOutputValue,
        shape: ObservationValueShape,
    },
    Probe {
        value: AuthoredOutputValue,
        shape: ObservationValueShape,
        domain: AuthoredFieldDomain,
    },
    History {
        sample: Box<AuthoredOutputType>,
        domain: AuthoredSeriesDomain,
    },
    SeriesExtremum {
        value: AuthoredOutputValue,
    },
    Envelope {
        value: AuthoredOutputValue,
        shape: ObservationValueShape,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredOutputValue {
    Physical(ObservationQuantity),
    Derived(runmat_types::physical::DynamicDimension),
    Dimensionless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredFieldDomain {
    Real,
    Complex,
    Statistical,
}
