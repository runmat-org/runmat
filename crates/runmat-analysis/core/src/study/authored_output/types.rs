use serde::{Deserialize, Serialize};

use crate::{
    AuthoredRegionSource, CompressionPolicy, FieldLocation, ObservationPurpose,
    ObservationQuantity, ObservationValueShape, PartialAccessPolicy, Position, RetentionPolicy,
    SamplingPolicy, SeriesWeighting, SignalPrecision,
};
use runmat_types::PhysicalDimension;

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
    SpatialReduction {
        reduction: AuthoredSpatialReduction,
        source: Box<AuthoredOutputExpression>,
        target: AuthoredRegionSource,
        ties: AuthoredExtremaTiePolicy,
    },
    SpatialSelection {
        source: Box<AuthoredOutputExpression>,
        target: AuthoredRegionSource,
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
    Moment { about: Position },
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
    Derived(PhysicalDimension),
    Dimensionless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredFieldDomain {
    Real,
    Complex,
    Statistical,
}
