use serde::{Deserialize, Serialize};

use super::quantity::FiniteQuantity;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueSchedule<T> {
    Constant(T),
    StepValues(Vec<StepValue<T>>),
    TimeHistory(TabulatedSchedule<T>),
    FrequencyResponse(TabulatedSchedule<T>),
    Parameter(StudyParameterValue<T>),
    FieldArtifact(FieldArtifactValue),
}

impl<T> ValueSchedule<T> {
    pub fn constant(&self) -> Option<&T> {
        match self {
            Self::Constant(value) => Some(value),
            _ => None,
        }
    }
}

impl<T: FiniteQuantity> ValueSchedule<T> {
    pub fn is_well_formed(&self) -> bool {
        match self {
            Self::Constant(value) => value.is_finite(),
            Self::StepValues(values) => {
                !values.is_empty()
                    && values
                        .iter()
                        .all(|entry| !entry.step_id.trim().is_empty() && entry.value.is_finite())
            }
            Self::TimeHistory(table) | Self::FrequencyResponse(table) => {
                table.abscissa.len() == table.values.len()
                    && !table.abscissa.is_empty()
                    && table.abscissa.iter().all(|value| value.is_finite())
                    && table.abscissa.windows(2).all(|pair| pair[0] < pair[1])
                    && table.values.iter().all(FiniteQuantity::is_finite)
            }
            Self::Parameter(value) => {
                !value.parameter_id.trim().is_empty() && value.scale.is_finite()
            }
            Self::FieldArtifact(value) => {
                !value.artifact_id.trim().is_empty()
                    && !value.field_id.trim().is_empty()
                    && !value.topology_digest.trim().is_empty()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepValue<T> {
    pub step_id: String,
    pub value: T,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TabulatedSchedule<T> {
    pub abscissa: Vec<f64>,
    pub values: Vec<T>,
    pub interpolation: ScheduleInterpolation,
    pub extrapolation: ScheduleExtrapolation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleInterpolation {
    Step,
    Linear,
    CubicHermite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleExtrapolation {
    Reject,
    Hold,
    Periodic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyParameterValue<T> {
    pub parameter_id: String,
    pub scale: T,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldArtifactValue {
    pub artifact_id: String,
    pub field_id: String,
    pub topology_digest: String,
}
