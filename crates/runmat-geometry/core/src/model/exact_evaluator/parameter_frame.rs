use crate::physical::{ParameterAxis, QuantityKind, UnitDefinition};
use serde::{Deserialize, Serialize};

/// Coordinate used by a geometry parameter, independent of the model's spatial units.
/// Native axes keep evaluator identity; they never acquire an invented SI dimension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GeometryParameterFrame {
    Physical {
        unit: UnitDefinition,
    },
    Native {
        evaluator: std::sync::Arc<str>,
        axis: GeometryParameterAxis,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryParameterAxis {
    Curve,
    SurfaceU,
    SurfaceV,
    Pcurve,
}

impl GeometryParameterFrame {
    pub fn native(evaluator: impl Into<std::sync::Arc<str>>, axis: GeometryParameterAxis) -> Self {
        Self::Native {
            evaluator: evaluator.into(),
            axis,
        }
    }
    pub fn radians() -> Self {
        Self::Physical {
            unit: UnitDefinition::try_for_kind(QuantityKind::Angle, 1.0, 0.0).expect("radian unit"),
        }
    }
    pub fn length(scale_to_meters: f64) -> Result<Self, crate::physical::UnitError> {
        Ok(Self::Physical {
            unit: UnitDefinition::try_for_kind(QuantityKind::Length, scale_to_meters, 0.0)?,
        })
    }
}
impl ParameterAxis for GeometryParameterFrame {
    fn validate_parameter_axis(&self) -> Result<(), String> {
        match self {
            Self::Native { evaluator, .. } if evaluator.trim().is_empty() => {
                Err("native parameter axis requires evaluator identity".into())
            }
            Self::Physical { unit } if unit.offset() != 0.0 => {
                Err("geometry parameter differentials cannot have affine offsets".into())
            }
            _ => Ok(()),
        }
    }
    fn parameter_unit(&self) -> Option<UnitDefinition> {
        match self {
            Self::Physical { unit } => Some(*unit),
            Self::Native { .. } => None,
        }
    }
}
