use serde::{Deserialize, Serialize};

use super::quantity::Position;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum CoordinateFrame {
    Global,
    Cartesian(CartesianFrame),
    Cylindrical(AxialFrame),
    Spherical(SphericalFrame),
    Material,
    Element,
    BoundaryNormalTangent,
}

impl CoordinateFrame {
    pub fn context_label(&self) -> String {
        match self {
            Self::Global => "global".to_string(),
            Self::Cartesian(frame) => format!("cartesian:{}", frame.frame_id),
            Self::Cylindrical(frame) => format!("cylindrical:{}", frame.frame_id),
            Self::Spherical(frame) => format!("spherical:{}", frame.frame_id),
            Self::Material => "material".to_string(),
            Self::Element => "element".to_string(),
            Self::BoundaryNormalTangent => "boundary_normal_tangent".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CartesianFrame {
    pub frame_id: String,
    pub origin: Position,
    pub x_axis: [f64; 3],
    pub y_axis: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct AxialFrame {
    pub frame_id: String,
    pub origin: Position,
    pub axis: [f64; 3],
    pub radial_reference: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SphericalFrame {
    pub frame_id: String,
    pub origin: Position,
    pub polar_axis: [f64; 3],
    pub azimuth_reference: [f64; 3],
}
