//! What a drawn boundary surface leaves out of the solution.
use super::{cast_position, deform_position, field_magnitude, FieldData};
use crate::analysis::{AnalysisFieldLocation, AnalysisRenderTopology};

/// What a drawn surface leaves out of the solution on the boundary.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SurfaceApproximation {
    /// `full`: every boundary node is drawn where it is. `corners`: each quadratic face is one
    /// triangle through its corners.
    pub detail: &'static str,
    /// Largest distance between a dropped midside node (as drawn, deformation included) and the
    /// midpoint of the drawn edge, in metres.
    pub position_error_m: f64,
    /// Largest difference between the colored value at a dropped midside node and the linear
    /// color drawn there; zero for element fields, absent without a field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_error: Option<f64>,
}

impl SurfaceApproximation {
    pub(super) fn of_dropped(
        detail: crate::analysis::SurfaceDetail,
        topology: &AnalysisRenderTopology,
        dropped: &[[u32; 3]],
        field: Option<&FieldData>,
        deformation: Option<&FieldData>,
        scale: f64,
    ) -> Result<Self, String> {
        let mut approximation = Self {
            detail: match detail {
                crate::analysis::SurfaceDetail::Full => "full",
                crate::analysis::SurfaceDetail::Corners => "corners",
            },
            position_error_m: 0.0,
            value_error: field.map(|_| 0.0),
        };
        let Some(mesh) = topology.meshes.first() else {
            return Ok(approximation);
        };
        for &[a, b, midside] in dropped {
            let drawn = |node: u32| {
                deform_position(
                    cast_position(mesh.vertices[node as usize]),
                    Some(node as usize),
                    deformation,
                    scale,
                )
                .map(|p| p.map(f64::from))
            };
            let (pa, pb, pm) = (drawn(a)?, drawn(b)?, drawn(midside)?);
            let gap = (0..3)
                .map(|axis| (pm[axis] - 0.5 * (pa[axis] + pb[axis])).powi(2))
                .sum::<f64>()
                .sqrt();
            approximation.position_error_m = approximation.position_error_m.max(gap);
            if let Some(field) = field.filter(|field| field.location == AnalysisFieldLocation::Node)
            {
                let value = |node: u32| field_magnitude(field, node as usize, 1);
                let error = (value(midside)? - 0.5 * (value(a)? + value(b)?)).abs();
                if let Some(worst) = approximation.value_error.as_mut() {
                    *worst = worst.max(error);
                }
            }
        }
        Ok(approximation)
    }
}
