//! Coordinate-frame geometry shared by structural loads and constraints.

use runmat_analysis_core::CoordinateFrame;
use thiserror::Error;

const GEOMETRY_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub(crate) enum StructuralFrameError {
    #[error("the coordinate frame has degenerate axes at the evaluation point")]
    Degenerate,
    #[error("a boundary-normal frame requires an oriented boundary normal")]
    BoundaryNormalUnavailable,
    #[error("the coordinate frame is not defined for pointwise structural geometry")]
    Unsupported,
}

pub(crate) fn translation_basis(
    frame: &CoordinateFrame,
    point: [f64; 3],
    boundary_normal: Option<[f64; 3]>,
) -> Result<[[f64; 3]; 3], StructuralFrameError> {
    match frame {
        CoordinateFrame::Global => Ok(identity()),
        CoordinateFrame::Cartesian(frame) => {
            let x = unit_vector(frame.x_axis)?;
            let y = unit_vector(frame.y_axis)?;
            require_orthogonal(x, y)?;
            Ok([x, y, unit_vector(cross(x, y))?])
        }
        CoordinateFrame::Cylindrical(frame) => {
            let axis = unit_vector(frame.axis)?;
            let reference = unit_vector(frame.radial_reference)?;
            if magnitude(cross(axis, reference)) <= GEOMETRY_TOLERANCE {
                return Err(StructuralFrameError::Degenerate);
            }
            let from_origin = subtract(point, frame.origin.canonical_value());
            let radial = subtract(from_origin, scale(axis, dot(from_origin, axis)));
            let radial = unit_vector(radial)?;
            let azimuthal = unit_vector(cross(axis, radial))?;
            Ok([radial, azimuthal, axis])
        }
        CoordinateFrame::Spherical(frame) => {
            let radial = unit_vector(subtract(point, frame.origin.canonical_value()))?;
            let polar_axis = unit_vector(frame.polar_axis)?;
            let azimuth_reference = unit_vector(frame.azimuth_reference)?;
            require_orthogonal(polar_axis, azimuth_reference)?;
            let mut azimuthal = cross(polar_axis, radial);
            if magnitude(azimuthal) <= GEOMETRY_TOLERANCE {
                azimuthal = cross(radial, azimuth_reference);
            }
            let azimuthal = unit_vector(azimuthal)?;
            let polar = unit_vector(cross(azimuthal, radial))?;
            Ok([radial, polar, azimuthal])
        }
        CoordinateFrame::BoundaryNormalTangent => {
            let normal = boundary_normal.ok_or(StructuralFrameError::BoundaryNormalUnavailable)?;
            let normal = unit_vector(normal)?;
            let reference = least_aligned_axis(normal);
            let tangent = unit_vector(cross(reference, normal))?;
            Ok([normal, tangent, cross(normal, tangent)])
        }
        CoordinateFrame::Material | CoordinateFrame::Element => {
            Err(StructuralFrameError::Unsupported)
        }
    }
}

pub(crate) fn boundary_tangent_basis(
    normal: [f64; 3],
) -> Result<[[f64; 3]; 2], StructuralFrameError> {
    let normal = unit_vector(normal)?;
    let tangent = unit_vector(cross(least_aligned_axis(normal), normal))?;
    Ok([tangent, cross(normal, tangent)])
}

pub(crate) fn transform_vector(
    frame: &CoordinateFrame,
    components: [f64; 3],
    point: [f64; 3],
    boundary_normal: Option<[f64; 3]>,
) -> Result<[f64; 3], StructuralFrameError> {
    let basis = translation_basis(frame, point, boundary_normal)?;
    Ok([
        basis[0][0] * components[0] + basis[1][0] * components[1] + basis[2][0] * components[2],
        basis[0][1] * components[0] + basis[1][1] * components[1] + basis[2][1] * components[2],
        basis[0][2] * components[0] + basis[1][2] * components[1] + basis[2][2] * components[2],
    ])
}

pub(crate) fn scale_vector(vector: [f64; 3], factor: f64) -> [f64; 3] {
    scale(vector, factor)
}

fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn least_aligned_axis(vector: [f64; 3]) -> [f64; 3] {
    let component = vector
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
        .map(|(index, _)| index)
        .unwrap_or(0);
    identity()[component]
}

fn require_orthogonal(left: [f64; 3], right: [f64; 3]) -> Result<(), StructuralFrameError> {
    if dot(left, right).abs() > 1.0e-10 {
        return Err(StructuralFrameError::Degenerate);
    }
    Ok(())
}

pub(crate) fn unit_vector(vector: [f64; 3]) -> Result<[f64; 3], StructuralFrameError> {
    crate::algebra::normalized_vector(vector).map_err(|_| StructuralFrameError::Degenerate)
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn magnitude(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

/// Material-attached axes retain the supplied tangent, unlike an arbitrary world-axis frame.
pub(crate) fn material_surface_basis(
    normal: [f64; 3],
    tangent: [f64; 3],
) -> Result<[[f64; 3]; 3], StructuralFrameError> {
    let normal = unit_vector(normal)?;
    let tangent = unit_vector(tangent)?;
    require_orthogonal(normal, tangent)?;
    Ok([tangent, unit_vector(cross(normal, tangent))?, normal])
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{AxialFrame, Position, SphericalFrame};

    use super::*;

    #[test]
    fn cylindrical_basis_tracks_position_and_axis() {
        let basis = translation_basis(
            &CoordinateFrame::Cylindrical(AxialFrame {
                frame_id: "shaft".into(),
                origin: Position::new([0.0; 3]),
                axis: [0.0, 0.0, 1.0],
                radial_reference: [1.0, 0.0, 0.0],
            }),
            [0.0, 2.0, 3.0],
            None,
        )
        .unwrap();

        assert_eq!(basis, [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn spherical_basis_is_defined_at_a_pole_by_its_reference() {
        let basis = translation_basis(
            &CoordinateFrame::Spherical(SphericalFrame {
                frame_id: "sphere".into(),
                origin: Position::new([0.0; 3]),
                polar_axis: [0.0, 0.0, 1.0],
                azimuth_reference: [1.0, 0.0, 0.0],
            }),
            [0.0, 0.0, 2.0],
            None,
        )
        .unwrap();

        assert_eq!(basis[0], [0.0, 0.0, 1.0]);
        assert_eq!(basis[1], [1.0, 0.0, 0.0]);
        assert_eq!(basis[2], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn degenerate_frame_axes_have_a_domain_neutral_error() {
        let error = translation_basis(
            &CoordinateFrame::Cylindrical(AxialFrame {
                frame_id: "bad".into(),
                origin: Position::new([0.0; 3]),
                axis: [0.0, 0.0, 1.0],
                radial_reference: [0.0, 0.0, 2.0],
            }),
            [1.0, 0.0, 0.0],
            None,
        )
        .unwrap_err();

        assert_eq!(error, StructuralFrameError::Degenerate);
    }

    #[test]
    fn vector_components_are_resolved_in_the_local_basis() {
        let transformed = transform_vector(
            &CoordinateFrame::BoundaryNormalTangent,
            [2.0, 3.0, 5.0],
            [0.0; 3],
            Some([0.0, 0.0, 1.0]),
        )
        .unwrap();

        assert_eq!(transformed, [5.0, -3.0, 2.0]);
    }
}
