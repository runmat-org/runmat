use runmat_analysis_core::CoordinateFrame;

use super::vector::{cross, dot, magnitude, scale, subtract};
use super::ConstraintPlanError;

pub(super) fn translation_basis(
    frame: &CoordinateFrame,
    point: [f64; 3],
    boundary_normal: Option<[f64; 3]>,
    constraint_id: &str,
) -> Result<[[f64; 3]; 3], ConstraintPlanError> {
    match frame {
        CoordinateFrame::Global => Ok(identity()),
        CoordinateFrame::Cartesian(frame) => {
            let x = unit(frame.x_axis, constraint_id)?;
            let y = unit(frame.y_axis, constraint_id)?;
            require_orthogonal(x, y, constraint_id)?;
            Ok([x, y, unit(cross(x, y), constraint_id)?])
        }
        CoordinateFrame::Cylindrical(frame) => {
            let axis = unit(frame.axis, constraint_id)?;
            let reference = unit(frame.radial_reference, constraint_id)?;
            if cross(axis, reference)
                .iter()
                .all(|value| value.abs() <= 1.0e-12)
            {
                return Err(ConstraintPlanError::DegenerateFrame {
                    constraint_id: constraint_id.into(),
                });
            }
            let from_origin = subtract(point, frame.origin.meters);
            let radial = subtract(from_origin, scale(axis, dot(from_origin, axis)));
            let radial = unit(radial, constraint_id)?;
            let azimuthal = unit(cross(axis, radial), constraint_id)?;
            Ok([radial, azimuthal, axis])
        }
        CoordinateFrame::Spherical(frame) => {
            let radial = unit(subtract(point, frame.origin.meters), constraint_id)?;
            let polar_axis = unit(frame.polar_axis, constraint_id)?;
            let azimuth_reference = unit(frame.azimuth_reference, constraint_id)?;
            require_orthogonal(polar_axis, azimuth_reference, constraint_id)?;
            let mut azimuthal = cross(polar_axis, radial);
            if magnitude(azimuthal) <= 1.0e-12 {
                azimuthal = cross(radial, azimuth_reference);
            }
            let azimuthal = unit(azimuthal, constraint_id)?;
            let polar = unit(cross(azimuthal, radial), constraint_id)?;
            Ok([radial, polar, azimuthal])
        }
        CoordinateFrame::BoundaryNormalTangent => {
            let normal = boundary_normal.ok_or_else(|| ConstraintPlanError::NormalUnavailable {
                constraint_id: constraint_id.into(),
            })?;
            let normal = unit(normal, constraint_id)?;
            let reference = least_aligned_axis(normal);
            let tangent = unit(cross(reference, normal), constraint_id)?;
            Ok([normal, tangent, cross(normal, tangent)])
        }
        CoordinateFrame::Material | CoordinateFrame::Element => {
            Err(ConstraintPlanError::UnsupportedFrame {
                constraint_id: constraint_id.into(),
                frame: frame.context_label(),
            })
        }
    }
}

pub(super) fn boundary_tangent_basis(
    normal: [f64; 3],
    constraint_id: &str,
) -> Result<[[f64; 3]; 2], ConstraintPlanError> {
    let normal = unit(normal, constraint_id)?;
    let tangent = unit(cross(least_aligned_axis(normal), normal), constraint_id)?;
    Ok([tangent, cross(normal, tangent)])
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

fn require_orthogonal(
    left: [f64; 3],
    right: [f64; 3],
    constraint_id: &str,
) -> Result<(), ConstraintPlanError> {
    if dot(left, right).abs() > 1.0e-10 {
        return Err(ConstraintPlanError::DegenerateFrame {
            constraint_id: constraint_id.into(),
        });
    }
    Ok(())
}

fn unit(vector: [f64; 3], constraint_id: &str) -> Result<[f64; 3], ConstraintPlanError> {
    let magnitude = magnitude(vector);
    if !magnitude.is_finite() || magnitude <= 1.0e-12 {
        return Err(ConstraintPlanError::DegenerateFrame {
            constraint_id: constraint_id.into(),
        });
    }
    Ok(scale(vector, magnitude.recip()))
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
            "radial",
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
            "pole",
        )
        .unwrap();

        assert_eq!(basis[0], [0.0, 0.0, 1.0]);
        assert_eq!(basis[1], [1.0, 0.0, 0.0]);
        assert_eq!(basis[2], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn degenerate_frame_axes_fail_with_authored_identity() {
        let error = translation_basis(
            &CoordinateFrame::Cylindrical(AxialFrame {
                frame_id: "bad".into(),
                origin: Position::new([0.0; 3]),
                axis: [0.0, 0.0, 1.0],
                radial_reference: [0.0, 0.0, 2.0],
            }),
            [1.0, 0.0, 0.0],
            None,
            "guide",
        )
        .unwrap_err();

        assert_eq!(
            error,
            ConstraintPlanError::DegenerateFrame {
                constraint_id: "guide".into()
            }
        );
    }
}
