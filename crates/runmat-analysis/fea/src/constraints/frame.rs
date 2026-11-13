use runmat_analysis_core::{CoordinateFrame, StructuralConstraint, StructuralTranslationDof};

use crate::physics::structural::frame::{self, StructuralFrameError};

use super::{CanonicalStructuralDof, ConstraintPlanError};

pub(super) fn require_global(constraint: &StructuralConstraint) -> Result<(), ConstraintPlanError> {
    if constraint.frame != CoordinateFrame::Global {
        return Err(ConstraintPlanError::UnsupportedFrame {
            constraint_id: constraint.constraint_id.clone(),
            frame: constraint.frame.context_label(),
        });
    }
    Ok(())
}

pub(super) fn require_normal_capable(
    constraint: &StructuralConstraint,
) -> Result<(), ConstraintPlanError> {
    if !matches!(
        constraint.frame,
        CoordinateFrame::Global | CoordinateFrame::BoundaryNormalTangent
    ) {
        return Err(ConstraintPlanError::UnsupportedFrame {
            constraint_id: constraint.constraint_id.clone(),
            frame: constraint.frame.context_label(),
        });
    }
    Ok(())
}

pub(super) fn translation_direction(dof: CanonicalStructuralDof, basis: [[f64; 3]; 3]) -> [f64; 3] {
    match dof {
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => basis[0],
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => basis[1],
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => basis[2],
        CanonicalStructuralDof::Rotation(_) => {
            unreachable!("translation values contain no rotational DOF")
        }
    }
}

pub(super) fn translation_basis(
    frame: &CoordinateFrame,
    point: [f64; 3],
    boundary_normal: Option<[f64; 3]>,
    constraint_id: &str,
) -> Result<[[f64; 3]; 3], ConstraintPlanError> {
    frame::translation_basis(frame, point, boundary_normal)
        .map_err(|error| map_frame_error(error, frame, constraint_id))
}

pub(super) fn boundary_tangent_basis(
    normal: [f64; 3],
    constraint_id: &str,
) -> Result<[[f64; 3]; 2], ConstraintPlanError> {
    frame::boundary_tangent_basis(normal).map_err(|error| {
        map_frame_error(
            error,
            &CoordinateFrame::BoundaryNormalTangent,
            constraint_id,
        )
    })
}

fn map_frame_error(
    error: StructuralFrameError,
    frame: &CoordinateFrame,
    constraint_id: &str,
) -> ConstraintPlanError {
    match error {
        StructuralFrameError::Degenerate => ConstraintPlanError::DegenerateFrame {
            constraint_id: constraint_id.into(),
        },
        StructuralFrameError::BoundaryNormalUnavailable => ConstraintPlanError::NormalUnavailable {
            constraint_id: constraint_id.into(),
        },
        StructuralFrameError::Unsupported => ConstraintPlanError::UnsupportedFrame {
            constraint_id: constraint_id.into(),
            frame: frame.context_label(),
        },
    }
}
