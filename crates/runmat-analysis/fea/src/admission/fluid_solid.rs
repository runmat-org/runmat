//! Fluid–solid face admission: the one rule that decides what a face joining a fluid cell and a
//! solid element is in a fluid–structure study. `runmat check` applies it to the shared faces of
//! the imported CAD and the run applies it to the conformal faces of the mesh, so both report
//! the same code and the same decision.
//!
//! - A face selected by one declared fluid–structure interface is coupled: the fluid sees the
//!   structure's no-slip wall and the structure carries the fluid's traction.
//! - A face selected by a fluid wall condition (`fluid.no_slip_wall`, `fluid.slip_wall` or
//!   `fluid.moving_wall`) and no interface is a rigid wall: the fluid takes that wall condition,
//!   the wall does not move, and no fluid load enters the solid.
//! - A face selected by neither, by both, or by two interfaces is an admission error.

use runmat_analysis_core::{AuthoredConstraintKind, FluidConstraintKind};

use super::AdmissionIssue;

pub const FLUID_SOLID_FACE_CODE: &str = "fea.admission.fluid_solid_face";

/// What a fluid–solid face is in a fluid–structure study.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidSolidFaceRole {
    /// Selected by a declared fluid–structure interface: two-way coupled.
    Coupled,
    /// Selected by a fluid wall condition only: a rigid wall that loads nothing.
    RigidWall,
}

/// Whether an authored constraint is a fluid wall condition that makes a fluid–solid face rigid.
pub fn is_authored_fluid_wall(kind: &AuthoredConstraintKind) -> bool {
    matches!(
        kind,
        AuthoredConstraintKind::FluidNoSlipWall { .. }
            | AuthoredConstraintKind::FluidSlipWall { .. }
            | AuthoredConstraintKind::FluidMovingWall { .. }
    )
}

/// Whether a realized fluid constraint is a fluid wall condition that makes a fluid–solid face
/// rigid. Same set as [`is_authored_fluid_wall`].
pub fn is_fluid_wall(kind: &FluidConstraintKind) -> bool {
    matches!(
        kind,
        FluidConstraintKind::NoSlipWall { .. }
            | FluidConstraintKind::SlipWall { .. }
            | FluidConstraintKind::MovingWall { .. }
    )
}

/// Classifies one fluid–solid face from the IDs of the declared fluid–structure interfaces and
/// fluid wall conditions that select it. `face` names the face in the message (its CAD names at
/// check, its CAD entity and location at run).
pub fn classify_fluid_solid_face(
    face: &str,
    interfaces: &[&str],
    walls: &[&str],
) -> Result<FluidSolidFaceRole, AdmissionIssue> {
    let issue = |message: String| AdmissionIssue {
        code: FLUID_SOLID_FACE_CODE,
        pending: None,
        message,
    };
    match (interfaces, walls) {
        ([], []) => Err(issue(format!(
            "fluid–solid face {face} has no fluid–structure interface and no fluid wall \
             condition; declare a fluid_structure interface on it to couple the fluid and the \
             solid, or a fluid wall condition (fluid.no_slip_wall, fluid.slip_wall or \
             fluid.moving_wall) to make it a rigid wall that loads nothing"
        ))),
        ([], _) => Ok(FluidSolidFaceRole::RigidWall),
        ([_], []) => Ok(FluidSolidFaceRole::Coupled),
        ([interface], walls) => Err(issue(format!(
            "fluid–solid face {face} is selected by fluid–structure interface `{interface}` and \
             by fluid wall condition{} {}; a coupled face takes its wall condition from the \
             interface, so keep the interface to couple it or the wall condition to make it rigid",
            if walls.len() == 1 { "" } else { "s" },
            quoted(walls)
        ))),
        (interfaces, _) => Err(issue(format!(
            "fluid–solid face {face} is selected by fluid–structure interfaces {}; each face \
             belongs to one interface",
            quoted(interfaces)
        ))),
    }
}

fn quoted(ids: &[&str]) -> String {
    ids.iter()
        .map(|id| format!("`{id}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wall_condition_without_an_interface_is_a_rigid_wall() {
        assert_eq!(
            classify_fluid_solid_face("\"Pipe wall\"", &[], &["walls"]),
            Ok(FluidSolidFaceRole::RigidWall)
        );
    }

    #[test]
    fn a_declared_interface_couples() {
        assert_eq!(
            classify_fluid_solid_face("\"Paddle wetted\"", &["wetted"], &[]),
            Ok(FluidSolidFaceRole::Coupled)
        );
    }

    #[test]
    fn a_face_with_neither_is_an_error_naming_both_remedies() {
        let issue = classify_fluid_solid_face("\"Pipe wall\"", &[], &[]).unwrap_err();
        assert_eq!(issue.code, FLUID_SOLID_FACE_CODE);
        assert!(issue.message.contains("\"Pipe wall\""));
        assert!(issue.message.contains("fluid_structure interface"));
        assert!(issue.message.contains("fluid.no_slip_wall"));
    }

    #[test]
    fn a_face_with_both_or_two_interfaces_is_an_error() {
        assert!(classify_fluid_solid_face("f", &["wetted"], &["walls"]).is_err());
        assert!(classify_fluid_solid_face("f", &["a", "b"], &[]).is_err());
    }
}
