//! Canonical fluid input capabilities shared by portable semantic admission and preparation.
//! Physical target, signal and projection validation remains with each operator owner.
use runmat_analysis_core::{ExcitationAction, PhysicalSignal};

pub(crate) fn supports_action(action: &ExcitationAction) -> bool {
    matches!(
        action,
        ExcitationAction::BodyForce(_)
            | ExcitationAction::MassFlow(_)
            | ExcitationAction::InletVelocity(_)
            | ExcitationAction::WallMotion(_)
    )
}

pub(crate) fn supports_signal(signal: &PhysicalSignal) -> bool {
    matches!(
        signal,
        PhysicalSignal::Velocity(_)
            | PhysicalSignal::Pressure(_)
            | PhysicalSignal::BodyForceDensity(_)
            | PhysicalSignal::MassFlowRate(_)
    )
}

/// Potential report metadata across an entire schedule, without realizing samples.
/// Includes gauge constraints conservatively; original source IDs remain borrowed.
pub(crate) fn potential_boundary_authorities<'a>(
    constraints: &'a [runmat_analysis_core::FluidConstraint],
    operators: &'a [runmat_analysis_core::AnalysisBoundary],
    actions: &'a [ExcitationAction],
) -> impl Iterator<Item = &'a str> + Clone {
    constraints
        .iter()
        .map(|c| c.constraint_id.as_str())
        .chain(operators.iter().map(|b| b.boundary_id.as_str()))
        .chain(actions.iter().filter_map(|a| match a {
            ExcitationAction::MassFlow(a)
            | ExcitationAction::InletVelocity(a)
            | ExcitationAction::WallMotion(a) => Some(a.excitation_id.0.as_str()),
            _ => None,
        }))
}
