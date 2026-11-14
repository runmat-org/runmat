//! The fields an analysis step publishes for `output <name> = analysis.<step>.field.<location>.<quantity>`,
//! for a host that offers outputs before the `.fea` exists (Simulation setup's `outputs` item).
//! A reference is admitted by the same rules [`super::lower_outputs`] applies: the quantity and
//! location are known, the location is one a procedure publishes today (no mode-located fields
//! yet, [`crate::semantic::pending::OUTPUT_MODE_LOCATION`]), and the step's physics produces the
//! quantity ([`runmat_analysis_core::family_produces_observation_quantity`]).
use runmat_analysis_core::{AnalysisProcedure, FieldLocation, PhysicsFamily};

use super::{field_location, field_reference_type};

/// The solid-continuum field references a study's output may name, in the order a host offers
/// them. [`step_published_fields`] keeps the ones a given step publishes.
pub const CONTINUUM_FIELD_REFERENCES: &[&str] = &[
    "field.node.displacement",
    "field.mode.modal_frequency",
    "field.element.von_mises_stress",
    "field.element.stress",
    "field.node.reaction_force",
    "field.node.temperature",
    "field.element.heat_flux",
];

/// Why a step can't publish a field reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepFieldRefusal {
    /// The text isn't `field.<location>.<quantity>` with a known location and quantity.
    Unknown,
    /// No procedure publishes fields at this location yet; the text carries the pending marker.
    Pending(String),
    /// The step's physics doesn't produce the quantity.
    NotProduced,
}

/// Whether a step of `family` running `procedure` publishes `reference`
/// (`field.<location>.<quantity>`).
pub fn step_field_admission(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
    reference: &str,
) -> Result<(), StepFieldRefusal> {
    let mut parts = reference.split('.');
    let (Some("field"), Some(location), Some(quantity), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(StepFieldRefusal::Unknown);
    };
    let location = field_location(location).ok_or(StepFieldRefusal::Unknown)?;
    let (quantity_kind, _) = field_reference_type(quantity).ok_or(StepFieldRefusal::Unknown)?;
    if location == FieldLocation::Mode {
        return Err(StepFieldRefusal::Pending(crate::semantic::pending::with_pending(
            format!("no procedure publishes mode-located fields (`field.mode.{quantity}`) yet"),
            Some(crate::semantic::pending::OUTPUT_MODE_LOCATION),
        )));
    }
    if !runmat_analysis_core::family_produces_observation_quantity(family, procedure, quantity_kind) {
        return Err(StepFieldRefusal::NotProduced);
    }
    Ok(())
}

/// The [`CONTINUUM_FIELD_REFERENCES`] a step of `family` running `procedure` publishes today.
pub fn step_published_fields(family: PhysicsFamily, procedure: AnalysisProcedure) -> Vec<&'static str> {
    CONTINUUM_FIELD_REFERENCES
        .iter()
        .copied()
        .filter(|reference| step_field_admission(family, procedure, reference).is_ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::CoupledPhysicsFamily;

    #[test]
    fn a_step_publishes_its_physics_fields_and_no_mode_located_ones_yet() {
        let static_fields = step_published_fields(PhysicsFamily::Structural, AnalysisProcedure::Static);
        assert_eq!(
            static_fields,
            ["field.node.displacement", "field.element.von_mises_stress", "field.element.stress", "field.node.reaction_force"]
        );
        // Prestressed modal offers its mode shapes; the frequencies wait for mode-located fields.
        let modes = step_published_fields(PhysicsFamily::Structural, AnalysisProcedure::PrestressedModalExtraction);
        assert!(modes.contains(&"field.node.displacement"));
        assert!(!modes.contains(&"field.mode.modal_frequency"));
        assert!(matches!(
            step_field_admission(PhysicsFamily::Structural, AnalysisProcedure::PrestressedModalExtraction, "field.mode.modal_frequency"),
            Err(StepFieldRefusal::Pending(text)) if text.contains("[pending:output.mode-location]")
        ));
        assert_eq!(
            step_published_fields(PhysicsFamily::Thermal, AnalysisProcedure::Static),
            ["field.node.temperature", "field.element.heat_flux"]
        );
        let coupled = step_published_fields(PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical), AnalysisProcedure::Static);
        assert!(coupled.contains(&"field.node.temperature") && coupled.contains(&"field.node.displacement"));
        assert_eq!(
            step_field_admission(PhysicsFamily::Thermal, AnalysisProcedure::Static, "field.node.displacement"),
            Err(StepFieldRefusal::NotProduced)
        );
        assert_eq!(
            step_field_admission(PhysicsFamily::Thermal, AnalysisProcedure::Static, "field.node.warmth"),
            Err(StepFieldRefusal::Unknown)
        );
    }
}
