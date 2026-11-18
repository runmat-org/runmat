//! The fields an analysis step publishes for `output <name> = analysis.<step>.field.<location>.<quantity>`.
//! [`step_field_admission`] is the one rule: the checker applies it to every authored field
//! reference ([`super::lower_outputs`]), and Simulation setup's `outputs` item offers exactly the
//! references it admits ([`step_published_fields`]). A reference is admitted when the quantity
//! and location are known, the location is one a procedure publishes today (no mode-located
//! fields yet, [`crate::semantic::pending::OUTPUT_MODE_LOCATION`]), the step's physics produces
//! the quantity ([`runmat_analysis_core::family_produces_observation_quantity`]), and that physics
//! publishes it at the location ([`runmat_analysis_core::published_locations`]).
use runmat_analysis_core::{AnalysisProcedure, FieldLocation, PhysicsFamily};

use super::fields::{location_name, FIELD_LOCATIONS};
use super::{field_location, field_reference_type};

/// The diagnostic code of a field reference at a location its step doesn't publish.
pub const FIELD_LOCATION_CODE: &str = "fea.semantic.output.field_location";

/// Why a step can't publish a field reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepFieldRefusal {
    /// The text isn't `field.<location>.<quantity>` with a known location and quantity.
    Unknown,
    /// No procedure publishes fields at this location yet; the text carries the pending marker.
    Pending(String),
    /// The step's physics doesn't produce the quantity.
    NotProduced,
    /// The step's physics publishes the quantity, at these locations only (`.fea` spellings).
    Location(Vec<&'static str>),
}

impl StepFieldRefusal {
    /// The checker's message for `reference` on step `step`.
    pub fn message(&self, step: &str, reference: &str) -> String {
        match self {
            Self::Unknown => format!("`{reference}` is not a known field reference"),
            Self::Pending(text) => text.clone(),
            Self::NotProduced => {
                let quantity = reference
                    .rsplit('.')
                    .next()
                    .and_then(runmat_analysis_core::ObservationQuantity::from_source_name);
                match quantity {
                    Some(quantity) => {
                        format!("analysis step `{step}` does not produce `{quantity:?}`")
                    }
                    None => format!("analysis step `{step}` does not produce `{reference}`"),
                }
            }
            Self::Location(published) => {
                let quantity = reference.rsplit('.').next().unwrap_or(reference);
                format!(
                    "step `{step}` doesn't publish `{quantity}` at this location; it publishes it at {}",
                    published.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ")
                )
            }
        }
    }
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
        return Err(StepFieldRefusal::Pending(
            crate::semantic::pending::with_pending(
                format!("no procedure publishes mode-located fields (`field.mode.{quantity}`) yet"),
                Some(crate::semantic::pending::OUTPUT_MODE_LOCATION),
            ),
        ));
    }
    if !runmat_analysis_core::family_produces_observation_quantity(family, procedure, quantity_kind)
    {
        return Err(StepFieldRefusal::NotProduced);
    }
    let published = runmat_analysis_core::published_locations(family, quantity_kind);
    if !published.contains(&location) {
        return Err(StepFieldRefusal::Location(
            published.into_iter().filter_map(location_name).collect(),
        ));
    }
    Ok(())
}

/// Every field reference a step of `family` running `procedure` publishes today: each
/// source-language quantity at each location its physics publishes it at, admitted by
/// [`step_field_admission`]. In the source-name table's order, locations in
/// [`FIELD_LOCATIONS`] order; a host that offers them in another order keeps only the order.
pub fn step_published_fields(family: PhysicsFamily, procedure: AnalysisProcedure) -> Vec<String> {
    runmat_analysis_core::SOURCE_QUANTITIES
        .iter()
        .flat_map(|row| {
            let published = runmat_analysis_core::published_locations(family, row.quantity);
            FIELD_LOCATIONS
                .iter()
                .filter(move |(_, location)| published.contains(location))
                .map(move |(location, _)| format!("field.{location}.{}", row.name))
        })
        .filter(|reference| step_field_admission(family, procedure, reference).is_ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::CoupledPhysicsFamily;

    fn fields(family: PhysicsFamily, procedure: AnalysisProcedure) -> Vec<String> {
        step_published_fields(family, procedure)
    }

    #[test]
    fn a_step_publishes_its_physics_fields_and_no_mode_located_ones_yet() {
        let static_fields = fields(PhysicsFamily::Structural, AnalysisProcedure::Static);
        for expected in [
            "field.node.displacement",
            "field.element.von_mises_stress",
            "field.node.von_mises_stress",
            "field.element.stress",
            "field.node.reaction_force",
        ] {
            assert!(
                static_fields.iter().any(|field| field == expected),
                "{expected}: {static_fields:?}"
            );
        }
        assert!(!static_fields
            .iter()
            .any(|field| field == "field.cell.displacement" || field.contains("temperature")));
        // Prestressed modal offers its mode shapes; the frequencies wait for mode-located fields.
        let modes = fields(
            PhysicsFamily::Structural,
            AnalysisProcedure::PrestressedModalExtraction,
        );
        assert!(modes.iter().any(|field| field == "field.node.displacement"));
        assert!(!modes.iter().any(|field| field.starts_with("field.mode.")));
        assert!(matches!(
            step_field_admission(PhysicsFamily::Structural, AnalysisProcedure::PrestressedModalExtraction, "field.mode.modal_frequency"),
            Err(StepFieldRefusal::Pending(text)) if text.contains("[pending:output.mode-location]")
        ));
        let thermal = fields(PhysicsFamily::Thermal, AnalysisProcedure::Static);
        assert!(thermal
            .iter()
            .any(|field| field == "field.node.temperature"));
        assert!(thermal
            .iter()
            .any(|field| field == "field.element.heat_flux"));
        assert!(!thermal.iter().any(|field| field.contains("displacement")));
        let coupled = fields(
            PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical),
            AnalysisProcedure::Static,
        );
        assert!(
            coupled
                .iter()
                .any(|field| field == "field.node.temperature")
                && coupled
                    .iter()
                    .any(|field| field == "field.node.displacement")
        );
        let fluid = fields(PhysicsFamily::Fluid, AnalysisProcedure::Static);
        assert!(
            fluid
                .iter()
                .any(|field| field == "field.cell.fluid_velocity")
                && fluid
                    .iter()
                    .any(|field| field == "field.node.fluid_velocity")
        );
        assert_eq!(
            step_field_admission(
                PhysicsFamily::Thermal,
                AnalysisProcedure::Static,
                "field.node.displacement"
            ),
            Err(StepFieldRefusal::NotProduced)
        );
        assert_eq!(
            step_field_admission(
                PhysicsFamily::Thermal,
                AnalysisProcedure::Static,
                "field.node.warmth"
            ),
            Err(StepFieldRefusal::Unknown)
        );
    }

    #[test]
    fn a_location_the_physics_does_not_publish_is_refused_naming_the_published_ones() {
        let refusal = step_field_admission(
            PhysicsFamily::Structural,
            AnalysisProcedure::Static,
            "field.cell.displacement",
        );
        assert_eq!(refusal, Err(StepFieldRefusal::Location(vec!["node"])));
        let message = refusal
            .unwrap_err()
            .message("load", "field.cell.displacement");
        assert!(message.contains("`node`"), "{message}");
        // Every published reference passes the admission, and every one parses back.
        for family in [
            PhysicsFamily::Structural,
            PhysicsFamily::Thermal,
            PhysicsFamily::Electromagnetic,
            PhysicsFamily::Acoustic,
            PhysicsFamily::Fluid,
        ] {
            for reference in fields(family, AnalysisProcedure::Static) {
                assert_eq!(
                    step_field_admission(family, AnalysisProcedure::Static, &reference),
                    Ok(()),
                    "{reference}"
                );
            }
        }
    }
}
