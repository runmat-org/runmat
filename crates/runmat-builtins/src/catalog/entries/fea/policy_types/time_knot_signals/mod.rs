//! Catalog identities of the `fea.*Signal.fromTimeKnots` time-history constructors.

use crate::*;

mod acceleration;
mod angular_acceleration;
mod angular_velocity;
mod body_force;
mod charge_density;
mod current_density;
mod displacement;
mod electric_charge;
mod electric_current;
mod electric_potential;
mod force;
mod heat_flux;
mod heat_generation;
mod magnetic_vector_potential;
mod mass_flow;
mod moment;
mod pressure;
mod rotation;
mod temperature;
mod temperature_difference;
mod traction;
mod velocity;
mod volume_flow;
#[cfg(test)]
mod tests;

pub(super) const HISTORY_BEHAVIOR: &str = concat!(
    "The knots must be a row or column cell array of at least two `fea.SignalKnot` values",
    " with strictly increasing times. The first and last knots must be continuous; jumps ",
    "are allowed only in between. Every knot must have the ordinate shape of the signal's",
    " quantity, or the call fails with `RunMat:fea:InvalidInput` before a signal is ",
    "returned. The id is stored as written.",
);
pub(super) const HISTORY_SAMPLING: &str = concat!(
    "Between knots the value follows the typed `fea.SignalInterpolation`: `linear` ",
    "interpolates between the outgoing value of the earlier knot and the incoming value ",
    "of the later knot, `leftEndpoint` holds the earlier knot's outgoing value (a ",
    "zero-order hold), and `rightEndpoint` uses the later knot's incoming value. Outside ",
    "the first and last knot times the typed `fea.SignalExtrapolation` applies: `reject` ",
    "makes sampling there an error, `zero` returns zero, and `holdEndpoint` returns the ",
    "value at the nearest end knot.",
);
pub(super) const HISTORY_RESULT: &str = concat!(
    "The result is an immutable, authored, tabulated time-domain signal stored in ",
    "binary64. Its `signal` property holds one field named for the quantity, with ",
    "`signal_id`, `shape`, `representation.tabulated` (domain `time_seconds`, knots, ",
    "interpolation, extrapolation), `precision`, `provenance`, and `evidence` (the knot ",
    "count).",
);
pub(super) const HISTORY_STUDY: &str = concat!(
    "Register the signal with `fea.model(..., \"Signals\", {signal})` and reference it by ",
    "its typed value from the excitation, initial condition, or motion that consumes it. ",
    "The same signal can drive several model objects and steps.",
);
pub(super) const HISTORY_FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Do the knot values carry units?",
    answer: concat!(
        "They use the canonical SI unit of the signal class. The constructor does not convert",
        " units; write values in SI before creating the knots.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &displacement::FEA_DISPLACEMENT_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &rotation::FEA_ROTATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &velocity::FEA_VELOCITY_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &acceleration::FEA_ACCELERATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &angular_velocity::FEA_ANGULAR_VELOCITY_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &angular_acceleration::FEA_ANGULAR_ACCELERATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &force::FEA_FORCE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &moment::FEA_MOMENT_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &traction::FEA_TRACTION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &pressure::FEA_PRESSURE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &body_force::FEA_BODY_FORCE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &temperature::FEA_TEMPERATURE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &temperature_difference::FEA_TEMPERATURE_DIFFERENCE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &heat_flux::FEA_HEAT_FLUX_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &heat_generation::FEA_HEAT_GENERATION_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &electric_potential::FEA_ELECTRIC_POTENTIAL_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &electric_current::FEA_ELECTRIC_CURRENT_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &electric_charge::FEA_ELECTRIC_CHARGE_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &current_density::FEA_CURRENT_DENSITY_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &charge_density::FEA_CHARGE_DENSITY_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &magnetic_vector_potential::FEA_MAGNETIC_VECTOR_POTENTIAL_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &mass_flow::FEA_MASS_FLOW_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
    &volume_flow::FEA_VOLUME_FLOW_SIGNAL_FROM_TIME_KNOTS_CATALOG_ENTRY,
];
