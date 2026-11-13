//! Catalog identities of the `fea.ObservationQuantity` typed policy class.

use crate::*;

mod acceleration;
mod acoustic_particle_velocity;
mod acoustic_pressure;
mod balance;
mod beam_axial_force;
mod beam_bending_moment;
mod beam_bending_stress;
mod beam_shear_force;
mod beam_torsion_moment;
mod beam_torsion_stress;
mod constraint_violation;
mod contact_gap;
mod contact_pressure;
mod displacement;
mod eigenvalue;
mod electric_charge;
mod electric_current;
mod electric_current_density;
mod electric_field;
mod electric_potential;
mod fluid_pressure;
mod fluid_velocity;
mod heat_flux;
mod heat_rate;
mod magnetic_field_intensity;
mod magnetic_flux_density;
mod magnetic_vector_potential;
mod mass_flow_rate;
mod modal_frequency;
mod modal_participation;
mod reaction_force;
mod reaction_moment;
mod residual;
mod rotation;
mod shell_bending_moment;
mod shell_membrane_force;
mod shell_transverse_shear;
mod shell_von_mises;
mod sound_pressure_level;
mod strain;
mod strain_energy;
mod stress;
mod temperature;
mod temperature_gradient;
mod velocity;
mod von_mises_stress;
mod vorticity;
mod wall_shear_stress;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "An observation quantity names the physical result an observation recovers. The ",
    "quantity fixes the canonical value shape (scalar, three-component vector, symmetric ",
    "tensor, or section-resultant components) and the canonical SI unit of every ",
    "recovered value. A quantity outside the step's physics family fails study ",
    "validation.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the quantity as the second argument of an observation constructor. ",
    "`fea.Observation.field`, `history`, `probe`, and `reduction` take `(id, quantity, ",
    "target, sourceLocation, outputLocation, recovery, component, frame, sign, ",
    "complexRepresentation, sampling, reduction, storage, purpose, cost)`. The same typed",
    " quantity is used by `fea.AccuracyTarget.quantity`, ",
    "`fea.AccuracyTolerance.absolute`, and monitor stop conditions.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Can I pass the quantity as text?",
    answer: concat!(
        "No. Observation and accuracy constructors require the typed value, so a misspelled ",
        "quantity is an error at the call that creates it, not a silent empty result.",
    ),
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &displacement::FEA_OBSERVATION_QUANTITY_DISPLACEMENT_CATALOG_ENTRY,
    &rotation::FEA_OBSERVATION_QUANTITY_ROTATION_CATALOG_ENTRY,
    &velocity::FEA_OBSERVATION_QUANTITY_VELOCITY_CATALOG_ENTRY,
    &acceleration::FEA_OBSERVATION_QUANTITY_ACCELERATION_CATALOG_ENTRY,
    &strain::FEA_OBSERVATION_QUANTITY_STRAIN_CATALOG_ENTRY,
    &stress::FEA_OBSERVATION_QUANTITY_STRESS_CATALOG_ENTRY,
    &von_mises_stress::FEA_OBSERVATION_QUANTITY_VON_MISES_STRESS_CATALOG_ENTRY,
    &reaction_force::FEA_OBSERVATION_QUANTITY_REACTION_FORCE_CATALOG_ENTRY,
    &reaction_moment::FEA_OBSERVATION_QUANTITY_REACTION_MOMENT_CATALOG_ENTRY,
    &strain_energy::FEA_OBSERVATION_QUANTITY_STRAIN_ENERGY_CATALOG_ENTRY,
    &beam_axial_force::FEA_OBSERVATION_QUANTITY_BEAM_AXIAL_FORCE_CATALOG_ENTRY,
    &beam_shear_force::FEA_OBSERVATION_QUANTITY_BEAM_SHEAR_FORCE_CATALOG_ENTRY,
    &beam_torsion_moment::FEA_OBSERVATION_QUANTITY_BEAM_TORSION_MOMENT_CATALOG_ENTRY,
    &beam_bending_moment::FEA_OBSERVATION_QUANTITY_BEAM_BENDING_MOMENT_CATALOG_ENTRY,
    &beam_bending_stress::FEA_OBSERVATION_QUANTITY_BEAM_BENDING_STRESS_CATALOG_ENTRY,
    &beam_torsion_stress::FEA_OBSERVATION_QUANTITY_BEAM_TORSION_STRESS_CATALOG_ENTRY,
    &shell_membrane_force::FEA_OBSERVATION_QUANTITY_SHELL_MEMBRANE_FORCE_CATALOG_ENTRY,
    &shell_bending_moment::FEA_OBSERVATION_QUANTITY_SHELL_BENDING_MOMENT_CATALOG_ENTRY,
    &shell_transverse_shear::FEA_OBSERVATION_QUANTITY_SHELL_TRANSVERSE_SHEAR_CATALOG_ENTRY,
    &shell_von_mises::FEA_OBSERVATION_QUANTITY_SHELL_VON_MISES_CATALOG_ENTRY,
    &temperature::FEA_OBSERVATION_QUANTITY_TEMPERATURE_CATALOG_ENTRY,
    &temperature_gradient::FEA_OBSERVATION_QUANTITY_TEMPERATURE_GRADIENT_CATALOG_ENTRY,
    &heat_flux::FEA_OBSERVATION_QUANTITY_HEAT_FLUX_CATALOG_ENTRY,
    &heat_rate::FEA_OBSERVATION_QUANTITY_HEAT_RATE_CATALOG_ENTRY,
    &electric_potential::FEA_OBSERVATION_QUANTITY_ELECTRIC_POTENTIAL_CATALOG_ENTRY,
    &electric_field::FEA_OBSERVATION_QUANTITY_ELECTRIC_FIELD_CATALOG_ENTRY,
    &electric_current::FEA_OBSERVATION_QUANTITY_ELECTRIC_CURRENT_CATALOG_ENTRY,
    &electric_current_density::FEA_OBSERVATION_QUANTITY_ELECTRIC_CURRENT_DENSITY_CATALOG_ENTRY,
    &electric_charge::FEA_OBSERVATION_QUANTITY_ELECTRIC_CHARGE_CATALOG_ENTRY,
    &magnetic_vector_potential::FEA_OBSERVATION_QUANTITY_MAGNETIC_VECTOR_POTENTIAL_CATALOG_ENTRY,
    &magnetic_flux_density::FEA_OBSERVATION_QUANTITY_MAGNETIC_FLUX_DENSITY_CATALOG_ENTRY,
    &magnetic_field_intensity::FEA_OBSERVATION_QUANTITY_MAGNETIC_FIELD_INTENSITY_CATALOG_ENTRY,
    &acoustic_pressure::FEA_OBSERVATION_QUANTITY_ACOUSTIC_PRESSURE_CATALOG_ENTRY,
    &acoustic_particle_velocity::FEA_OBSERVATION_QUANTITY_ACOUSTIC_PARTICLE_VELOCITY_CATALOG_ENTRY,
    &sound_pressure_level::FEA_OBSERVATION_QUANTITY_SOUND_PRESSURE_LEVEL_CATALOG_ENTRY,
    &fluid_pressure::FEA_OBSERVATION_QUANTITY_FLUID_PRESSURE_CATALOG_ENTRY,
    &fluid_velocity::FEA_OBSERVATION_QUANTITY_FLUID_VELOCITY_CATALOG_ENTRY,
    &vorticity::FEA_OBSERVATION_QUANTITY_VORTICITY_CATALOG_ENTRY,
    &mass_flow_rate::FEA_OBSERVATION_QUANTITY_MASS_FLOW_RATE_CATALOG_ENTRY,
    &wall_shear_stress::FEA_OBSERVATION_QUANTITY_WALL_SHEAR_STRESS_CATALOG_ENTRY,
    &modal_frequency::FEA_OBSERVATION_QUANTITY_MODAL_FREQUENCY_CATALOG_ENTRY,
    &modal_participation::FEA_OBSERVATION_QUANTITY_MODAL_PARTICIPATION_CATALOG_ENTRY,
    &eigenvalue::FEA_OBSERVATION_QUANTITY_EIGENVALUE_CATALOG_ENTRY,
    &residual::FEA_OBSERVATION_QUANTITY_RESIDUAL_CATALOG_ENTRY,
    &balance::FEA_OBSERVATION_QUANTITY_BALANCE_CATALOG_ENTRY,
    &constraint_violation::FEA_OBSERVATION_QUANTITY_CONSTRAINT_VIOLATION_CATALOG_ENTRY,
    &contact_gap::FEA_OBSERVATION_QUANTITY_CONTACT_GAP_CATALOG_ENTRY,
    &contact_pressure::FEA_OBSERVATION_QUANTITY_CONTACT_PRESSURE_CATALOG_ENTRY,
];
