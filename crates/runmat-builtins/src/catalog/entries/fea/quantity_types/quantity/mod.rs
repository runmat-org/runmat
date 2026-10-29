//! Dimensioned FEA values (`fea.Displacement.meters`, `fea.Temperature.celsius`,
//! `fea.Position.millimeters`, ...): one pure constructor per class and input
//! unit, generated from a unit table; prose is per constructor.

use crate::fea_contract::doc_links;
use crate::*;

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why use typed values instead of plain numbers?", answer: "The class carries the physical dimension and the constructor name carries the input unit, so a millimeter value cannot be mistaken for meters and a temperature cannot be passed where a pressure is expected. All values are stored in SI." },
    BuiltinDocumentationFaq { question: "Can I pass integer inputs?", answer: "Yes. Any integer class is accepted and converted once to double precision; very large 64-bit integers can round in that conversion." },
];
pub(super) const LINKS: &[BuiltinDocumentationLink] =
    &[doc_links::TYPES_AND_UNITS, doc_links::MODELS];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[doc_links::FEA_VALUES_SOURCE];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] = &[
    doc_links::QUANTITY_SI_TEST,
    doc_links::QUANTITY_RANGE_TEST,
    doc_links::VECTOR_QUANTITY_TEST,
];

mod complex_phase_degrees;
mod complex_phase_radians;
mod displacement_meters;
mod displacement_micrometers;
mod displacement_millimeters;
mod dynamic_viscosity_pascal_seconds;
mod electric_current_amperes;
mod electric_potential_volts;
mod frequency_hertz;
mod friction_coefficient_ratio;
mod heat_flux_watts_per_square_meter;
mod heat_transfer_coefficient_watts_per_square_meter_kelvin;
mod length_meters;
mod length_micrometers;
mod length_millimeters;
mod magnetic_vector_potential_webers_per_meter;
mod mass_density_kilograms_per_cubic_meter;
mod position_meters;
mod position_millimeters;
mod pressure_pascals;
mod rayleigh_mass_coefficient_per_second;
mod rayleigh_stiffness_coefficient_seconds;
mod rotation_degrees;
mod rotation_radians;
mod specific_acoustic_impedance_pascal_seconds_per_meter;
mod temperature_celsius;
mod temperature_difference_celsius;
mod temperature_difference_kelvin;
mod temperature_kelvin;
mod thermal_contact_resistance_square_meter_kelvin_per_watt;
mod time_seconds;
mod translation_meters;
mod translation_millimeters;
mod turbulence_intensity_ratio;
mod velocity_meters_per_second;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &displacement_meters::ENTRY,
    &displacement_millimeters::ENTRY,
    &displacement_micrometers::ENTRY,
    &rotation_radians::ENTRY,
    &rotation_degrees::ENTRY,
    &temperature_kelvin::ENTRY,
    &temperature_celsius::ENTRY,
    &temperature_difference_kelvin::ENTRY,
    &temperature_difference_celsius::ENTRY,
    &pressure_pascals::ENTRY,
    &velocity_meters_per_second::ENTRY,
    &heat_flux_watts_per_square_meter::ENTRY,
    &heat_transfer_coefficient_watts_per_square_meter_kelvin::ENTRY,
    &specific_acoustic_impedance_pascal_seconds_per_meter::ENTRY,
    &electric_potential_volts::ENTRY,
    &electric_current_amperes::ENTRY,
    &magnetic_vector_potential_webers_per_meter::ENTRY,
    &length_meters::ENTRY,
    &length_millimeters::ENTRY,
    &length_micrometers::ENTRY,
    &complex_phase_radians::ENTRY,
    &complex_phase_degrees::ENTRY,
    &thermal_contact_resistance_square_meter_kelvin_per_watt::ENTRY,
    &friction_coefficient_ratio::ENTRY,
    &frequency_hertz::ENTRY,
    &time_seconds::ENTRY,
    &mass_density_kilograms_per_cubic_meter::ENTRY,
    &dynamic_viscosity_pascal_seconds::ENTRY,
    &turbulence_intensity_ratio::ENTRY,
    &rayleigh_mass_coefficient_per_second::ENTRY,
    &rayleigh_stiffness_coefficient_seconds::ENTRY,
    &position_meters::ENTRY,
    &position_millimeters::ENTRY,
    &translation_meters::ENTRY,
    &translation_millimeters::ENTRY,
];
