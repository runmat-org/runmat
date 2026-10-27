//! Canonical domains documentation and unchanged public constructor metadata.
mod contract;
mod electro_thermal_domain_at_reference_temperature;
mod electromagnetic_domain_at_frequency;
mod fluid_domain_newtonian;
mod thermo_mechanical_domain_at_reference_temperature;
pub use self::electro_thermal_domain_at_reference_temperature::ELECTROTHERMAL_DOMAIN_CATALOG_ENTRY;
pub use self::electromagnetic_domain_at_frequency::ELECTROMAGNETIC_DOMAIN_CATALOG_ENTRY;
pub use self::fluid_domain_newtonian::FLUID_DOMAIN_CATALOG_ENTRY;
pub use self::thermo_mechanical_domain_at_reference_temperature::THERMOMECHANICAL_DOMAIN_CATALOG_ENTRY;
#[cfg(test)]
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &ELECTROTHERMAL_DOMAIN_CATALOG_ENTRY,
    &THERMOMECHANICAL_DOMAIN_CATALOG_ENTRY,
    &FLUID_DOMAIN_CATALOG_ENTRY,
    &ELECTROMAGNETIC_DOMAIN_CATALOG_ENTRY,
];
