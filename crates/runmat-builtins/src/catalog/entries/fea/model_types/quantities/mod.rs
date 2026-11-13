//! Canonical quantities documentation and unchanged public constructor metadata.
mod contract;
mod dynamic_viscosity_pascal_seconds;
mod frequency_hertz;
mod mass_density_kilograms_per_cubic_meter;
mod turbulence_intensity_ratio;
pub use self::dynamic_viscosity_pascal_seconds::DYNAMIC_VISCOSITY_CATALOG_ENTRY;
pub use self::frequency_hertz::FREQUENCY_CATALOG_ENTRY;
pub use self::mass_density_kilograms_per_cubic_meter::MASS_DENSITY_CATALOG_ENTRY;
pub use self::turbulence_intensity_ratio::TURBULENCE_INTENSITY_CATALOG_ENTRY;
#[cfg(test)]
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &TURBULENCE_INTENSITY_CATALOG_ENTRY,
    &MASS_DENSITY_CATALOG_ENTRY,
    &DYNAMIC_VISCOSITY_CATALOG_ENTRY,
    &FREQUENCY_CATALOG_ENTRY,
];
