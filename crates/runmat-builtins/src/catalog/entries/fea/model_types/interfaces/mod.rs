//! Canonical interfaces documentation and unchanged public constructor metadata.
mod conjugate_heat_transfer_interface_conductance;
mod conjugate_heat_transfer_interface_perfect;
mod conjugate_heat_transfer_interface_resistance;
mod contact_interface_coulomb;
mod contact_interface_frictionless;
mod contract;
mod fixture;
mod fluid_structure_interface_no_slip;
pub use self::conjugate_heat_transfer_interface_conductance::CHT_CONDUCTANCE_CATALOG_ENTRY;
pub use self::conjugate_heat_transfer_interface_perfect::CHT_PERFECT_CATALOG_ENTRY;
pub use self::conjugate_heat_transfer_interface_resistance::CHT_RESISTANCE_CATALOG_ENTRY;
pub use self::contact_interface_coulomb::CONTACT_COULOMB_CATALOG_ENTRY;
pub use self::contact_interface_frictionless::CONTACT_FRICTIONLESS_CATALOG_ENTRY;
pub use self::fluid_structure_interface_no_slip::FLUID_STRUCTURE_INTERFACE_CATALOG_ENTRY;
#[cfg(test)]
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CONTACT_COULOMB_CATALOG_ENTRY,
    &CONTACT_FRICTIONLESS_CATALOG_ENTRY,
    &CHT_RESISTANCE_CATALOG_ENTRY,
    &CHT_CONDUCTANCE_CATALOG_ENTRY,
    &FLUID_STRUCTURE_INTERFACE_CATALOG_ENTRY,
    &CHT_PERFECT_CATALOG_ENTRY,
];
