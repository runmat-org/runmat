//! Structural selectors and dimensioned prescribed-value pairs.
mod contract;
mod ux;
pub use self::ux::FEA_STRUCTURAL_DOF_UX_CATALOG_ENTRY;
mod uy;
pub use self::uy::FEA_STRUCTURAL_DOF_UY_CATALOG_ENTRY;
mod uz;
pub use self::uz::FEA_STRUCTURAL_DOF_UZ_CATALOG_ENTRY;
mod rx;
pub use self::rx::FEA_STRUCTURAL_DOF_RX_CATALOG_ENTRY;
mod ry;
pub use self::ry::FEA_STRUCTURAL_DOF_RY_CATALOG_ENTRY;
mod rz;
pub use self::rz::FEA_STRUCTURAL_DOF_RZ_CATALOG_ENTRY;
mod translation_value;
pub use self::translation_value::FEA_STRUCTURAL_DOF_VALUE_TRANSLATION_CATALOG_ENTRY;
mod rotation_value;
pub use self::rotation_value::FEA_STRUCTURAL_DOF_VALUE_ROTATION_CATALOG_ENTRY;
#[cfg(test)]
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &FEA_STRUCTURAL_DOF_UX_CATALOG_ENTRY,
    &FEA_STRUCTURAL_DOF_UY_CATALOG_ENTRY,
    &FEA_STRUCTURAL_DOF_UZ_CATALOG_ENTRY,
    &FEA_STRUCTURAL_DOF_RX_CATALOG_ENTRY,
    &FEA_STRUCTURAL_DOF_RY_CATALOG_ENTRY,
    &FEA_STRUCTURAL_DOF_RZ_CATALOG_ENTRY,
    &FEA_STRUCTURAL_DOF_VALUE_TRANSLATION_CATALOG_ENTRY,
    &FEA_STRUCTURAL_DOF_VALUE_ROTATION_CATALOG_ENTRY,
];
