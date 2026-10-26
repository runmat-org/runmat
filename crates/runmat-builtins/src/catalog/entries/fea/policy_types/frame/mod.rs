//! Coordinate-frame policies and explicit axis-frame constructors.
mod contract;
mod global_frame;
pub use self::global_frame::FEA_COORDINATE_FRAME_GLOBAL_CATALOG_ENTRY;
mod material_frame;
pub use self::material_frame::FEA_COORDINATE_FRAME_MATERIAL_CATALOG_ENTRY;
mod element_frame;
pub use self::element_frame::FEA_COORDINATE_FRAME_ELEMENT_CATALOG_ENTRY;
mod boundary_normal_tangent;
pub use self::boundary_normal_tangent::FEA_COORDINATE_FRAME_BOUNDARY_CATALOG_ENTRY;
mod cartesian;
pub use self::cartesian::FEA_COORDINATE_FRAME_CARTESIAN_CATALOG_ENTRY;
mod cylindrical;
pub use self::cylindrical::FEA_COORDINATE_FRAME_CYLINDRICAL_CATALOG_ENTRY;
mod spherical;
pub use self::spherical::FEA_COORDINATE_FRAME_SPHERICAL_CATALOG_ENTRY;
#[cfg(test)]
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &FEA_COORDINATE_FRAME_GLOBAL_CATALOG_ENTRY,
    &FEA_COORDINATE_FRAME_MATERIAL_CATALOG_ENTRY,
    &FEA_COORDINATE_FRAME_ELEMENT_CATALOG_ENTRY,
    &FEA_COORDINATE_FRAME_BOUNDARY_CATALOG_ENTRY,
    &FEA_COORDINATE_FRAME_CARTESIAN_CATALOG_ENTRY,
    &FEA_COORDINATE_FRAME_CYLINDRICAL_CATALOG_ENTRY,
    &FEA_COORDINATE_FRAME_SPHERICAL_CATALOG_ENTRY,
];
