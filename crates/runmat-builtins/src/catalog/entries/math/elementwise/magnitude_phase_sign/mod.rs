mod abs;
mod angle;
mod sign;

#[cfg(test)]
mod tests;

pub use abs::*;
pub use angle::*;
pub use sign::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &ABS_CATALOG_ENTRY,
    &ANGLE_CATALOG_ENTRY,
    &SIGN_CATALOG_ENTRY,
];
