mod descriptor;
mod documentation;
mod entry;
mod fields;
mod inference;
#[cfg(test)]
mod tests;
pub use descriptor::*;
pub use entry::INTEGER_CONSTRAINT_CATALOG_ENTRY;
pub use fields::IntegerConstraintField;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&INTEGER_CONSTRAINT_CATALOG_ENTRY];
