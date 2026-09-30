//! Prediction across fitted statistical and supported deep-learning models.

mod descriptor;
mod documentation;
mod entry;
mod extensions;
mod inference;
mod integer;

#[cfg(test)]
mod tests;

pub use descriptor::{PREDICT_DESCRIPTOR, PREDICT_ERROR_INTERNAL, PREDICT_ERROR_INVALID_ARGUMENT};
pub use entry::PREDICT_CATALOG_ENTRY;
pub use extensions::{PREDICT_INTEGER_CONTROLS_EXTENSION, PREDICT_INTEGER_PREDICTORS_EXTENSION};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PREDICT_CATALOG_ENTRY];
