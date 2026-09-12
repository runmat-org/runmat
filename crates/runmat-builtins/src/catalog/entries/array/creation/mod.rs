mod constants;
mod full;
mod zeros;

pub use full::*;
pub use zeros::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(full::ENTRIES.iter().copied());
    values.extend(zeros::ENTRIES.iter().copied());
}

pub(super) fn extend_constants(values: &mut Vec<crate::BuiltinConstantCatalogEntry>) {
    values.extend(constants::CONSTANTS.iter().copied());
}
