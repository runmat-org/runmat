mod core;

pub(super) fn extend_constants(values: &mut Vec<crate::BuiltinConstantCatalogEntry>) {
    values.extend(core::CONSTANTS.iter().copied());
}
