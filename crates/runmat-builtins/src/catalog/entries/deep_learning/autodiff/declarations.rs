//! Canonical differentiable-operation registration order.
use super::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DLARRAY_MINUS_CATALOG_ENTRY,
    &DLARRAY_MTIMES_CATALOG_ENTRY,
    &DLARRAY_PLUS_CATALOG_ENTRY,
    &DLARRAY_RDIVIDE_CATALOG_ENTRY,
    &DLARRAY_SUM_CATALOG_ENTRY,
    &DLARRAY_TIMES_CATALOG_ENTRY,
    &DLGRADIENT_CATALOG_ENTRY,
];

#[cfg(test)]
mod tests {
    #[test]
    fn autodiff_parent_aggregates_the_original_identity_order() {
        let mut entries = Vec::new();
        super::super::extend_entries(&mut entries);
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.identity.name)
                .collect::<Vec<_>>(),
            [
                "dlarray.minus",
                "dlarray.mtimes",
                "dlarray.plus",
                "dlarray.rdivide",
                "dlarray.sum",
                "dlarray.times",
                "dlgradient",
            ]
        );
    }
}
