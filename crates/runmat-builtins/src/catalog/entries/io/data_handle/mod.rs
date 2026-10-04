//! Dataset, DataArray, and DataTransaction receiver contracts.

mod contract;
mod data_array;
mod data_transaction;
mod descriptor;
mod documentation;
mod inference;
mod integer;
mod transaction_inference;
mod transaction_integer;
mod dataset;

pub use data_array::*;
pub use data_transaction::*;
pub use dataset::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(data_array::ENTRIES.iter().copied());
    let (commit_alias, transaction_methods) = data_transaction::ENTRIES
        .split_last()
        .expect("transaction identities include the commit alias");
    values.extend(transaction_methods.iter().copied());
    values.extend(dataset::ENTRIES.iter().copied());
    values.push(*commit_alias);
}

#[cfg(test)]
mod tests {
    #[test]
    fn all_thirty_five_handle_identities_are_unique_and_ordered() {
        let mut entries = Vec::new();
        super::extend_entries(&mut entries);
        assert_eq!(entries.len(), 35);
        let names: Vec<_> = entries.iter().map(|entry| entry.identity.name).collect();
        assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
        let errors = crate::validate_builtin_catalog(&entries);
        assert!(errors.is_empty(), "{errors:#?}");
    }
}
