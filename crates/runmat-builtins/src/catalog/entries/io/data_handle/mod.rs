//! Dataset, DataArray, and DataTransaction receiver contracts.

macro_rules! receiver_descriptor {
    ($name:ident, $label:literal, $inputs:expr, $outputs:expr) => {
        pub const $name: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &[crate::BuiltinSignatureDescriptor {
                label: $label,
                inputs: $inputs,
                outputs: $outputs,
            }],
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: super::super::descriptor::ERRORS,
        };
    };
}

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

pub use super::data_family::{DATA_ARRAY_CLASS, DATA_TRANSACTION_CLASS};
pub use data_array::*;
pub use data_transaction::*;
pub use dataset::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(data_array::ENTRIES.iter().copied());
    values.extend(data_transaction::ENTRIES.iter().copied());
    values.extend(dataset::ENTRIES.iter().copied());
}
