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
mod descriptor;
mod documentation;
mod inference;
mod integer;
mod dataset;

pub use super::data_family::{DATA_ARRAY_CLASS, DATA_TRANSACTION_CLASS};
pub use dataset::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = dataset::ENTRIES;
