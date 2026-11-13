mod binary;
mod bitcmp;
mod bitget;
mod bitset;
mod bitshift;
mod support;
mod swapbytes;

pub use binary::*;
pub use bitcmp::*;
pub use bitget::*;
pub use bitset::*;
pub use bitshift::*;
pub use support::*;
pub use swapbytes::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    binary::extend_entries(values);
    bitcmp::extend_entries(values);
    bitget::extend_entries(values);
    bitset::extend_entries(values);
    bitshift::extend_entries(values);
    swapbytes::extend_entries(values);
}
