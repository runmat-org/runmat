use crate::*;

mod constructor;
mod descriptor;
mod extensions;
mod inference;
mod integer;
mod is_key;
mod keys;
mod remove;
mod subsasgn;
mod subsref;
mod values;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub use extensions::*;
pub use integer::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &constructor::ENTRY,
    &keys::ENTRY,
    &values::ENTRY,
    &is_key::ENTRY,
    &remove::ENTRY,
    &subsref::ENTRY,
    &subsasgn::ENTRY,
];
