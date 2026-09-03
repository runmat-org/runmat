mod and;
mod not;
mod or;
mod support;
mod xor;

#[cfg(test)]
mod tests;

pub use and::*;
pub use not::*;
pub use or::*;
pub use xor::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    &[and::ENTRIES, or::ENTRIES, xor::ENTRIES, not::ENTRIES];
