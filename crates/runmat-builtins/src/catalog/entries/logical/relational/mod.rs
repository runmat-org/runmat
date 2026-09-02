mod eq;
mod ge;
mod gt;
mod le;
mod lt;
mod ne;
mod ordering_documentation;
mod support;

pub use eq::*;
pub use ge::*;
pub use gt::*;
pub use le::*;
pub use lt::*;
pub use ne::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    eq::ENTRIES,
    ne::ENTRIES,
    lt::ENTRIES,
    le::ENTRIES,
    gt::ENTRIES,
    ge::ENTRIES,
];
