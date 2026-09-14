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

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(eq::ENTRIES.iter().copied());
    values.extend(ne::ENTRIES.iter().copied());
    values.extend(lt::ENTRIES.iter().copied());
    values.extend(le::ENTRIES.iter().copied());
    values.extend(gt::ENTRIES.iter().copied());
    values.extend(ge::ENTRIES.iter().copied());
}
