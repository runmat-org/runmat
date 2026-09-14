mod binornd;
mod gamrnd;
mod wblrnd;

pub use binornd::*;
pub use gamrnd::*;
pub use wblrnd::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(binornd::ENTRIES.iter().copied());
    values.extend(gamrnd::ENTRIES.iter().copied());
    values.extend(wblrnd::ENTRIES.iter().copied());
}
