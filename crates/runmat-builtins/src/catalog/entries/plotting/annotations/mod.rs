use crate::BuiltinCatalogEntry;
mod shared;
mod sgtitle;
mod subtitle;
mod suptitle;
mod text;
mod title;
mod xlabel;
mod xline;
mod ylabel;
mod yline;
mod zlabel;
pub use sgtitle::*;
pub use subtitle::*;
pub use suptitle::*;
pub use text::*;
pub use title::*;
pub use xlabel::*;
pub use xline::*;
pub use ylabel::*;
pub use yline::*;
pub use zlabel::*;
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &sgtitle::SGTITLE_CATALOG_ENTRY,
    &subtitle::SUBTITLE_CATALOG_ENTRY,
    &suptitle::SUPTITLE_CATALOG_ENTRY,
    &text::TEXT_CATALOG_ENTRY,
    &title::TITLE_CATALOG_ENTRY,
    &xlabel::XLABEL_CATALOG_ENTRY,
    &xline::XLINE_CATALOG_ENTRY,
    &ylabel::YLABEL_CATALOG_ENTRY,
    &yline::YLINE_CATALOG_ENTRY,
    &zlabel::ZLABEL_CATALOG_ENTRY,
];
