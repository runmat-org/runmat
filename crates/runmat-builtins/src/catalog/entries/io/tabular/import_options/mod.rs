mod shared;
mod setvaropts;
mod setvartype;
mod getvaropts;

pub use getvaropts::{GETVAROPTS_CATALOG_ENTRY, GETVAROPTS_DESCRIPTOR};
pub use setvaropts::{SETVAROPTS_CATALOG_ENTRY, SETVAROPTS_DESCRIPTOR};
pub use setvartype::{SETVARTYPE_CATALOG_ENTRY, SETVARTYPE_DESCRIPTOR};

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(setvaropts::ENTRIES.iter().copied());
    values.extend(setvartype::ENTRIES.iter().copied());
    values.extend(getvaropts::ENTRIES.iter().copied());
}
