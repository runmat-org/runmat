mod conversion;
mod im2double;
mod im2uint16;
mod im2uint8;

pub use conversion::ImageIndexMode;
pub use im2double::*;
pub use im2uint16::*;
pub use im2uint8::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &IM2DOUBLE_CATALOG_ENTRY,
    &IM2UINT16_CATALOG_ENTRY,
    &IM2UINT8_CATALOG_ENTRY,
];
