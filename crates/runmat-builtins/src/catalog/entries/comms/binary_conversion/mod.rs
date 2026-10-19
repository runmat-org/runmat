mod bi2de;
mod bin2dec;
mod dec2bin;
mod dec2hex;
mod de2bi;
mod hex2dec;
mod inference;
mod text_inference;

pub use bi2de::*;
pub use bin2dec::*;
pub use de2bi::*;
pub use dec2bin::*;
pub use dec2hex::*;
pub use hex2dec::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &BI2DE_CATALOG_ENTRY,
    &DE2BI_CATALOG_ENTRY,
    &DEC2BIN_CATALOG_ENTRY,
    &DEC2HEX_CATALOG_ENTRY,
    &BIN2DEC_CATALOG_ENTRY,
    &HEX2DEC_CATALOG_ENTRY,
];
