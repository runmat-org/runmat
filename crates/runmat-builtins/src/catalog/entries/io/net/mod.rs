//! TCP socket construction and stream I/O identities.

mod datatype;
mod read;
mod readline;
mod tcpclient;
mod write;

pub use datatype::NetDataType;
pub use read::*;
pub use readline::*;
pub use tcpclient::*;
pub use write::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &READ_CATALOG_ENTRY,
    &READLINE_CATALOG_ENTRY,
    &TCPCLIENT_CATALOG_ENTRY,
    &WRITE_CATALOG_ENTRY,
];
