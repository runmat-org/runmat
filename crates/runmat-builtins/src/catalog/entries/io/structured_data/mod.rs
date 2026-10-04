//! Canonical contracts for the supported lightweight XML forms.

mod readstruct;
mod xmlread;
mod xmlwrite;
mod inference;

pub use readstruct::{READSTRUCT_CATALOG_ENTRY, READSTRUCT_DESCRIPTOR};
pub use xmlread::{XMLREAD_CATALOG_ENTRY, XMLREAD_DESCRIPTOR, XMLREAD_INTEGER_AUDIT};
pub use xmlwrite::{XMLWRITE_CATALOG_ENTRY, XMLWRITE_DESCRIPTOR};

pub const XML_DOCUMENT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("org.w3c.dom.Document");
pub const XML_ELEMENT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("org.w3c.dom.Element");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadStructOption {
    FileType,
    AttributeSuffix,
    TextNodeName,
}

impl ReadStructOption {
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("FileType") {
            Some(Self::FileType)
        } else if name.eq_ignore_ascii_case("AttributeSuffix") {
            Some(Self::AttributeSuffix)
        } else if name.eq_ignore_ascii_case("TextNodeName") {
            Some(Self::TextNodeName)
        } else {
            None
        }
    }
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &READSTRUCT_CATALOG_ENTRY,
    &XMLREAD_CATALOG_ENTRY,
    &XMLWRITE_CATALOG_ENTRY,
];
