//! Canonical contracts for the supported lightweight XML forms.

mod readstruct;
mod xmlread;
mod xmlwrite;
mod inference;
mod diagnostics;
mod contracts;

pub use contracts::{ReadStructOption, XML_DOCUMENT_CLASS, XML_ELEMENT_CLASS};
pub use diagnostics::{XmlDiagnosticKind, XmlOperation};

pub use readstruct::{READSTRUCT_CATALOG_ENTRY, READSTRUCT_DESCRIPTOR};
pub use xmlread::{XMLREAD_CATALOG_ENTRY, XMLREAD_DESCRIPTOR, XMLREAD_INTEGER_AUDIT};
pub use xmlwrite::{XMLWRITE_CATALOG_ENTRY, XMLWRITE_DESCRIPTOR};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &READSTRUCT_CATALOG_ENTRY,
    &XMLREAD_CATALOG_ENTRY,
    &XMLWRITE_CATALOG_ENTRY,
];
