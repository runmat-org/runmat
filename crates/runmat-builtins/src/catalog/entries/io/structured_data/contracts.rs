//! Shared, typed XML class and readstruct option identities.

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
