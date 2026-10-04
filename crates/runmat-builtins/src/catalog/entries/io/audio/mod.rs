//! Typed catalog authority for host audio metadata and sample import.

pub mod audioinfo;
pub mod audioread;
mod contract;
mod datatype;
mod decode_facts;
mod fixtures;

pub use audioinfo::AUDIOINFO_CATALOG_ENTRY;
pub use audioread::AUDIOREAD_CATALOG_ENTRY;
pub use datatype::AudioReadDataType;
pub use decode_facts::{AudioDecodeResourceFacts, AudioSampleCoding};

pub use contract::{
    AUDIOINFO_DESCRIPTOR, AUDIOINFO_ERROR_ARGUMENT, AUDIOINFO_ERROR_FORMAT, AUDIOINFO_ERROR_IO,
    AUDIOREAD_DESCRIPTOR, AUDIOREAD_ERROR_ARGUMENT, AUDIOREAD_ERROR_FORMAT, AUDIOREAD_ERROR_IO,
    AUDIOREAD_INTEGER_CAPABILITIES,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &audioinfo::AUDIOINFO_CATALOG_ENTRY,
    &audioread::AUDIOREAD_CATALOG_ENTRY,
];
