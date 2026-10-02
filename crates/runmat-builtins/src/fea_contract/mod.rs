//! Single typed descriptor and integer boundary authority for the FEA family.
//! Runtime entrypoints re-export these facts while catalog identities move in
//! bounded slices; no constructor or solver semantics live here.

pub mod descriptors;
pub mod catalog_entry;
pub mod constructor_inference;
pub use constructor_inference as inference;
pub mod integer;
pub mod operation_entry;
pub mod operation_inference;
pub mod parameters;
