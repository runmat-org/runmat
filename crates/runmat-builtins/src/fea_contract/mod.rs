//! Single typed descriptor and integer boundary authority for the FEA family.
//! Runtime entrypoints re-export these facts while catalog identities move in
//! bounded slices; no constructor or solver semantics live here.

pub mod descriptors;
pub mod example_fixtures;
pub mod identities;
pub mod catalog_entry;
pub mod constructor_inference;
pub mod integer;
pub mod operation_entry;
pub mod operation_inference;
pub mod options;
pub mod parameters;
pub mod result_inference;
pub mod results_option;
