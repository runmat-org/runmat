//! Shared conservative owned-field metadata policy, excluding numerical value payloads.
//!
//! Covers typed field/container headers, identifier strings, shapes and conversion scratch for
//! generated field identifiers. This is a planner estimate, not an allocator or arbitrary-string
//! bound; producer model/source evidence and serialized transport must be reserved separately.

pub(super) const FIELD_METADATA_BYTES: u64 = 4 * 1024;
