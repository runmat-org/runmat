//! Shared contract facts, inference, layout, and example fixtures for the
//! solver-meshing builtins. Moved from `runmat-runtime/src/builtins/meshing/metadata.rs`;
//! the runtime binding re-exports these definitions.

mod descriptor;
mod documentation;
mod inference;
mod integer;
mod layout;
#[cfg(test)]
mod tests;

pub use descriptor::*;
pub(super) use documentation::*;
pub(super) use inference::*;
pub use integer::*;
pub(super) use layout::*;

pub use crate::fea_contract::identities::MESHING_MESH_CLASS as MESHING_MESH_OBJECT_CLASS;
pub const MESHING_EVIDENCE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("meshing.Evidence");
