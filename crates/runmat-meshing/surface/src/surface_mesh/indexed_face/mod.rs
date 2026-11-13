//! Lossless surface transport shares identities and quality columns.
//! Original semantic meshes and independent validation remain authoritative.
use super::{ExactSurfaceMeshError, ExactSurfaceMeshErrorKind};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
mod column;
mod decoding;
mod encoding;
mod inventory;
pub(crate) mod partition;
mod quality;
mod rows;
pub(crate) mod surface;
mod value;
use column::{Column, DecodedColumn};
use value::{DigestWords, NodeReference};

fn invalid(reason: impl Into<String>) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidEncoding, reason)
}
fn checkpoint(control: &dyn MeshingCancellationSignal) -> Result<(), ExactSurfaceMeshError> {
    super::control::checkpoint(Some(control))
}
