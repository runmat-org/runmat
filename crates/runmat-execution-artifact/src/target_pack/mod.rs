//! Portable selected-target runtime pack metadata and authenticated contents.
//! Neither successful decoding nor byte verification admits machine-code
//! execution, qualifies a numerical backend, or probes the build host's ABI.
mod codec;
mod components;
mod manifest;
mod profile;

pub use codec::{TargetPackId, VerifiedTargetPackContents};
pub use components::{
    ComponentId, NativeBuildInput, NativeBuildInputs, NativeLinkInput, NativeLinkRetention,
    TargetPackComponent, TargetPackComponentKind,
};
pub use manifest::{
    TargetPackManifest, TargetPackService, TargetPackServiceBehavior, TargetPackServiceKind,
    TARGET_PACK_SCHEMA_VERSION,
};
pub use profile::{
    ExternalNumericalAdmission, TargetPackProfile, TargetPackRequirements, TargetPackSelection,
};

use crate::{ArtifactError, ArtifactResult};
use runmat_execution::Digest;

const MAX_COMPONENTS: usize = 4096;
const MAX_EDGES: usize = 65536;
const MAX_LINK_INPUTS: usize = 16384;
const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
const MAX_COMPONENT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_PACK_BYTES: u64 = 4 * 1024 * 1024 * 1024;

fn invalid(message: impl Into<String>) -> ArtifactError {
    ArtifactError::Invalid(message.into())
}
fn nonzero(digest: Digest) -> ArtifactResult<()> {
    if digest.bytes().iter().all(|byte| *byte == 0) {
        return Err(invalid("target pack identity is missing"));
    }
    Ok(())
}
fn contract(error: runmat_execution::ContractError) -> ArtifactError {
    invalid(error.to_string())
}

#[cfg(test)]
mod tests;
