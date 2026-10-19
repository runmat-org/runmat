//! Consume producer inventories before cold verification of saved CLI outputs.
use std::{fs::File, io::Read, path::Path};

use anyhow::{bail, Context, Result};
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingEvidence, MeshingResourceUsage, MeshingStageEvidence,
    SolverMeshArtifact, StableDigest,
};
use sha2::{Digest, Sha256};

pub(crate) struct MeshOutputSummary {
    pub canonical_digest: StableDigest,
    pub node_count: usize,
    pub element_count: usize,
    pub boundary_face_count: usize,
    pub resources: MeshingResourceUsage,
    pub stages: Vec<MeshingStageEvidence>,
}

pub(super) enum ExistingOutputPolicy {
    CreateNew,
    Replace,
    ReuseIdentical { replace_different: bool },
}

pub(super) fn persist_completed_mesh(
    output: &Path,
    evidence_path: &Path,
    artifact: SolverMeshArtifact,
    mut evidence: MeshingEvidence,
    policy: ExistingOutputPolicy,
) -> Result<MeshOutputSummary> {
    let artifact_bytes = artifact
        .canonical_encode()
        .context("failed to encode solver mesh")?;
    let evidence_bytes = evidence
        .canonical_encode()
        .context("failed to encode meshing evidence")?;
    let artifact_identity = FileIdentity::of(&artifact_bytes);
    let evidence_identity = FileIdentity::of(&evidence_bytes);
    let summary = MeshOutputSummary {
        canonical_digest: artifact.canonical_digest,
        node_count: artifact.topology.nodes.len(),
        element_count: artifact.topology.volume_elements.len(),
        boundary_face_count: artifact.topology.boundary_faces.len(),
        resources: evidence.resources,
        stages: std::mem::take(&mut evidence.stages),
    };
    // The summary owns no mesh/provenance inventories. Do not retain the producer
    // while reconstructing an independently validated persisted topology.
    drop(artifact);
    drop(evidence);
    let (force, allow_identical_existing) = match policy {
        ExistingOutputPolicy::CreateNew => (false, false),
        ExistingOutputPolicy::Replace => (true, false),
        ExistingOutputPolicy::ReuseIdentical { replace_different } => (replace_different, true),
    };
    let exists = output.exists() || evidence_path.exists();
    let reuse = allow_identical_existing
        && exists
        && artifact_identity.matches(output)
        && evidence_identity.matches(evidence_path);
    if !reuse {
        if allow_identical_existing && exists && !force {
            bail!("mesh output destinations already contain a different or incomplete result; resubmit with --force to authorize replacement");
        }
        runmat_runtime::meshing::write_canonical_artifact_pair(
            output,
            &artifact_bytes,
            evidence_path,
            &evidence_bytes,
            force,
        )
        .map_err(anyhow::Error::msg)?;
    }
    drop(artifact_bytes);
    drop(evidence_bytes);
    let bytes = artifact_identity.read_verified(output)?;
    let artifact = SolverMeshArtifact::canonical_decode(&bytes)
        .context("persisted solver mesh failed canonical decoding")?;
    drop(bytes);
    let bytes = evidence_identity.read_verified(evidence_path)?;
    let evidence = MeshingEvidence::canonical_decode(&bytes)
        .context("persisted meshing evidence failed canonical decoding")?;
    drop(bytes);
    evidence
        .validate(&artifact)
        .context("persisted meshing outputs failed cross-validation")?;
    Ok(summary)
}

struct FileIdentity {
    length: u64,
    sha256: [u8; 32],
}
impl FileIdentity {
    fn of(bytes: &[u8]) -> Self {
        Self {
            length: bytes.len() as u64,
            sha256: Sha256::digest(bytes).into(),
        }
    }
    fn matches(&self, path: &Path) -> bool {
        let Ok(file) = File::open(path) else {
            return false;
        };
        let mut input = file.take(self.length.saturating_add(1));
        let mut buffer = [0; 64 * 1024];
        let mut hash = Sha256::new();
        let mut length = 0;
        loop {
            let Ok(count) = input.read(&mut buffer) else {
                return false;
            };
            if count == 0 {
                break;
            }
            length += count as u64;
            hash.update(&buffer[..count]);
        }
        length == self.length && <[u8; 32]>::from(hash.finalize()) == self.sha256
    }
    fn read_verified(&self, path: &Path) -> Result<Vec<u8>> {
        let limit = self
            .length
            .checked_add(1)
            .context("persisted output length overflows")?;
        let mut bytes = Vec::with_capacity(usize::try_from(limit)?);
        File::open(path)
            .with_context(|| format!("open persisted mesh output {}", path.display()))?
            .take(limit)
            .read_to_end(&mut bytes)
            .with_context(|| format!("read persisted mesh output {}", path.display()))?;
        if bytes.len() as u64 != self.length
            || <[u8; 32]>::from(Sha256::digest(&bytes)) != self.sha256
        {
            bail!("persisted meshing outputs differ from the completed canonical bytes");
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests;
