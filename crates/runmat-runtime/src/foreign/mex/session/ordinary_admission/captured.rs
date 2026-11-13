//! Checked ordinary-source capture, without inventing exact-build provenance.
use std::collections::BTreeSet;
use std::path::Path;

use runmat_execution::Digest;
use runmat_types::{
    CapabilityRequirement, CapabilitySet, ExecutionStackRequirement, ForeignAdapterId,
    ForeignAdapterRequirement, ForeignArtifactIdentity, InteropManifest,
    INTEROP_MANIFEST_SCHEMA_VERSION,
};

use super::{artifact_catalog, runtime_error};
use crate::foreign::mex::isolation::UnmanifestedMexPolicy;
use crate::RuntimeError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum OrdinaryMexEvidence {
    Exact { manifest_digest: Digest },
    Compatible { host_abi: u32, adapter_version: u32 },
}

impl OrdinaryMexEvidence {
    pub(super) fn admit_policy(&self, policy: UnmanifestedMexPolicy) -> Result<(), RuntimeError> {
        if matches!(self, Self::Compatible { .. }) && policy == UnmanifestedMexPolicy::Deny {
            return Err(runtime_error(
                "MEX:UnmanifestedBinaryDenied",
                "unmanifested MEX module is disabled by runtime policy",
            ));
        }
        Ok(())
    }
}

pub(super) struct CapturedOrdinaryMex {
    name: String,
    bytes: Vec<u8>,
    identity: ForeignArtifactIdentity,
    evidence: OrdinaryMexEvidence,
    exact: Option<ExactManifest>,
}

struct ExactManifest {
    bytes: Vec<u8>,
    manifest: runmat_mex::MexArtifactManifest,
}

/// Only actual absence selects compatibility. A dangling link, directory,
/// unreadable or malformed existing sidecar must go through exact admission.
fn sidecar_exists(module: &Path) -> Result<bool, RuntimeError> {
    let sidecar = runmat_mex::MexArtifactManifest::path_for_module(module);
    match std::fs::symlink_metadata(sidecar) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(runtime_error(
            "MEX:Artifact",
            format!("could not inspect MEX manifest: {error}"),
        )),
    }
}

impl CapturedOrdinaryMex {
    pub(super) fn capture(
        module: &Path,
        policy: UnmanifestedMexPolicy,
    ) -> Result<Self, RuntimeError> {
        if sidecar_exists(module)? {
            let captured = artifact_catalog::capture_artifact(module, None)?;
            return Ok(Self {
                name: captured.name,
                bytes: captured.module_bytes,
                identity: captured.installed.identity,
                evidence: OrdinaryMexEvidence::Exact {
                    manifest_digest: Digest::sha256(&captured.manifest_bytes),
                },
                exact: Some(ExactManifest {
                    bytes: captured.manifest_bytes,
                    manifest: captured.manifest,
                }),
            });
        }
        let evidence = OrdinaryMexEvidence::Compatible {
            host_abi: runmat_mex::MEX_HOST_ABI_VERSION,
            adapter_version: runmat_mex::MEX_ADAPTER_VERSION,
        };
        evidence.admit_policy(policy)?;
        let suffix = runmat_mex::mex_suffix()
            .ok_or_else(|| runtime_error("MEX:Artifact", "current platform has no MEX suffix"))?;
        if module.extension().and_then(|value| value.to_str()) != Some(suffix) {
            return Err(runtime_error(
                "MEX:Artifact",
                "compatible MEX suffix differs from the current target",
            ));
        }
        let name = module
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or_else(|| runtime_error("MEX:Artifact", "MEX module has no valid callable name"))?
            .to_owned();
        let bytes = artifact_catalog::read_bounded(
            module,
            artifact_catalog::MAX_MEX_MODULE_BYTES,
            "MEX module",
        )?;
        if sidecar_exists(module)? {
            return Err(runtime_error(
                "MEX:Artifact",
                "MEX sidecar changed during compatible capture",
            ));
        }
        let identity = Digest::sha256_serialized(&(
            "runmat.compatible-isolated-mex.v1",
            runmat_mex::MEX_HOST_ABI_VERSION,
            runmat_mex::MEX_ADAPTER_VERSION,
            suffix,
            Digest::sha256(&bytes),
        ))
        .map_err(|error| runtime_error("MEX:Artifact", error.to_string()))?;
        let identity = ForeignArtifactIdentity::new(identity.to_string())
            .map_err(|error| runtime_error("MEX:Artifact", error.to_string()))?;
        // This binds only captured bytes and the current compatibility contract.
        // Actual adapter symbols, private ABI and API mode are validated by
        // the canonical loader INSIDE the isolated host before any gateway call.
        Ok(Self {
            name,
            bytes,
            identity,
            evidence,
            exact: None,
        })
    }

    pub(super) fn name(&self) -> &str {
        &self.name
    }
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub(super) fn manifest_bytes(&self) -> Option<&[u8]> {
        self.exact.as_ref().map(|exact| exact.bytes.as_slice())
    }
    pub(super) fn identity(&self) -> &ForeignArtifactIdentity {
        &self.identity
    }
    pub(super) fn evidence(&self) -> OrdinaryMexEvidence {
        self.evidence.clone()
    }
    pub(super) fn interop_manifest(&self) -> InteropManifest {
        match &self.exact {
            Some(exact) => exact.manifest.interop_manifest(),
            None => InteropManifest {
                schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
                foreign_types: Vec::new(),
                adapters: vec![ForeignAdapterRequirement {
                    adapter: ForeignAdapterId::new(runmat_mex::MEX_ADAPTER_ID)
                        .expect("built-in MEX adapter identity"),
                    minimum_version: runmat_mex::MEX_ADAPTER_VERSION,
                    capabilities: CapabilitySet(BTreeSet::from([
                        CapabilityRequirement::NativeCode,
                        CapabilityRequirement::ForeignRuntime,
                    ])),
                    execution_stack: ExecutionStackRequirement::Process,
                    artifact_identities: vec![self.identity.clone()],
                }],
                adapter_contracts: Vec::new(),
            },
        }
    }
    pub(super) fn require_ordinary_resources(&self) -> Result<(), RuntimeError> {
        if let Some(exact) = &self.exact {
            let requirements = exact
                .manifest
                .accelerator_requirements()
                .map_err(|error| runtime_error("MEX:Capability", error.to_string()))?;
            if !requirements.is_empty() {
                return Err(runtime_error(
                    "MEX:Capability",
                    "ordinary CUDA MEX requires an admitted accelerator resource lease",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "captured/tests.rs"]
mod tests;
