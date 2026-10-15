use super::*;
pub use runmat_execution::host::TargetPackId;
use std::collections::BTreeMap;

impl TargetPackManifest {
    pub fn canonical_bytes(&self) -> ArtifactResult<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| crate::ArtifactError::Encoding(error.to_string()))?;
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(invalid("target pack manifest exceeds its byte bound"));
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> ArtifactResult<Self> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(invalid("target pack manifest exceeds its byte bound"));
        }
        #[derive(serde::Deserialize)]
        struct Admission {
            schema_version: u16,
        }
        let admission: Admission = serde_json::from_slice(bytes)
            .map_err(|error| crate::ArtifactError::Encoding(error.to_string()))?;
        if admission.schema_version != TARGET_PACK_SCHEMA_VERSION {
            return Err(invalid("unsupported target runtime pack schema"));
        }
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|error| crate::ArtifactError::Encoding(error.to_string()))?;
        if value.canonical_bytes()? != bytes {
            return Err(invalid("target pack manifest encoding is not canonical"));
        }
        Ok(value)
    }

    pub fn id(&self) -> ArtifactResult<TargetPackId> {
        let bytes = self.canonical_bytes()?;
        let mut digest = runmat_execution::DigestStream::new();
        digest.update(b"runmat-target-runtime-pack-v2\0");
        digest.update(&bytes);
        TargetPackId::from_digest(digest.finish()).map_err(contract)
    }

    /// Authenticate the exact selected immutable byte closure, not machine
    /// code or target execution. No filesystem, host ABI, dynamic loading,
    /// backend qualification, or numerical execution is performed here.
    pub fn verify_selected_bytes<'a>(
        &'a self,
        selected: &'a TargetPackSelection,
        contents: &'a BTreeMap<ComponentId, Vec<u8>>,
    ) -> ArtifactResult<VerifiedTargetPackContents<'a>> {
        if selected.pack() != self.id()?
            || contents.len() != selected.components().len()
            || !contents.keys().eq(selected.components().iter())
        {
            return Err(invalid("target pack selected content inventory differs"));
        }
        let index = components::index(&self.components)?;
        for id in selected.components() {
            let component = index
                .get(id)
                .ok_or_else(|| invalid("selected target pack component is missing"))?;
            let bytes = &contents[id];
            if u64::try_from(bytes.len()).ok() != Some(component.object.encoded_length)
                || Digest::sha256(bytes) != component.object.digest
            {
                return Err(crate::ArtifactError::Identity(
                    "selected target pack bytes differ from their exact descriptor".into(),
                ));
            }
        }
        let probe = super::probe::verify(self, contents)?;
        Ok(VerifiedTargetPackContents {
            manifest: self,
            selected,
            contents,
            probe,
        })
    }
}

/// Borrows the very bytes that were authenticated. Consumers cannot replace
/// an input file behind a digest-only success token. Further backend admission
/// remains required before linking or executing these bytes.
pub struct VerifiedTargetPackContents<'a> {
    manifest: &'a TargetPackManifest,
    selected: &'a TargetPackSelection,
    contents: &'a BTreeMap<ComponentId, Vec<u8>>,
    probe: runmat_execution::host::NativeAbiProbe,
}
impl VerifiedTargetPackContents<'_> {
    pub fn manifest(&self) -> &TargetPackManifest {
        self.manifest
    }
    pub fn abi_probe(&self) -> &runmat_execution::host::NativeAbiProbe {
        &self.probe
    }
    pub fn selection(&self) -> &TargetPackSelection {
        self.selected
    }
    pub fn component_bytes(&self, id: &ComponentId) -> Option<&[u8]> {
        self.contents.get(id).map(Vec::as_slice)
    }
    /// Canonical SCC-safe graph projection within these authenticated bytes.
    /// Returned IDs are metadata membership, never execution authority.
    pub fn component_dependency_closure(
        &self,
        root: &ComponentId,
    ) -> ArtifactResult<std::collections::BTreeSet<ComponentId>> {
        if self.component_bytes(root).is_none() {
            return Err(invalid("dependency root is not in this verified selection"));
        }
        let index = components::index(&self.manifest.components)?;
        let mut reached = std::collections::BTreeSet::new();
        components::retain(&index, [root.clone()], &mut reached)?;
        if reached.iter().any(|id| self.component_bytes(id).is_none()) {
            return Err(invalid(
                "dependency closure is not authenticated in this selection",
            ));
        }
        Ok(reached)
    }
}
