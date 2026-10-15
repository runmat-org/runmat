use super::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TargetPackId(Digest);
impl TargetPackId {
    pub fn digest(self) -> Digest {
        self.0
    }
}

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
        digest.update(b"runmat-target-runtime-pack-v1\0");
        digest.update(&bytes);
        Ok(TargetPackId(digest.finish()))
    }

    /// Authenticate the exact selected immutable byte closure, not machine
    /// code or target execution. No filesystem, host ABI, dynamic loading,
    /// backend qualification, or numerical execution is performed here.
    pub fn verify_selected_bytes<'a>(
        &self,
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
        Ok(VerifiedTargetPackContents { selected, contents })
    }
}

/// Borrows the very bytes that were authenticated. Consumers cannot replace
/// an input file behind a digest-only success token. Further backend admission
/// remains required before linking or executing these bytes.
pub struct VerifiedTargetPackContents<'a> {
    selected: &'a TargetPackSelection,
    contents: &'a BTreeMap<ComponentId, Vec<u8>>,
}
impl VerifiedTargetPackContents<'_> {
    pub fn selection(&self) -> &TargetPackSelection {
        self.selected
    }
    pub fn component_bytes(&self, id: &ComponentId) -> Option<&[u8]> {
        self.contents.get(id).map(Vec::as_slice)
    }
}
