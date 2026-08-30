use runmat_execution::Digest;
use runmat_types::{ForeignAdapterId, ForeignArtifactIdentity};
use serde::{Deserialize, Serialize};

use crate::{ArtifactError, ArtifactResult};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignArtifactClosure {
    pub adapter: ForeignAdapterId,
    pub identity: ForeignArtifactIdentity,
    pub object_digests: Vec<Digest>,
}

impl ForeignArtifactClosure {
    pub fn new(
        adapter: ForeignAdapterId,
        identity: ForeignArtifactIdentity,
        mut object_digests: Vec<Digest>,
    ) -> ArtifactResult<Self> {
        object_digests.sort();
        object_digests.dedup();
        let closure = Self {
            adapter,
            identity,
            object_digests,
        };
        closure.validate()?;
        Ok(closure)
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.object_digests.is_empty()
            || self
                .object_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(ArtifactError::Invalid(
                "foreign artifact closure is empty or non-canonical".into(),
            ));
        }
        Ok(())
    }
}
