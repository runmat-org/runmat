use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use super::Digest;

/// Compact immutable reference to one authenticated foreign-policy object.
/// Artifact owns namespace, name, media type, and byte-bound admission.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignPolicyRef {
    digest: Digest,
    encoded_length: NonZeroU64,
}

impl ForeignPolicyRef {
    pub const fn new(digest: Digest, encoded_length: NonZeroU64) -> Self {
        Self {
            digest,
            encoded_length,
        }
    }

    pub const fn digest(self) -> Digest {
        self.digest
    }

    pub const fn encoded_length(self) -> NonZeroU64 {
        self.encoded_length
    }
}
