use super::{CanonicalPackageId, ContentDigest, PackageVersion, SourceId};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PackageInstanceId {
    pub package: CanonicalPackageId,
    pub source: SourceId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<PackageVersion>,
    pub tree_digest: ContentDigest,
    pub identity_digest: ContentDigest,
}

impl PackageInstanceId {
    pub fn new(
        package: CanonicalPackageId,
        source: SourceId,
        version: Option<PackageVersion>,
        tree_digest: ContentDigest,
    ) -> Self {
        let canonical = format!(
            "package-instance-v1\npackage={package}\nsource={source}\nversion={}\ntree={tree_digest}\n",
            version
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default()
        );
        let identity_digest = ContentDigest::sha256(canonical);
        Self {
            package,
            source,
            version,
            tree_digest,
            identity_digest,
        }
    }
}

impl Display for PackageInstanceId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}#{}", self.package, self.identity_digest)
    }
}
