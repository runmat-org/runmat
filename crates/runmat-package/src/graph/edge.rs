use crate::{ContentDigest, DependencyGroup, PackageAlias, TargetPredicate};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphEdge {
    pub from: ContentDigest,
    pub alias: PackageAlias,
    pub to: ContentDigest,
    pub group: DependencyGroup,
    pub optional: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<TargetPredicate>,
}
