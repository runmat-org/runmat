//! Portable code/capture claims, without job or current-session authority.

use super::{
    GraphClassDependency, GraphClassMethodDependency, GraphStaticDependency, PayloadGraph,
};
use serde::{Deserialize, Serialize};

/// Ordered selected V3 products and a single graph of actual callback roots.
/// Every field is untrusted data until archive and VM code/source/static/
/// interop admission succeeds. Stored source ordinals are portable, never
/// RuntimeSourceIds. Code retains no runtime, callback invoker or resource grant.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableSelectedCode {
    pub schema_version: u16,
    pub primary_unit: Vec<u8>,
    pub retained_units: Vec<Vec<u8>>,
    pub graph: PayloadGraph,
    pub class_dependencies: Vec<GraphClassDependency>,
    pub class_method_dependencies: Vec<GraphClassMethodDependency>,
    pub static_dependencies: Vec<GraphStaticDependency>,
    pub selected_bundle_archives: Vec<Option<Vec<u8>>>,
    pub selected_foreign_policies: Vec<Option<crate::ForeignPolicyRef>>,
}

impl PortableSelectedCode {
    /// Portable callable-root ordinal capacity; independent of task/job roles.
    pub const MAX_CALLABLE_ROOTS: usize = 4096;
}
