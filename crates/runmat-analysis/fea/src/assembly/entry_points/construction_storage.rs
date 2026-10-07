//! Map shared topology construction failures at the structural assembly boundary.
use super::super::StructuralAssemblyError;
use crate::topology::storage::TopologyStorageError;
impl From<TopologyStorageError> for StructuralAssemblyError {
    fn from(error: TopologyStorageError) -> Self {
        match error {
            TopologyStorageError::Cancelled => Self::Cancelled,
            other => Self::MixedTopology(other.to_string()),
        }
    }
}
