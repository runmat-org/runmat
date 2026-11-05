//! Map shared topology construction failures at the structural assembly boundary.
use super::super::StructuralAssemblyError;
use crate::topology::storage::TopologyStorageError;
use crate::{
    constraints::ConstraintPlanError,
    physics::structural::topology::node_space::StructuralNodeSpaceError,
};
impl From<TopologyStorageError> for StructuralAssemblyError {
    fn from(error: TopologyStorageError) -> Self {
        match error {
            TopologyStorageError::Cancelled => Self::Cancelled,
            other => Self::MixedTopology(other.to_string()),
        }
    }
}

impl From<StructuralNodeSpaceError> for StructuralAssemblyError {
    fn from(error: StructuralNodeSpaceError) -> Self {
        match error {
            StructuralNodeSpaceError::Storage(TopologyStorageError::Cancelled) => Self::Cancelled,
            other => Self::ExplicitTopologyIdentity(other.to_string()),
        }
    }
}

impl From<ConstraintPlanError> for StructuralAssemblyError {
    fn from(error: ConstraintPlanError) -> Self {
        match error {
            ConstraintPlanError::Cancelled => Self::Cancelled,
            ConstraintPlanError::Accounting(error) => Self::Accounting(error),
            other => Self::InvalidConstraintPlan(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_cancellation_retains_its_category_at_assembly_boundary() {
        for error in [
            StructuralAssemblyError::from(TopologyStorageError::Cancelled),
            StructuralAssemblyError::from(StructuralNodeSpaceError::Storage(
                TopologyStorageError::Cancelled,
            )),
            StructuralAssemblyError::from(ConstraintPlanError::Cancelled),
        ] {
            assert_eq!(error, StructuralAssemblyError::Cancelled);
        }
        assert!(matches!(
            StructuralAssemblyError::from(StructuralNodeSpaceError::Storage(
                TopologyStorageError::Capacity
            )),
            StructuralAssemblyError::ExplicitTopologyIdentity(_)
        ));
    }
}
