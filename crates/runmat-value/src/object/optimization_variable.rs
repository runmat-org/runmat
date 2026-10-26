//! Optimization-variable data. Runtime owns public argument admission and GC handles.
//!
//! This module owns no registry, callback, scheduler, or second variable identity.
//! Definitions belong to ordinary GC-owned handle objects; indexed references
//! retain that same definition handle rather than copying its bounds.
mod bounds;
mod definition;
mod domain;
pub use domain::OptimizationVariableDomain;
mod selection;

pub use bounds::{
    OptimizationBoundArray, OptimizationBoundRole, OptimizationBoundStorage, OptimizationBoundView,
};
pub use definition::{
    OptimizationVariableCandidate, OptimizationVariableData, OptimizationVariableLayoutError,
    OptimizationVariablePreparation, OptimizationVariableValidation,
};
pub use selection::{
    OptimizationVariableAxisSource, OptimizationVariableIndices, OptimizationVariableReferenceRole,
    OptimizationVariableSelection, OptimizationVariableSelectionCandidate,
    OptimizationVariableSelectionError, OptimizationVariableSelectionPreparation,
    OptimizationVariableSelectionValidation,
};

#[cfg(test)]
mod tests;

/// Private-definition versus public-reference roles share the existing native
/// payload container. Only a Reference is a publishable public variable value.
/// This data role grants no class issuance, GC-target or allocation authority.
#[derive(Debug, PartialEq)]
pub enum OptimizationVariablePayload {
    Definition(OptimizationVariableData),
    Reference(OptimizationVariableSelection),
}
impl OptimizationVariablePayload {
    pub fn value_at(&self, index: usize) -> Option<&crate::Value> {
        let Self::Definition(definition) = self else {
            return None;
        };
        let mut next = 0;
        for role in [OptimizationBoundRole::Lower, OptimizationBoundRole::Upper] {
            if let OptimizationBoundStorage::PerElement(array) = definition.bound_storage(role) {
                if next == index {
                    return Some(array.value());
                }
                next += 1;
            }
        }
        None
    }
}
impl runmat_gc_api::Trace for OptimizationVariablePayload {
    fn trace(&self, tracer: &mut dyn runmat_gc_api::Tracer) {
        if let Self::Reference(reference) = self {
            runmat_gc_api::Trace::trace(reference, tracer);
        }
        let mut index = 0;
        while let Some(value) = self.value_at(index) {
            runmat_gc_api::Trace::trace(value, tracer);
            index += 1;
        }
    }
}
