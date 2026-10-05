//! Immutable source references of the captured model composition.
use super::*;

impl RealizedStudy {
    /// Declared immutable sources across the complete captured model composition.
    /// Activation and physical admission remain part of exact model realization.
    pub fn initial_field_references(
        &self,
    ) -> impl Iterator<Item = &crate::InitialFieldArtifactRef> {
        self.included_models
            .iter()
            .chain(std::iter::once(&self.model))
            .flat_map(|model| &model.initial_conditions)
            .filter_map(|condition| condition.kind.field_artifact_ref())
    }
}
