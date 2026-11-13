use super::{
    GeometryRevisionConflict, GeometryRevisionConflictKind, GeometryRevisionMap,
    GeometryRevisionMappingError, GeometryRevisionOperation, GeometryRevisionResolution,
};
use crate::PersistentEntityId;

impl GeometryRevisionMap {
    /// Returns the recorded disposition without inventing a policy for splits.
    pub fn resolve(
        &self,
        source: &PersistentEntityId,
    ) -> Result<GeometryRevisionResolution, GeometryRevisionMappingError> {
        self.validate()
            .map_err(GeometryRevisionMappingError::InvalidMap)?;
        source
            .validate()
            .map_err(GeometryRevisionMappingError::InvalidMap)?;

        for operation in &self.operations {
            if let Some(resolution) = resolve_operation(operation, source) {
                return Ok(resolution);
            }
        }
        Err(conflict(
            source,
            GeometryRevisionConflictKind::SourceNotMapped,
            Vec::new(),
        ))
    }

    /// Resolves consumers that require exactly one target. Deletes return `None`; splits fail
    /// with every canonical candidate so product code must make an explicit semantic choice.
    pub fn resolve_unique(
        &self,
        source: &PersistentEntityId,
    ) -> Result<Option<PersistentEntityId>, GeometryRevisionMappingError> {
        unique_resolution(source, self.resolve(source)?)
    }
}

pub(super) fn resolve_operation(
    operation: &GeometryRevisionOperation,
    source: &PersistentEntityId,
) -> Option<GeometryRevisionResolution> {
    match operation {
        GeometryRevisionOperation::Retain {
            source: candidate,
            target,
        } if candidate == source => Some(GeometryRevisionResolution::Retained(target.clone())),
        GeometryRevisionOperation::Replace {
            source: candidate,
            target,
        } if candidate == source => Some(GeometryRevisionResolution::Replaced(target.clone())),
        GeometryRevisionOperation::Split {
            source: candidate,
            targets,
        } if candidate == source => Some(GeometryRevisionResolution::Split(targets.clone())),
        GeometryRevisionOperation::Merge { sources, target }
            if sources.binary_search(source).is_ok() =>
        {
            Some(GeometryRevisionResolution::Merged(target.clone()))
        }
        GeometryRevisionOperation::Delete { source: candidate } if candidate == source => {
            Some(GeometryRevisionResolution::Deleted)
        }
        _ => None,
    }
}

pub(super) fn unique_resolution(
    source: &PersistentEntityId,
    resolution: GeometryRevisionResolution,
) -> Result<Option<PersistentEntityId>, GeometryRevisionMappingError> {
    match resolution {
        GeometryRevisionResolution::Retained(target)
        | GeometryRevisionResolution::Replaced(target)
        | GeometryRevisionResolution::Merged(target) => Ok(Some(target)),
        GeometryRevisionResolution::Deleted => Ok(None),
        GeometryRevisionResolution::Split(candidates) => Err(conflict(
            source,
            GeometryRevisionConflictKind::MultipleCandidates,
            candidates,
        )),
    }
}

fn conflict(
    source: &PersistentEntityId,
    kind: GeometryRevisionConflictKind,
    candidate_entities: Vec<PersistentEntityId>,
) -> GeometryRevisionMappingError {
    GeometryRevisionMappingError::Conflict(GeometryRevisionConflict {
        source_entity: source.clone(),
        kind,
        candidate_entities,
    })
}
