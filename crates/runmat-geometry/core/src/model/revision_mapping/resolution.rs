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
            match operation {
                GeometryRevisionOperation::Retain {
                    source: candidate,
                    target,
                } if candidate == source => {
                    return Ok(GeometryRevisionResolution::Retained(target.clone()));
                }
                GeometryRevisionOperation::Replace {
                    source: candidate,
                    target,
                } if candidate == source => {
                    return Ok(GeometryRevisionResolution::Replaced(target.clone()));
                }
                GeometryRevisionOperation::Split {
                    source: candidate,
                    targets,
                } if candidate == source => {
                    return Ok(GeometryRevisionResolution::Split(targets.clone()));
                }
                GeometryRevisionOperation::Merge { sources, target }
                    if sources.binary_search(source).is_ok() =>
                {
                    return Ok(GeometryRevisionResolution::Merged(target.clone()));
                }
                GeometryRevisionOperation::Delete { source: candidate } if candidate == source => {
                    return Ok(GeometryRevisionResolution::Deleted);
                }
                _ => {}
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
        match self.resolve(source)? {
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
