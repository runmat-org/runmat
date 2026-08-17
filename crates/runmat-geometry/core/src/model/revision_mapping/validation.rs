use std::collections::BTreeSet;

use super::{GeometryRevisionMap, GeometryRevisionOperation, GEOMETRY_REVISION_MAP_SCHEMA_VERSION};
use crate::{GeometryContractError, PersistentEntityId};

const MAX_REVISION_OPERATIONS: usize = 10_000_000;
const MAX_ENTITIES_PER_OPERATION: usize = 1_000_000;

impl GeometryRevisionMap {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        if self.schema_version != GEOMETRY_REVISION_MAP_SCHEMA_VERSION {
            return Err(invalid(
                "geometry revision map schema",
                "unsupported version",
            ));
        }
        self.source_geometry_digest
            .validate_nonzero("source geometry digest")?;
        self.target_geometry_digest
            .validate_nonzero("target geometry digest")?;
        if self.source_geometry_digest == self.target_geometry_digest {
            return Err(invalid(
                "geometry revision map digests",
                "a topology revision must change authoritative geometry identity",
            ));
        }
        validate_revision_chain(self)?;
        if self.operations.is_empty() || self.operations.len() > MAX_REVISION_OPERATIONS {
            return Err(invalid(
                "geometry revision operations",
                "the operation inventory must be non-empty and within its hard bound",
            ));
        }

        let mut consumed_sources = BTreeSet::new();
        let mut produced_targets = BTreeSet::new();
        let mut prior_primary_source = None;
        for operation in &self.operations {
            validate_operation(operation)?;
            let primary_source = operation.sources()[0].clone();
            if prior_primary_source
                .as_ref()
                .is_some_and(|prior| prior >= &primary_source)
            {
                return Err(invalid(
                    "geometry revision operation order",
                    "operations must be strictly ordered by their first source entity",
                ));
            }
            prior_primary_source = Some(primary_source);

            for source in operation.sources() {
                if !consumed_sources.insert(source.clone()) {
                    return Err(invalid(
                        "geometry revision source ownership",
                        "a source entity may occur in exactly one operation",
                    ));
                }
            }
            for target in operation.targets() {
                if !produced_targets.insert(target.clone()) {
                    return Err(invalid(
                        "geometry revision target ownership",
                        "a target entity may be produced by exactly one operation",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl GeometryRevisionOperation {
    pub(super) fn sources(&self) -> &[PersistentEntityId] {
        match self {
            Self::Retain { source, .. }
            | Self::Replace { source, .. }
            | Self::Split { source, .. }
            | Self::Delete { source } => std::slice::from_ref(source),
            Self::Merge { sources, .. } => sources,
        }
    }

    pub(super) fn targets(&self) -> &[PersistentEntityId] {
        match self {
            Self::Retain { target, .. }
            | Self::Replace { target, .. }
            | Self::Merge { target, .. } => std::slice::from_ref(target),
            Self::Split { targets, .. } => targets,
            Self::Delete { .. } => &[],
        }
    }
}

fn validate_revision_chain(map: &GeometryRevisionMap) -> Result<(), GeometryContractError> {
    let source = &map.source_revision;
    let target = &map.target_revision;
    if source.revision == 0
        || target.revision == 0
        || source.persistent_mapping_version == 0
        || target.persistent_mapping_version == 0
        || target.revision <= source.revision
    {
        return Err(invalid(
            "geometry revision chain",
            "revisions and mapping versions must be non-zero and the target revision must advance",
        ));
    }
    if target.parent_document_digest != Some(map.source_geometry_digest) {
        return Err(invalid(
            "geometry revision parent",
            "the target revision must name the source authoritative geometry digest as its parent",
        ));
    }
    Ok(())
}

fn validate_operation(operation: &GeometryRevisionOperation) -> Result<(), GeometryContractError> {
    let sources = operation.sources();
    let targets = operation.targets();
    if sources.len() > MAX_ENTITIES_PER_OPERATION || targets.len() > MAX_ENTITIES_PER_OPERATION {
        return Err(invalid(
            "geometry revision operation size",
            "an operation exceeds its hard entity bound",
        ));
    }
    validate_ids("geometry revision sources", sources, false)?;
    validate_ids(
        "geometry revision targets",
        targets,
        matches!(operation, GeometryRevisionOperation::Delete { .. }),
    )?;

    let kind = sources[0].kind;
    if sources.iter().any(|source| source.kind != kind)
        || targets.iter().any(|target| target.kind != kind)
    {
        return Err(invalid(
            "geometry revision entity kind",
            "revision operations cannot change entity kind",
        ));
    }
    match operation {
        GeometryRevisionOperation::Retain { source, target } if source != target => Err(invalid(
            "retained geometry entity",
            "retained entities must preserve their complete persistent identity",
        )),
        GeometryRevisionOperation::Replace { source, target } if source == target => Err(invalid(
            "replaced geometry entity",
            "replacement requires a new persistent identity",
        )),
        GeometryRevisionOperation::Split { targets, .. } if targets.len() < 2 => Err(invalid(
            "split geometry entity",
            "a split must produce at least two target entities",
        )),
        GeometryRevisionOperation::Merge { sources, .. } if sources.len() < 2 => Err(invalid(
            "merged geometry entity",
            "a merge must consume at least two source entities",
        )),
        _ => Ok(()),
    }
}

fn validate_ids(
    field: &str,
    ids: &[PersistentEntityId],
    allow_empty: bool,
) -> Result<(), GeometryContractError> {
    if ids.is_empty() && !allow_empty {
        return Err(invalid(field, "entity list must be non-empty"));
    }
    let mut prior = None;
    for id in ids {
        id.validate()?;
        if prior.is_some_and(|value| value >= id) {
            return Err(invalid(field, "entity list must be strictly canonical"));
        }
        prior = Some(id);
    }
    Ok(())
}

fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
