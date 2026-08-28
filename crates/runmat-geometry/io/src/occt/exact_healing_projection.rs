//! Projects successful native orientation repair into geometry-owned revision evidence.

use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    encode_exact_topology, GeometryDigest, GeometryHealingFailure, GeometryHealingOperation,
    GeometryHealingOperationKind, GeometryHealingReport, GeometryRevisionMap,
    GeometryRevisionOperation, GeometryTolerancePolicy, PersistentEntityId, PersistentEntityKind,
    TopologyValidity, GEOMETRY_HEALING_REPORT_SCHEMA_VERSION, GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

use super::{
    exact_persistent_names::digest_name,
    exact_projection_identity::{scoped_id, ROOT_SCOPE},
    ffi::bridge,
};
use crate::{exact::ImportedExactCad, import::GeometryImportError};

pub(super) struct NativeHealingEvidence<'a> {
    pub original_digest: &'a [u8],
    pub original_kernel_valid: bool,
    pub post_duplicate_kernel_valid: bool,
    pub duplicates_consolidated: bool,
    pub orientation_repaired: bool,
    pub sewn: bool,
    pub gaps_repaired: bool,
    pub post_sewing_kernel_valid: bool,
    pub short_edges_simplified: bool,
    pub sliver_faces_simplified: bool,
    pub post_small_topology_kernel_valid: bool,
    pub relations: &'a [bridge::OcctHealingRelationPayload],
    pub maximum_displacement_m: f64,
    pub displacement_original_m: [f64; 3],
    pub displacement_proposed_m: [f64; 3],
}

pub(super) fn healing_report(
    evidence: NativeHealingEvidence<'_>,
    imported: &ImportedExactCad,
) -> Result<GeometryHealingReport, GeometryImportError> {
    let original_topology_digest = parse_digest(evidence.original_digest)?;
    let topology_bytes =
        encode_exact_topology(&imported.topology, &imported.model).map_err(contract_failure)?;
    let healed_topology_digest = GeometryDigest::from_bytes(Sha256::digest(topology_bytes).into());
    let source_revision = imported.analysis.revision.clone();
    let mut target_revision = source_revision.clone();
    target_revision.revision = target_revision.revision.checked_add(1).ok_or_else(|| {
        GeometryImportError::InvalidOptions("exact geometry revision cannot advance".into())
    })?;
    target_revision.parent_document_digest = Some(original_topology_digest);

    let has_sewing = evidence.sewn || evidence.gaps_repaired;
    let has_small_topology = evidence.short_edges_simplified || evidence.sliver_faces_simplified;
    let has_lineage_mutation = has_sewing || has_small_topology;
    let lineage = if has_lineage_mutation {
        project_relations(evidence.relations, imported)?
    } else {
        retained_lineage(topology_entities(imported))
    };
    let revision_map = GeometryRevisionMap {
        schema_version: GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
        source_geometry_digest: original_topology_digest,
        source_revision,
        target_geometry_digest: healed_topology_digest,
        target_revision,
        operations: lineage.operations.clone(),
    };
    let original_validity = TopologyValidity {
        kernel_valid: evidence.original_kernel_valid,
        incidence_consistent: !has_sewing,
        orientation_consistent: !evidence.orientation_repaired,
        shells_closed: !has_sewing,
        nesting_consistent: !evidence.orientation_repaired,
    };
    let healed_validity = TopologyValidity {
        kernel_valid: true,
        incidence_consistent: true,
        orientation_consistent: true,
        shells_closed: true,
        nesting_consistent: true,
    };
    let tolerance = GeometryTolerancePolicy {
        source_tolerance_m: imported
            .topology
            .vertices
            .iter()
            .map(|vertex| vertex.tolerance_m)
            .fold(0.0_f64, f64::max),
        absolute_floor_m: imported.analysis.absolute_tolerance_floor_m,
        model_relative_term: imported.analysis.model_relative_tolerance,
        requested_deviation_m: imported.analysis.requested_deviation_m,
        maximum_healing_displacement_m: imported.analysis.maximum_healing_displacement_m,
    };
    let mut operations = Vec::new();
    let mut prior_validity = original_validity;
    if has_lineage_mutation
        && evidence.maximum_displacement_m > tolerance.maximum_healing_displacement_m
    {
        let operation = if evidence.gaps_repaired {
            GeometryHealingOperationKind::RepairGap
        } else if evidence.sewn {
            GeometryHealingOperationKind::Sew
        } else if evidence.sliver_faces_simplified {
            GeometryHealingOperationKind::SimplifySliverFace
        } else {
            GeometryHealingOperationKind::SimplifyShortEdge
        };
        let failure = GeometryHealingFailure {
            operation,
            affected_entities: lineage.affected_before.clone(),
            measured_displacement_m: evidence.maximum_displacement_m,
            permitted_displacement_m: tolerance.maximum_healing_displacement_m,
            original_point_m: evidence.displacement_original_m,
            proposed_point_m: evidence.displacement_proposed_m,
            reason: "OCCT topology repair exceeded the admitted healing displacement".into(),
        };
        failure.validate().map_err(contract_failure)?;
        return Err(GeometryImportError::HealingLimitExceeded { failure });
    }
    if evidence.duplicates_consolidated {
        let duplicate_validity = TopologyValidity {
            kernel_valid: evidence.post_duplicate_kernel_valid,
            incidence_consistent: true,
            orientation_consistent: !evidence.orientation_repaired,
            shells_closed: true,
            nesting_consistent: !evidence.orientation_repaired,
        };
        let affected = duplicate_entities(imported);
        operations.push(GeometryHealingOperation {
            sequence: operations.len() as u64,
            kind: GeometryHealingOperationKind::ConsolidateDuplicate,
            affected_before: affected.clone(),
            affected_after: affected,
            maximum_displacement_m: 0.0,
            reason: "OCCT consolidated indistinguishable duplicate compound children into one semantic entity without moving geometry".into(),
            before_validity: prior_validity,
            after_validity: duplicate_validity,
        });
        prior_validity = duplicate_validity;
    }
    if has_sewing {
        let operation_kind = if evidence.gaps_repaired {
            GeometryHealingOperationKind::RepairGap
        } else {
            GeometryHealingOperationKind::Sew
        };
        let sewing_validity = TopologyValidity {
            kernel_valid: evidence.post_sewing_kernel_valid,
            incidence_consistent: true,
            orientation_consistent: !evidence.orientation_repaired,
            shells_closed: true,
            nesting_consistent: !evidence.orientation_repaired,
        };
        operations.push(GeometryHealingOperation {
            sequence: operations.len() as u64,
            kind: operation_kind,
            affected_before: lineage.affected_before.clone(),
            affected_after: lineage.affected_after.clone(),
            maximum_displacement_m: evidence.maximum_displacement_m,
            reason: if evidence.gaps_repaired {
                "OCCT sewed tolerance-scale boundary gaps within the admitted displacement".into()
            } else {
                "OCCT sewed exact boundary uses without exceeding the admitted displacement".into()
            },
            before_validity: prior_validity,
            after_validity: sewing_validity,
        });
        prior_validity = sewing_validity;
    }
    if has_small_topology {
        let small_topology_validity = TopologyValidity {
            kernel_valid: evidence.post_small_topology_kernel_valid,
            incidence_consistent: true,
            orientation_consistent: !evidence.orientation_repaired,
            shells_closed: true,
            nesting_consistent: !evidence.orientation_repaired,
        };
        if evidence.short_edges_simplified {
            operations.push(GeometryHealingOperation {
                sequence: operations.len() as u64,
                kind: GeometryHealingOperationKind::SimplifyShortEdge,
                affected_before: lineage.affected_before.clone(),
                affected_after: lineage.affected_after.clone(),
                maximum_displacement_m: evidence.maximum_displacement_m,
                reason: "OCCT merged tolerance-scale short edges without dropping isolated boundary uses".into(),
                before_validity: prior_validity,
                after_validity: small_topology_validity,
            });
            prior_validity = small_topology_validity;
        }
        if evidence.sliver_faces_simplified {
            operations.push(GeometryHealingOperation {
                sequence: operations.len() as u64,
                kind: GeometryHealingOperationKind::SimplifySliverFace,
                affected_before: lineage.affected_before.clone(),
                affected_after: lineage.affected_after.clone(),
                maximum_displacement_m: evidence.maximum_displacement_m,
                reason: "OCCT removed tolerance-scale spot or strip faces within the admitted displacement".into(),
                before_validity: prior_validity,
                after_validity: small_topology_validity,
            });
            prior_validity = small_topology_validity;
        }
    }
    if evidence.orientation_repaired {
        let affected = orientation_entities(imported);
        operations.push(GeometryHealingOperation {
            sequence: operations.len() as u64,
            kind: GeometryHealingOperationKind::RepairOrientation,
            affected_before: affected.clone(),
            affected_after: affected,
            maximum_displacement_m: 0.0,
            reason: "OCCT repaired solid, shell, and face use orientation without moving geometry"
                .into(),
            before_validity: prior_validity,
            after_validity: healed_validity,
        });
    }
    let report = GeometryHealingReport {
        schema_version: GEOMETRY_HEALING_REPORT_SCHEMA_VERSION,
        original_topology_digest,
        healed_topology_digest,
        policy: imported.analysis.healing.clone(),
        tolerance,
        revision_map,
        original_validity,
        healed_validity,
        operations,
    };
    report.validate().map_err(contract_failure)?;
    Ok(report)
}

struct ProjectedLineage {
    operations: Vec<GeometryRevisionOperation>,
    affected_before: Vec<PersistentEntityId>,
    affected_after: Vec<PersistentEntityId>,
}

fn retained_lineage(entities: Vec<PersistentEntityId>) -> ProjectedLineage {
    ProjectedLineage {
        operations: entities
            .into_iter()
            .map(|entity| GeometryRevisionOperation::Retain {
                source: entity.clone(),
                target: entity,
            })
            .collect(),
        affected_before: Vec::new(),
        affected_after: Vec::new(),
    }
}

fn project_relations(
    relations: &[bridge::OcctHealingRelationPayload],
    imported: &ImportedExactCad,
) -> Result<ProjectedLineage, GeometryImportError> {
    if relations.is_empty() {
        return Err(invalid("OCCT sewing returned no persistent lineage"));
    }
    let target_inventory = topology_entities(imported)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut grouped = BTreeMap::<PersistentEntityId, Vec<PersistentEntityId>>::new();
    let mut deleted = Vec::new();
    let mut sources = BTreeSet::new();
    for relation in relations {
        if relation.path_segments.is_empty() || relation.path_segments[0] != ROOT_SCOPE {
            return Err(invalid(
                "OCCT healing relation has a noncanonical occurrence path",
            ));
        }
        let kind = match relation.kind {
            bridge::OcctHealingEntityKind::Vertex => PersistentEntityKind::Vertex,
            bridge::OcctHealingEntityKind::Edge => PersistentEntityKind::Edge,
            bridge::OcctHealingEntityKind::Face => PersistentEntityKind::Face,
            _ => return Err(invalid("OCCT healing relation has an invalid entity kind")),
        };
        let source = scoped_id(
            kind,
            &digest_name(&relation.source_digest)?,
            &relation.path_segments,
        );
        if !sources.insert(source.clone()) {
            return Err(invalid(
                "OCCT healing lineage contains a duplicate source identity",
            ));
        }
        if relation.target_digest.is_empty() {
            deleted.push(source);
            continue;
        }
        let target = scoped_id(
            kind,
            &digest_name(&relation.target_digest)?,
            &relation.path_segments,
        );
        if !target_inventory.contains(&target) {
            return Err(invalid("OCCT healing lineage targets absent topology"));
        }
        grouped.entry(target).or_default().push(source);
    }

    let mut ordered = Vec::new();
    for (target, mut source_group) in grouped {
        source_group.sort();
        let primary = source_group[0].clone();
        let operation = if source_group.len() > 1 {
            GeometryRevisionOperation::Merge {
                sources: source_group,
                target,
            }
        } else {
            let source = source_group.pop().expect("one source was checked");
            if source == target {
                GeometryRevisionOperation::Retain {
                    source: source.clone(),
                    target,
                }
            } else {
                GeometryRevisionOperation::Replace {
                    source: source.clone(),
                    target,
                }
            }
        };
        ordered.push((primary, operation));
    }
    ordered.extend(
        deleted
            .into_iter()
            .map(|source| (source.clone(), GeometryRevisionOperation::Delete { source })),
    );
    ordered.sort_by(|left, right| left.0.cmp(&right.0));
    let operations = ordered
        .into_iter()
        .map(|(_, operation)| operation)
        .collect::<Vec<_>>();
    let mut affected_before = sources.into_iter().collect::<Vec<_>>();
    affected_before.sort();
    let mut affected_after = operations
        .iter()
        .flat_map(revision_targets)
        .cloned()
        .collect::<Vec<_>>();
    affected_after.sort();
    affected_after.dedup();
    Ok(ProjectedLineage {
        operations,
        affected_before,
        affected_after,
    })
}

fn revision_targets(operation: &GeometryRevisionOperation) -> &[PersistentEntityId] {
    match operation {
        GeometryRevisionOperation::Retain { target, .. }
        | GeometryRevisionOperation::Replace { target, .. }
        | GeometryRevisionOperation::Merge { target, .. } => std::slice::from_ref(target),
        GeometryRevisionOperation::Split { targets, .. } => targets,
        GeometryRevisionOperation::Delete { .. } => &[],
    }
}

fn parse_digest(bytes: &[u8]) -> Result<GeometryDigest, GeometryImportError> {
    let digest: [u8; 32] = bytes.try_into().map_err(|_| {
        GeometryImportError::InvalidGeometry(
            "OCCT healing returned a malformed original geometry digest".into(),
        )
    })?;
    Ok(GeometryDigest::from_bytes(digest))
}

fn topology_entities(imported: &ImportedExactCad) -> Vec<PersistentEntityId> {
    let topology = &imported.topology;
    let mut entities = topology
        .assemblies
        .iter()
        .map(|value| value.id.clone())
        .chain(topology.instances.iter().map(|value| value.id.clone()))
        .chain(topology.bodies.iter().map(|value| value.id.clone()))
        .chain(topology.lumps.iter().map(|value| value.id.clone()))
        .chain(topology.solids.iter().map(|value| value.id.clone()))
        .chain(topology.shells.iter().map(|value| value.id.clone()))
        .chain(topology.faces.iter().map(|value| value.id.clone()))
        .chain(topology.wires.iter().map(|value| value.id.clone()))
        .chain(topology.coedges.iter().map(|value| value.id.clone()))
        .chain(topology.edges.iter().map(|value| value.id.clone()))
        .chain(topology.vertices.iter().map(|value| value.id.clone()))
        .collect::<Vec<_>>();
    entities.sort();
    entities
}

fn orientation_entities(imported: &ImportedExactCad) -> Vec<PersistentEntityId> {
    let mut entities = imported
        .topology
        .solids
        .iter()
        .map(|value| value.id.clone())
        .chain(
            imported
                .topology
                .shells
                .iter()
                .map(|value| value.id.clone()),
        )
        .chain(imported.topology.faces.iter().map(|value| value.id.clone()))
        .collect::<Vec<_>>();
    entities.sort();
    entities
}

fn duplicate_entities(imported: &ImportedExactCad) -> Vec<PersistentEntityId> {
    let topology = &imported.topology;
    let mut entities = topology
        .bodies
        .iter()
        .map(|value| value.id.clone())
        .chain(topology.lumps.iter().map(|value| value.id.clone()))
        .chain(topology.solids.iter().map(|value| value.id.clone()))
        .chain(topology.shells.iter().map(|value| value.id.clone()))
        .chain(topology.faces.iter().map(|value| value.id.clone()))
        .chain(topology.wires.iter().map(|value| value.id.clone()))
        .chain(topology.coedges.iter().map(|value| value.id.clone()))
        .chain(topology.edges.iter().map(|value| value.id.clone()))
        .chain(topology.vertices.iter().map(|value| value.id.clone()))
        .collect::<Vec<_>>();
    entities.sort();
    entities
}

fn contract_failure(error: runmat_geometry_core::GeometryContractError) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(format!("invalid OCCT healing evidence: {error}"))
}

fn invalid(reason: impl Into<String>) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(reason.into())
}
