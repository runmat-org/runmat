//! Projects successful native orientation repair into geometry-owned revision evidence.

use runmat_geometry_core::{
    encode_exact_topology, GeometryDigest, GeometryHealingOperation, GeometryHealingOperationKind,
    GeometryHealingReport, GeometryRevisionMap, GeometryRevisionOperation, GeometryTolerancePolicy,
    PersistentEntityId, TopologyValidity, GEOMETRY_HEALING_REPORT_SCHEMA_VERSION,
    GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

use crate::{exact::ImportedExactCad, import::GeometryImportError};

pub(super) fn healing_report(
    original_digest: &[u8],
    original_kernel_valid: bool,
    post_duplicate_kernel_valid: bool,
    duplicates_consolidated: bool,
    orientation_repaired: bool,
    imported: &ImportedExactCad,
) -> Result<GeometryHealingReport, GeometryImportError> {
    let original_topology_digest = parse_digest(original_digest)?;
    let topology_bytes =
        encode_exact_topology(&imported.topology, &imported.model).map_err(contract_failure)?;
    let healed_topology_digest = GeometryDigest::from_bytes(Sha256::digest(topology_bytes).into());
    let source_revision = imported.analysis.revision.clone();
    let mut target_revision = source_revision.clone();
    target_revision.revision = target_revision.revision.checked_add(1).ok_or_else(|| {
        GeometryImportError::InvalidOptions("exact geometry revision cannot advance".into())
    })?;
    target_revision.parent_document_digest = Some(original_topology_digest);

    let all_entities = topology_entities(imported);
    let revision_map = GeometryRevisionMap {
        schema_version: GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
        source_geometry_digest: original_topology_digest,
        source_revision,
        target_geometry_digest: healed_topology_digest,
        target_revision,
        operations: all_entities
            .into_iter()
            .map(|entity| GeometryRevisionOperation::Retain {
                source: entity.clone(),
                target: entity,
            })
            .collect(),
    };
    let original_validity = TopologyValidity {
        kernel_valid: original_kernel_valid,
        incidence_consistent: true,
        orientation_consistent: !orientation_repaired,
        shells_closed: true,
        nesting_consistent: !orientation_repaired,
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
    if duplicates_consolidated {
        let duplicate_validity = TopologyValidity {
            kernel_valid: post_duplicate_kernel_valid,
            incidence_consistent: true,
            orientation_consistent: !orientation_repaired,
            shells_closed: true,
            nesting_consistent: !orientation_repaired,
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
    if orientation_repaired {
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
