use super::*;
use crate::{
    GeometryRevisionIdentity, GeometryRevisionOperation, PersistentEntityKind,
    GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
};

fn entity(name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Edge,
        source_topology_id: name.into(),
        assembly_path: vec!["root".into()],
    }
}

fn valid() -> TopologyValidity {
    TopologyValidity {
        kernel_valid: true,
        incidence_consistent: true,
        orientation_consistent: true,
        shells_closed: true,
        nesting_consistent: true,
    }
}

fn invalid_orientation() -> TopologyValidity {
    TopologyValidity {
        orientation_consistent: false,
        ..valid()
    }
}

fn report() -> GeometryHealingReport {
    let original = GeometryDigest::from_bytes([1; 32]);
    let healed = GeometryDigest::from_bytes([2; 32]);
    let source = entity("edge-a");
    let target = entity("edge-b");
    GeometryHealingReport {
        schema_version: GEOMETRY_HEALING_REPORT_SCHEMA_VERSION,
        original_topology_digest: original,
        healed_topology_digest: healed,
        policy: GeometryHealingPolicy {
            algorithm_version: "occt-healing/1".into(),
            sew: true,
            repair_orientation: true,
            consolidate_duplicates: true,
            repair_tolerance_scale_gaps: true,
            simplify_short_edges_and_sliver_faces: true,
        },
        tolerance: crate::GeometryTolerancePolicy {
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-9,
            requested_deviation_m: 1.0e-5,
            maximum_healing_displacement_m: 1.0e-6,
        },
        revision_map: GeometryRevisionMap {
            schema_version: GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
            source_geometry_digest: original,
            source_revision: GeometryRevisionIdentity {
                revision: 1,
                persistent_mapping_version: 1,
                parent_document_digest: None,
            },
            target_geometry_digest: healed,
            target_revision: GeometryRevisionIdentity {
                revision: 2,
                persistent_mapping_version: 1,
                parent_document_digest: Some(original),
            },
            operations: vec![GeometryRevisionOperation::Replace {
                source: source.clone(),
                target: target.clone(),
            }],
        },
        original_validity: invalid_orientation(),
        healed_validity: valid(),
        operations: vec![GeometryHealingOperation {
            sequence: 0,
            kind: GeometryHealingOperationKind::RepairOrientation,
            affected_before: vec![source],
            affected_after: vec![target],
            maximum_displacement_m: 0.0,
            reason: "reverse inconsistent edge use".into(),
            before_validity: invalid_orientation(),
            after_validity: valid(),
        }],
    }
}

#[test]
fn successful_report_binds_topology_map_and_validity_chain() {
    let report = report();
    report.validate().unwrap();
    let encoded = serde_json::to_vec(&report).unwrap();
    assert_eq!(
        serde_json::from_slice::<GeometryHealingReport>(&encoded).unwrap(),
        report
    );
}

#[test]
fn disabled_or_over_limit_mutation_cannot_be_reported_as_success() {
    let mut disabled = report();
    disabled.policy.repair_orientation = false;
    assert!(disabled.validate().is_err());

    let mut displaced = report();
    displaced.tolerance.maximum_healing_displacement_m = 1.0e-6;
    displaced.operations[0].maximum_displacement_m = 2.0e-6;
    assert!(displaced.validate().is_err());
}

#[test]
fn report_requires_explicit_mapping_and_contiguous_validity() {
    let mut unmapped = report();
    unmapped.operations[0].affected_before[0] = entity("unmapped");
    assert!(unmapped.validate().is_err());

    let mut unproduced = report();
    unproduced.operations[0].affected_after[0] = entity("unproduced");
    assert!(unproduced.validate().is_err());

    let mut invalid_end = report();
    invalid_end.healed_validity.orientation_consistent = false;
    assert!(invalid_end.validate().is_err());
}

#[test]
fn limit_failure_names_entities_and_geometric_witness() {
    let failure = GeometryHealingFailure {
        operation: GeometryHealingOperationKind::RepairGap,
        affected_entities: vec![entity("edge-a")],
        measured_displacement_m: 2.0e-4,
        permitted_displacement_m: 1.0e-4,
        original_point_m: [0.0, 0.0, 0.0],
        proposed_point_m: [2.0e-4, 0.0, 0.0],
        reason: "gap repair exceeds model policy".into(),
    };
    failure.validate().unwrap();

    let mut not_exceeded = failure;
    not_exceeded.measured_displacement_m = not_exceeded.permitted_displacement_m;
    assert!(not_exceeded.validate().is_err());

    let mut inconsistent_witness = GeometryHealingFailure {
        operation: GeometryHealingOperationKind::RepairGap,
        affected_entities: vec![entity("edge-a")],
        measured_displacement_m: 2.0e-4,
        permitted_displacement_m: 1.0e-4,
        original_point_m: [0.0, 0.0, 0.0],
        proposed_point_m: [3.0e-4, 0.0, 0.0],
        reason: "gap repair exceeds model policy".into(),
    };
    assert!(inconsistent_witness.validate().is_err());
    inconsistent_witness.proposed_point_m[0] = 2.0e-4;
    inconsistent_witness.validate().unwrap();
}
