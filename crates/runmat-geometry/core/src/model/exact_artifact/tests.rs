use super::*;
use crate::{
    model::{
        exact_evaluator::tests::registry,
        exact_topology_tests::{model, topology},
    },
    ExactBRepModel, GeometryDocument, GeometryHealingOperation, GeometryHealingOperationKind,
    GeometryHealingPolicy, GeometryHealingReport, GeometryModel, GeometryObjectRef,
    GeometryRevisionIdentity, GeometryRevisionMap, GeometryRevisionOperation, GeometrySourceFormat,
    GeometrySourceIdentity, GeometryTolerancePolicy, TopologyValidity, UnitSystem,
    EXACT_BREP_MEDIA_TYPE, GEOMETRY_DOCUMENT_SCHEMA_VERSION,
    GEOMETRY_HEALING_REPORT_SCHEMA_VERSION, GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
    GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
};

fn object(bytes: &[u8], media_type: &str) -> GeometryObjectRef {
    GeometryObjectRef {
        digest: super::codec::digest(bytes).unwrap(),
        encoded_length: bytes.len() as u64,
        media_type: media_type.into(),
        schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
    }
}

fn document(exact_model: ExactBRepModel) -> GeometryDocument {
    GeometryDocument {
        schema_version: GEOMETRY_DOCUMENT_SCHEMA_VERSION,
        source: GeometrySourceIdentity {
            content_digest: crate::GeometryDigest::from_bytes([7; 32]),
            format: GeometrySourceFormat::Step,
            importer_version: "step-import/3".into(),
            kernel_version: Some("occt/7.9".into()),
            source_units: UnitSystem::Meter,
            meters_per_source_unit: 1.0,
        },
        revision: GeometryRevisionIdentity {
            revision: 1,
            persistent_mapping_version: 1,
            parent_document_digest: None,
        },
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-9,
            requested_deviation_m: 1.0e-5,
            maximum_healing_displacement_m: 1.0e-6,
        },
        healing: GeometryHealingPolicy {
            algorithm_version: "occt-healing/1".into(),
            sew: true,
            repair_orientation: true,
            consolidate_duplicates: true,
            repair_tolerance_scale_gaps: true,
            simplify_short_edges_and_sliver_faces: false,
        },
        model: GeometryModel::ExactBRep { model: exact_model },
        display_tessellations: Vec::new(),
    }
}

fn closure() -> (GeometryDocument, Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut exact_model = model();
    let topology = topology();
    let evaluators = registry();
    let topology_bytes = encode_exact_topology(&topology, &exact_model).unwrap();
    let evaluator_bytes = encode_exact_evaluators(&evaluators, &topology, &exact_model).unwrap();
    let parent = crate::GeometryDigest::from_bytes([9; 32]);
    let revision = GeometryRevisionIdentity {
        revision: 2,
        persistent_mapping_version: 1,
        parent_document_digest: Some(parent),
    };
    let manifest = ExactGeometryManifest {
        schema_version: EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION,
        source_digest: crate::GeometryDigest::from_bytes([7; 32]),
        revision: revision.clone(),
        kernel_abi: exact_model.kernel_abi.clone(),
        topology: object(&topology_bytes, EXACT_TOPOLOGY_MEDIA_TYPE),
        evaluators: object(&evaluator_bytes, EXACT_EVALUATOR_MEDIA_TYPE),
        healing_report: None,
    };
    let manifest_bytes = manifest.canonical_encode().unwrap();
    exact_model.artifact = object(&manifest_bytes, EXACT_BREP_MEDIA_TYPE);
    let mut document = document(exact_model);
    document.revision = revision;
    (document, manifest_bytes, topology_bytes, evaluator_bytes)
}

#[test]
fn optional_healing_evidence_must_produce_the_admitted_topology() {
    let (mut document, _, topology_bytes, evaluator_bytes) = closure();
    let GeometryModel::ExactBRep { model } = &document.model else {
        unreachable!()
    };
    let healed_topology = decode_exact_topology(&topology_bytes, model).unwrap();
    let edge = healed_topology.edges[0].id.clone();
    let original_digest = crate::GeometryDigest::from_bytes([9; 32]);
    let healed_digest = object(&topology_bytes, EXACT_TOPOLOGY_MEDIA_TYPE).digest;
    let valid = TopologyValidity {
        kernel_valid: true,
        incidence_consistent: true,
        orientation_consistent: true,
        shells_closed: true,
        nesting_consistent: true,
    };
    let before = TopologyValidity {
        orientation_consistent: false,
        ..valid
    };
    let report = GeometryHealingReport {
        schema_version: GEOMETRY_HEALING_REPORT_SCHEMA_VERSION,
        original_topology_digest: original_digest,
        healed_topology_digest: healed_digest,
        policy: document.healing.clone(),
        tolerance: document.tolerance,
        revision_map: GeometryRevisionMap {
            schema_version: GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
            source_geometry_digest: original_digest,
            source_revision: GeometryRevisionIdentity {
                revision: 1,
                persistent_mapping_version: 1,
                parent_document_digest: None,
            },
            target_geometry_digest: healed_digest,
            target_revision: document.revision.clone(),
            operations: vec![GeometryRevisionOperation::Retain {
                source: edge.clone(),
                target: edge.clone(),
            }],
        },
        original_validity: before,
        healed_validity: valid,
        operations: vec![GeometryHealingOperation {
            sequence: 0,
            kind: GeometryHealingOperationKind::RepairOrientation,
            affected_before: vec![edge.clone()],
            affected_after: vec![edge],
            maximum_displacement_m: 0.0,
            reason: "repair inconsistent edge-use orientation".into(),
            before_validity: before,
            after_validity: valid,
        }],
    };
    let healing_bytes = encode_geometry_healing_report(&report).unwrap();
    let mut manifest = ExactGeometryManifest {
        schema_version: EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION,
        source_digest: document.source.content_digest,
        revision: document.revision.clone(),
        kernel_abi: model.kernel_abi.clone(),
        topology: object(&topology_bytes, EXACT_TOPOLOGY_MEDIA_TYPE),
        evaluators: object(&evaluator_bytes, EXACT_EVALUATOR_MEDIA_TYPE),
        healing_report: Some(object(&healing_bytes, GEOMETRY_HEALING_MEDIA_TYPE)),
    };
    let manifest_bytes = manifest.canonical_encode().unwrap();
    let GeometryModel::ExactBRep { model } = &mut document.model else {
        unreachable!()
    };
    model.artifact = object(&manifest_bytes, EXACT_BREP_MEDIA_TYPE);
    let admitted = admit_exact_geometry_closure(
        &document,
        &manifest_bytes,
        &topology_bytes,
        &evaluator_bytes,
        Some(&healing_bytes),
    )
    .unwrap();
    assert_eq!(admitted.healing_report, Some(report));

    manifest.revision.revision += 1;
    assert!(manifest.validate_against_document(&document).is_err());
}

#[test]
fn exact_geometry_closure_round_trips_and_is_independently_admitted() {
    let (document, manifest, topology, evaluators) = closure();
    let admitted =
        admit_exact_geometry_closure(&document, &manifest, &topology, &evaluators, None).unwrap();
    assert_eq!(
        admitted.topology,
        crate::model::exact_topology_tests::topology()
    );
    assert_eq!(admitted.evaluators, registry());
    assert_eq!(admitted.manifest.canonical_encode().unwrap(), manifest);
}

#[test]
fn closure_rejects_corruption_missing_components_and_identity_drift() {
    let (document, manifest, mut topology, evaluators) = closure();
    let corrupt_index = topology.len() / 2;
    topology[corrupt_index] ^= 1;
    assert!(
        admit_exact_geometry_closure(&document, &manifest, &topology, &evaluators, None,).is_err()
    );

    let (mut document, manifest, topology, evaluators) = closure();
    document.source.content_digest = crate::GeometryDigest::from_bytes([8; 32]);
    assert!(
        admit_exact_geometry_closure(&document, &manifest, &topology, &evaluators, None,).is_err()
    );
}

#[test]
fn component_decoders_reject_wrong_domains_and_trailing_bytes() {
    let (document, _, topology_bytes, evaluator_bytes) = closure();
    let GeometryModel::ExactBRep { model } = &document.model else {
        unreachable!()
    };
    assert!(decode_exact_evaluators(&topology_bytes, &topology(), model).is_err());
    assert!(decode_exact_topology(&evaluator_bytes, model).is_err());

    let mut trailing = topology_bytes;
    trailing.push(0);
    assert!(decode_exact_topology(&trailing, model).is_err());
}

#[test]
fn manifest_rejects_duplicate_component_identity_and_unknown_schema() {
    let (_, manifest_bytes, _, _) = closure();
    let mut manifest = ExactGeometryManifest::canonical_decode(&manifest_bytes).unwrap();
    manifest.evaluators.digest = manifest.topology.digest;
    assert!(manifest.validate().is_err());

    manifest.evaluators.digest = crate::GeometryDigest::from_bytes([4; 32]);
    manifest.schema_version += 1;
    assert!(manifest.canonical_encode().is_err());
}
