use runmat_analysis_core::{AnalysisFieldLayout};
use super::*;
use crate::{ComputeBackend, ThermalProcedureControl};
use runmat_analysis_core::{
    AnalysisSequence, ProcedureStep, ProcedureStepId, Temperature, ThermalDomain,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> ThermalWorkload {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_transient("heat_a")]);
    ThermalWorkload {
        initial_fields: Vec::new(),
        model,
        step_id: ProcedureStepId("heat_a".into()),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

#[test]
fn structural_selection_binds_producer_through_recovery_request() {
    use crate::execution::{FieldRecoveryPlan, SequenceIdentity};
    use crate::results::field::{
        RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    };
    use runmat_analysis_core::{
        AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
    };
    let sequence = SequenceIdentity::new(Digest::sha256("selection sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("producer")).unwrap();
    let other_producer = StepIdentity::new(sequence, Digest::sha256("other producer")).unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("result")).unwrap();
    let topology = CanonicalStructuralTopologyIdentity::Explicit {
        structural_topology_digest: Digest::sha256("explicit topology"),
    };
    let request = FieldRecoveryRequest::new(
        producer,
        FieldRecoverySource::SolverResult(result),
        crate::execution::FieldTopologyIdentity::Structural { topology },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let field = RecoveredField::new(
        RecoveredFieldDefinition::new(
            "displacement",
            ObservationQuantity::Displacement,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64("displacement", AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node), vec![1, 3], vec![0.0; 3]),
    )
    .unwrap();
    let manifest = RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        topology.discretization_digest(),
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: 1,
            volume_elements: 0,
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 1,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field.clone()],
    )
    .unwrap()
    .manifest()
    .unwrap();
    let selection =
        ObservationFieldSelection::structural(&request, &manifest, vec![field.identity()]).unwrap();
    selection.validate_structural(&request, &manifest).unwrap();
    assert_eq!(selection.producer(), producer);
    let volume_request = FieldRecoveryRequest::new(
        producer,
        request.source,
        crate::execution::FieldTopologyIdentity::VolumeMesh {
            mesh_digest: StableDigest::from_bytes(*topology.discretization_digest().bytes()),
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    // Matching digest and authentic catalog do not grant structural family authority.
    let volume_manifest = RecoveredFieldArtifact::new(
        volume_request.identity(),
        volume_request.source,
        manifest.topology_digest,
        manifest.entity_counts,
        vec![field.clone()],
    )
    .unwrap()
    .manifest()
    .unwrap();
    assert_eq!(
        volume_request.topology.discretization_digest(),
        topology.discretization_digest()
    );
    assert!(ObservationFieldSelection::structural(
        &volume_request,
        &volume_manifest,
        vec![field.identity()],
    )
    .unwrap_err()
    .to_string()
    .contains("structural topology authority"));
    let foreign_request = FieldRecoveryRequest::new(
        other_producer,
        request.source,
        crate::execution::FieldTopologyIdentity::Structural { topology },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    assert!(ObservationFieldSelection::structural(
        &foreign_request,
        &manifest,
        vec![field.identity()]
    )
    .is_err());
    assert!(selection
        .validate_structural(&foreign_request, &manifest)
        .is_err());
    let mut forged = selection.clone();
    forged.producer = other_producer;
    forged.identity = forged.derive_identity().unwrap();
    assert!(forged.validate_structural(&request, &manifest).is_err());
}

#[test]
fn thermal_selection_roundtrip_and_authority_substitution() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let workload = workload();
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let result = workload.execute(&mesh).unwrap();
        let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
        let selected = manifest.fields[1].iter().map(|entry| entry.field).collect();
        let selection =
            ObservationFieldSelection::thermal(&workload, &manifest, &mesh, 1, selected).unwrap();
        assert_eq!(selection.fields().len(), 4);
        let decoded =
            ObservationFieldSelection::from_canonical_bytes(&selection.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(decoded, selection);
        decoded
            .validate_thermal(&workload, &manifest, &mesh)
            .unwrap();
        let partitions_at = |ordinal: usize| {
            let entry = &manifest.fields[ordinal][0];
            let view = crate::results::field::RecoveredFieldView::new(
                &entry.definition,
                entry.sample,
                &result.temperature_snapshots[ordinal],
            )
            .unwrap();
            RecoveredFieldPartitionSet::new(
                view.canonical_partitions()
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
            )
            .unwrap()
        };
        decoded.validate_partitions(&partitions_at(1)).unwrap();
        assert!(decoded.validate_partitions(&partitions_at(0)).is_err());
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        assert_eq!(
            decoded.source().observation_topology_digest(),
            projection.projection_digest
        );
        assert_ne!(projection.projection_digest, manifest.topology_digest);
        let mut foreign = workload.clone();
        foreign.residual_target *= 0.1;
        assert!(decoded
            .validate_thermal(&foreign, &manifest, &mesh)
            .is_err());
        let foreign_mesh = canonical_tetrahedron_solver_mesh(if order == ElementOrder::Tet4 {
            ElementOrder::Tet10
        } else {
            ElementOrder::Tet4
        });
        assert!(decoded
            .validate_thermal(&workload, &manifest, &foreign_mesh)
            .is_err());
        assert!(ObservationFieldSelection::thermal(
            &workload,
            &manifest,
            &mesh,
            0,
            vec![manifest.fields[1][0].field]
        )
        .is_err());
        assert!(ObservationFieldSelection::thermal(
            &workload,
            &manifest,
            &mesh,
            1,
            vec![manifest.fields[1][0].field; 2]
        )
        .is_err());
        let mut tampered = selection.clone();
        tampered.fields[0].partitions[0].entity_count += 1;
        let bytes =
            encode_contract(PREFIX, "observation-field-selection", &tampered, LIMITS).unwrap();
        assert!(ObservationFieldSelection::from_canonical_bytes(&bytes).is_err());
        // Even a self-consistent new digest cannot establish membership in the producing catalog.
        tampered.identity = tampered.derive_identity().unwrap();
        assert!(tampered
            .validate_thermal(&workload, &manifest, &mesh)
            .is_err());
        let mut forged_projection = selection.clone();
        if let ObservationFieldSource::ThermalResult {
            projection_digest, ..
        } = &mut forged_projection.source
        {
            *projection_digest = Digest::sha256("foreign projection");
        }
        forged_projection.identity = forged_projection.derive_identity().unwrap();
        assert!(forged_projection
            .validate_thermal(&workload, &manifest, &mesh)
            .is_err());
    }
}

#[test]
fn one_thermal_authority_preserves_every_original_sample_and_rejects_foreign_history() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        workload.procedure = crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 17 };
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let result = workload.execute(&mesh).unwrap();
        let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
        let authority = ThermalFieldSourceAuthority::new(&workload, &manifest, &mesh).unwrap();
        let bound = authority.construction_resident_bytes();
        let planned = thermal::resources::construction_resident_bytes_admitted(
            &workload,
            &manifest,
            &mesh,
            u64::MAX,
        )
        .unwrap();
        assert!(planned >= bound);
        assert_eq!(
            thermal::resources::construction_resident_bytes_admitted(
                &workload, &manifest, &mesh, planned
            )
            .unwrap(),
            planned
        );
        assert!(thermal::resources::construction_resident_bytes_admitted(
            &workload,
            &manifest,
            &mesh,
            planned - 1
        )
        .is_err());
        let admitted =
            ThermalFieldSourceAuthority::new_admitted(&workload, &manifest, &mesh, bound).unwrap();
        assert!(
            ThermalFieldSourceAuthority::new_admitted(&workload, &manifest, &mesh, bound - 1)
                .is_err()
        );
        assert!(bound >= AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap());
        for (ordinal, entries) in manifest.fields.iter().enumerate() {
            let fields = entries.iter().map(|entry| entry.field).collect::<Vec<_>>();
            let selected = authority.selection(ordinal, fields.clone()).unwrap();
            assert_eq!(
                selected,
                ObservationFieldSelection::thermal(&workload, &manifest, &mesh, ordinal, fields,)
                    .unwrap()
            );
            authority.validate_selection(&selected).unwrap();
            assert_eq!(
                admitted
                    .selection(ordinal, entries.iter().map(|entry| entry.field).collect())
                    .unwrap(),
                selected
            );
            let mut forged = selected.clone();
            if let ObservationFieldSource::ThermalResult { sample_ordinal, .. } = &mut forged.source
            {
                *sample_ordinal = (ordinal + 1) % manifest.fields.len();
            }
            forged.identity = forged.derive_identity().unwrap();
            assert!(authority.validate_selection(&forged).is_err());
        }
        let mut different_schedule = workload.clone();
        different_schedule.procedure =
            crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 16 };
        assert!(ThermalFieldSourceAuthority::new(&different_schedule, &manifest, &mesh).is_err());
        assert!(authority.selection(manifest.fields.len(), vec![]).is_err());
    }
}
