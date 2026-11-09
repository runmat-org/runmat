//! Canonical worker input closure; not a process-restart or production qualification.
use super::*;
use runmat_analysis_core::{
    AnalysisBoundary, AnalysisConstraint, BoundaryOperator, CoordinateFrame, HeatFlux,
    InitialCondition, InitialConditionTarget, InitialConditionValue, InitialFieldArtifactRef,
    InitialValue, PrescribedValue, RegionTarget, ThermalConstraint, ThermalConstraintKind,
};
use runmat_analysis_fea::{
    execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, InitialFieldBinding,
        ThermalSolverResultManifest,
    },
    results::field::{RecoveredField, RecoveredFieldArtifact, RecoveredFieldEntityCounts},
};

#[test]
fn native_thermal_worker_consumes_source_bound_cold_initial_field() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let (mut workload, _) = task(order, 1024);
        let mut source_workload = workload.clone();
        source_workload.procedure = ThermalProcedureControl::Steady;
        source_workload.model.sequence =
            AnalysisSequence::from_steps("source", vec![ProcedureStep::thermal_steady("heat")]);
        let face_id = |index: usize| {
            mesh.topology.boundary_faces[index]
                .provenance
                .iter()
                .find(|entity| entity.kind == runmat_meshing_core::PersistentEntityKind::Face)
                .unwrap()
                .source_topology_id
                .clone()
        };
        source_workload.model.constraints = vec![AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "source-temperature".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(face_id(0))),
                ),
                value: PrescribedValue::Constant(Temperature::new(300.0)),
            },
        })];
        source_workload.model.boundaries = vec![AnalysisBoundary {
            boundary_id: "source-flux".into(),
            target: RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(face_id(1))),
            ),
            operator: BoundaryOperator::ThermalHeatFlux {
                value: PrescribedValue::Constant(HeatFlux::new(6.0)),
            },
        }];
        let producer = source_workload.execute(&mesh).unwrap();
        assert_eq!(
            producer.ordered_node_ids,
            mesh.topology
                .nodes
                .iter()
                .map(|node| node.node_id)
                .collect::<Vec<_>>()
        );
        let source_values = producer
            .temperature_snapshots
            .last()
            .unwrap()
            .as_host_f64()
            .unwrap();
        let minimum = source_values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = source_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            maximum - minimum > 1e-4,
            "worker fixture must expose nodal ordering errors"
        );
        let source = producer.source_result_identity().unwrap();
        let manifest = ThermalSolverResultManifest::from_result(&producer).unwrap();
        let entry = manifest.terminal_temperature().unwrap();
        let request = FieldRecoveryRequest::new(
            producer.producer,
            FieldRecoverySource::SolverResult(source),
            runmat_analysis_fea::execution::FieldTopologyIdentity::VolumeMesh {
                mesh_digest: mesh.canonical_digest,
            },
            FieldRecoveryPlan::default(),
        )
        .unwrap();
        let field = RecoveredField::new(
            entry.definition.clone(),
            entry.sample,
            producer.temperature_snapshots.last().unwrap().clone(),
        )
        .unwrap();
        let artifact = RecoveredFieldArtifact::new(
            request.identity(),
            FieldRecoverySource::SolverResult(source),
            producer.topology_digest,
            RecoveredFieldEntityCounts {
                oriented_faces: 0,
                nodes: producer.ordered_node_ids.len(),
                volume_elements: producer.ordered_element_ids.len(),
                volume_integration_points: 0,
                oriented_edges: 0,
                beam_elements: 0,
                shell_elements: 0,
                connectors: 0,
                constraint_groups: 0,
            },
            vec![field],
        )
        .unwrap();
        let reference = InitialFieldArtifactRef {
            artifact_id: "worker-source-temperature".into(),
            content_digest: artifact.identity().digest().to_string(),
            topology_digest: artifact.topology_digest.to_string(),
            field_id: entry.definition.field_id().to_owned(),
        };
        workload.model.initial_conditions =
            vec![InitialCondition::Temperature(InitialConditionValue {
                initial_condition_id: "cold-field".into(),
                target: InitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                value: InitialValue::FieldArtifact(reference.clone()),
                applies_before_step_id: "heat".into(),
            })];
        workload.initial_fields = vec![InitialFieldBinding::new(reference, artifact).unwrap()];
        let workload =
            ThermalWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap()).unwrap();
        // This comparison qualifies canonical worker transport, not independent physics.
        // The FEA test separately checks the nonuniform heat-equation reference.
        let expected = workload.execute(&mesh).unwrap();
        let digest = Digest::sha256("thermal-initial-field-native");
        let task = prepare_thermal_task(ThermalTaskPreparation {
            maximum_resident_bytes: u64::MAX,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            field_source: None,
            authored_study_digest: digest,
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: ArtifactAccess {
                authorization_scope: "study:thermal-field".into(),
                encryption_context: digest,
            },
            workload: workload.clone(),
            mesh,
            output_policy: policy(1024),
            required_capabilities: Default::default(),
        })
        .unwrap();
        let mut outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::SolveThermal,
                host: &task.host,
                inputs: &task.inputs,
                output_policy: policy(1024),
            })
            .unwrap()
            .into_iter();
        let result = outputs
            .next()
            .unwrap()
            .into_prepared(1024, ObjectInventoryLimits::default())
            .unwrap();
        let receipt = outputs
            .next()
            .unwrap()
            .into_prepared(1024, ObjectInventoryLimits::default())
            .unwrap();
        let manifest = decode_prepared_thermal_manifest(&result).unwrap();
        assert_eq!(manifest.source_workload, workload.identity().unwrap());
        assert_eq!(
            manifest.initial_state.authored_node_count,
            producer.ordered_node_ids.len()
        );
        ThermalNumericalReceiptArtifact::from_canonical_bytes(&receipt.payload())
            .unwrap()
            .validate_against(&workload, &manifest)
            .unwrap();
        for (ordinal, sample) in manifest.fields.iter().enumerate() {
            for (index, field) in sample.iter().enumerate() {
                let expected = [
                    &expected.temperature_snapshots,
                    &expected.temperature_gradient_snapshots,
                    &expected.heat_flux_snapshots,
                    &expected.energy_storage_rate_snapshots,
                ][index][ordinal]
                    .as_host_f64()
                    .unwrap();
                let mut actual = Vec::new();
                for descriptor in &field.partitions {
                    let partition = RecoveredFieldPartition::from_canonical_bytes(
                        &result
                            .partition_payload(descriptor.partition.digest())
                            .unwrap(),
                    )
                    .unwrap();
                    actual.extend_from_slice(partition.storage.as_host_f64().unwrap());
                }
                assert_eq!(actual.len(), expected.len());
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| (actual - expected).abs() < 1e-8));
            }
        }
    }
}
