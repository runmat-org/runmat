use super::*;
use crate::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, InitialFieldBinding,
    ThermalSolverResultManifest,
};
use crate::results::field::{RecoveredField, RecoveredFieldArtifact, RecoveredFieldEntityCounts};
use runmat_analysis_core::{
    CoordinateFrame, InitialCondition, InitialConditionTarget, InitialConditionValue,
    InitialFieldArtifactRef, InitialValue, Temperature,
};

mod continuum;
mod nonuniform;

fn solved_temperature_binding(
    producer: &FeaThermalRunResult,
    mesh: &SolverMeshArtifact,
) -> InitialFieldBinding {
    let manifest = ThermalSolverResultManifest::from_result(producer).unwrap();
    let entry = manifest.terminal_temperature().unwrap();
    assert_eq!(
        producer.ordered_node_ids,
        mesh.topology
            .nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>()
    );
    let source = producer.source_result_identity().unwrap();
    let request = FieldRecoveryRequest::new(
        producer.producer,
        FieldRecoverySource::SolverResult(source),
        crate::execution::FieldTopologyIdentity::VolumeMesh {
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
            nodes: mesh.topology.nodes.len(),
            volume_elements: mesh.topology.volume_elements.len(),
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
        artifact_id: "solved-temperature".into(),
        content_digest: artifact.identity().digest().to_string(),
        topology_digest: artifact.topology_digest.to_string(),
        field_id: entry.definition.field_id().to_owned(),
    };
    InitialFieldBinding::new(reference, artifact).unwrap()
}

#[test]
fn cold_field_input_from_real_producer_preserves_insulated_temperature_and_lineage() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let producer = workload().execute(&mesh).unwrap();
        let binding = solved_temperature_binding(&producer, &mesh);
        let mut consumer = workload();
        consumer.model.initial_conditions =
            vec![InitialCondition::Temperature(InitialConditionValue::<
                Temperature,
            > {
                initial_condition_id: "imported-temperature".into(),
                target: InitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                value: InitialValue::FieldArtifact(binding.reference().clone()),
                applies_before_step_id: consumer.step_id.0.clone(),
            })];
        consumer.initial_fields = vec![binding];
        let identity = consumer.identity().unwrap();
        let cold =
            ThermalWorkload::from_canonical_bytes(&consumer.canonical_bytes().unwrap()).unwrap();
        assert_eq!(cold.identity().unwrap(), identity);
        let result = cold.execute(&mesh).unwrap();
        result.validate_complete().unwrap();
        assert_eq!(result.source_workload, identity);
        assert_eq!(
            result.initial_state.authored_node_count,
            mesh.topology.nodes.len()
        );
        // Independent insulated, source-free heat equation: constant T, zero gradient/flux.
        for field in &result.temperature_snapshots {
            assert!(field
                .as_host_f64()
                .unwrap()
                .iter()
                .all(|value| (*value - 293.15).abs() < 1e-9));
        }
        for field in result
            .temperature_gradient_snapshots
            .iter()
            .chain(&result.heat_flux_snapshots)
            .chain(&result.energy_storage_rate_snapshots)
        {
            assert!(field
                .as_host_f64()
                .unwrap()
                .iter()
                .all(|value| value.abs() < 1e-8));
        }
        let mut missing = cold.clone();
        missing.initial_fields.clear();
        assert!(missing.execute(&mesh).is_err());
        let mut duplicate = cold.clone();
        duplicate
            .initial_fields
            .push(duplicate.initial_fields[0].clone());
        assert!(duplicate.execute(&mesh).is_err());
        let mut unselected = cold.clone();
        unselected.model.initial_conditions.clear();
        assert!(unselected.execute(&mesh).is_err());
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let mut region = cold.clone();
        let InitialCondition::Temperature(condition) = &mut region.model.initial_conditions[0]
        else {
            unreachable!()
        };
        condition.target =
            InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::volume(
                projection.elements[0]
                    .region_id
                    .analysis_selection_id()
                    .unwrap(),
            ));
        assert_eq!(
            region
                .execute(&mesh)
                .unwrap()
                .initial_state
                .authored_node_count,
            mesh.topology.nodes.len()
        );
        let mut conflicting = cold.clone();
        conflicting
            .model
            .initial_conditions
            .push(InitialCondition::Temperature(InitialConditionValue {
                initial_condition_id: "conflicting-uniform".into(),
                target: InitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                value: InitialValue::Uniform(Temperature::new(310.0)),
                applies_before_step_id: conflicting.step_id.0.clone(),
            }));
        assert!(conflicting
            .execute(&mesh)
            .unwrap_err()
            .to_string()
            .contains("conflicts at solver node"));
        let mut prescribed = cold.clone();
        prescribed
            .model
            .constraints
            .push(runmat_analysis_core::AnalysisConstraint::Thermal(
                runmat_analysis_core::ThermalConstraint {
                    constraint_id: "conflicting-boundary".into(),
                    kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
                        target: runmat_analysis_core::RegionTarget::boundary(
                            projection.boundary_faces[0].provenance[0]
                                .analysis_selection_id()
                                .unwrap(),
                        ),
                        value: runmat_analysis_core::PrescribedValue::Constant(Temperature::new(
                            310.0,
                        )),
                    },
                },
            ));
        assert!(prescribed
            .execute(&mesh)
            .unwrap_err()
            .to_string()
            .contains("conflicts with prescribed temperature"));
        let mut aggregate_capped = cold.clone();
        aggregate_capped.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(
            crate::planning::resource::initial_field_reservation_bytes(
                &aggregate_capped.initial_fields,
                LIMITS,
            )
            .unwrap()
            .0,
        );
        assert!(
            aggregate_capped.validate().is_ok(),
            "field-only capacity must fit this fixture"
        );
        assert!(
            matches!(
                aggregate_capped.prepare(&mesh),
                Err(FeaRunError::Resource(
                    crate::planning::resource::ResourcePlanningError::AccuracyCeiling { .. }
                ))
            ),
            "combined model and field input must reject before owned preparation"
        );
        let mut capped = cold;
        capped.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(1);
        assert!(capped.prepare(&mesh).is_err());
    }
}
