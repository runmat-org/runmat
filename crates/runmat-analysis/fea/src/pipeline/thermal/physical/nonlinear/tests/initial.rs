//! Genuine linear thermal producer supplies the manufactured continuum's initial state.
use super::*;
use crate::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
    InitialFieldBinding, ThermalSolverResultManifest,
};
use crate::results::field::{RecoveredField, RecoveredFieldArtifact, RecoveredFieldEntityCounts};

pub(super) fn solved_linear_profile(mesh: &SolverMeshArtifact) -> InitialFieldBinding {
    let mut input = workload();
    input
        .model
        .boundaries
        .retain(|boundary| !matches!(boundary.operator, BoundaryOperator::ThermalRadiation { .. }));
    left_temperature(
        &mut input,
        PrescribedValue::Constant(Temperature::new(400.0)),
    );
    input
        .model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "fixed-right-initial-producer".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary("slab_right"),
                value: PrescribedValue::Constant(Temperature::new(390.0)),
            },
        }));
    let result = input.execute(mesh).unwrap();
    for (node, &value) in mesh
        .topology
        .nodes
        .iter()
        .zip(result.temperature_snapshots[0].as_host_f64().unwrap())
    {
        assert!((value - (400.0 - 10.0 * node.coordinates_m[0])).abs() < 1.0e-7);
    }
    let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
    let entry = manifest.terminal_temperature().unwrap();
    let source = result.source_result_identity().unwrap();
    let request = FieldRecoveryRequest::new(
        result.producer,
        FieldRecoverySource::SolverResult(source),
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let field = RecoveredField::new(
        entry.definition.clone(),
        entry.sample,
        result.temperature_snapshots[0].clone(),
    )
    .unwrap();
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        FieldRecoverySource::SolverResult(source),
        result.topology_digest,
        RecoveredFieldEntityCounts {
            nodes: mesh.topology.nodes.len(),
            volume_elements: mesh.topology.volume_elements.len(),
            oriented_faces: 0,
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
        artifact_id: "original-linear-temperature".into(),
        content_digest: artifact.identity().digest().to_string(),
        topology_digest: artifact.topology_digest.to_string(),
        field_id: entry.definition.field_id().to_owned(),
    };
    InitialFieldBinding::new(reference, artifact).unwrap()
}
