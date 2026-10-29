//! Original linear-conduction producer intent; execution is supplied by the normal host.
use super::*;
pub fn linear_profile() -> ThermalWorkload {
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
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_right",
                )),
                value: PrescribedValue::Constant(Temperature::new(390.0)),
            },
        }));
    input
}

/// Bind the original terminal temperature only; lineage does not certify its accuracy.
pub fn temperature_binding(
    manifest: &crate::execution::ThermalSolverResultManifest,
    field: crate::results::field::RecoveredField,
    mesh: &SolverMeshArtifact,
) -> crate::execution::InitialFieldBinding {
    use crate::execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
        InitialFieldBinding,
    };
    use crate::results::field::{RecoveredFieldArtifact, RecoveredFieldEntityCounts};
    manifest.validate().unwrap();
    let entry = manifest.terminal_temperature().unwrap();
    assert_eq!(field.identity(), entry.field);
    assert_eq!(
        manifest.ordered_node_ids,
        mesh.topology
            .nodes
            .iter()
            .map(|n| n.node_id)
            .collect::<Vec<_>>()
    );
    let source = manifest.identity();
    let request = FieldRecoveryRequest::new(
        manifest.producer,
        FieldRecoverySource::SolverResult(source),
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        FieldRecoverySource::SolverResult(source),
        manifest.topology_digest,
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
