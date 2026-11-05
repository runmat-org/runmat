use super::*;
use runmat_analysis_core::AnalysisModel;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    physics::structural::topology::target::MixedStructuralTargetOwnership,
    topology::AnalysisProjection,
};

/// A model connector plan projects a supplied solid mesh once. Empty plans
/// have no targets to resolve and perform no numerical projection.
pub(crate) fn connector_plan_projection_work_upper_bound(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<runmat_simulation_usage::OperationCount, crate::topology::AnalysisProjectionError> {
    if model.connectors.is_empty() {
        return Ok(runmat_simulation_usage::OperationCount::ZERO);
    }
    mesh.map(AnalysisProjection::derivation_work_upper_bound)
        .transpose()
        .map(|work| work.unwrap_or(runmat_simulation_usage::OperationCount::ZERO))
}

/// Compile the same canonical connector realization used by numerical assembly without
/// constructing sparse operators. Observation and resource admission may call this before solve.
pub(crate) fn compile_model_connector_plan(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    reference_point_count: usize,
) -> Result<CanonicalStructuralConnectorPlan, StructuralAssemblyError> {
    let empty_structural = StructuralModel::default();
    let structural = model.structural.as_ref().unwrap_or(&empty_structural);
    let topology = match (mesh, model.structural.as_ref()) {
        (Some(mesh), Some(structural)) => CanonicalStructuralNodeSpace::derive(mesh, structural),
        (Some(mesh), None) => CanonicalStructuralNodeSpace::derive_solid(mesh),
        (None, Some(structural)) => CanonicalStructuralNodeSpace::derive_explicit(structural),
        (None, None) => return Err(StructuralAssemblyError::MissingDiscretization),
    }
    .map_err(|error| StructuralAssemblyError::MixedTopology(error.to_string()))?;
    if model.connectors.is_empty() {
        return CanonicalStructuralConnectorPlan::empty(topology.identity, topology.nodes.len())
            .map_err(|_| StructuralAssemblyError::InvalidExplicitConnector {
                connector_id: "connector-plan".into(),
                reason: "canonical connector realization is invalid",
            });
    }
    let projection =
        mesh.map(AnalysisProjection::derive)
            .transpose()
            .map_err(|error| match error {
                crate::topology::AnalysisProjectionError::Cancelled => {
                    StructuralAssemblyError::Cancelled
                }
                crate::topology::AnalysisProjectionError::Accounting(cause) => {
                    StructuralAssemblyError::Accounting(cause)
                }
                other => StructuralAssemblyError::MixedTopology(other.to_string()),
            })?;
    let layout = super::super::mixed_dof_layout(&topology, reference_point_count);
    let coordinates_m = topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect::<Vec<_>>();
    let node_identities = topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<Vec<_>>();
    let compiled = compile_connectors_with_resolver(
        &model.connectors,
        ConnectorAssemblyContext {
            topology_identity: topology.identity,
            layout: &layout,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(&coordinates_m),
            node_identities: &node_identities,
        },
        |connector, target, role| {
            let Some(mesh) = mesh else {
                return match role {
                    ConnectorTargetRole::Point => resolve_point_target(
                        connector,
                        target,
                        structural_target_nodes(structural, &target.region_id)?,
                    ),
                    ConnectorTargetRole::Surface => {
                        resolve_explicit_surface(connector, target, structural, &topology)
                    }
                };
            };
            let ownership = super::super::mixed_structural::resolve_target(
                &connector.connector_id,
                target,
                mesh,
                structural,
                &topology,
            )?;
            match role {
                ConnectorTargetRole::Point => Ok(ResolvedConnectorTarget {
                    node_indices: ownership.canonical_nodes().to_vec(),
                    surface_samples: Vec::new(),
                }),
                ConnectorTargetRole::Surface => match ownership {
                    MixedStructuralTargetOwnership::Solid { .. }
                    | MixedStructuralTargetOwnership::Shared { .. } => resolve_projected_surface(
                        connector,
                        target,
                        projection.as_ref().expect("mesh owns a projection"),
                    ),
                    MixedStructuralTargetOwnership::Explicit { .. } => {
                        resolve_explicit_surface(connector, target, structural, &topology)
                    }
                },
            }
        },
    )?;
    Ok(compiled.plan)
}

pub(crate) fn resolve_projected_surface(
    connector: &AnalysisConnector,
    target: &RegionTarget,
    projection: &AnalysisProjection,
) -> Result<ResolvedConnectorTarget, StructuralAssemblyError> {
    let faces = projection
        .boundary_faces
        .iter()
        .filter(|face| {
            face.provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
        })
        .collect::<Vec<_>>();
    if faces.is_empty() {
        return Err(StructuralAssemblyError::UnmappedExplicitConnector {
            connector_id: connector.connector_id.clone(),
            target: "surface",
            region_id: target.region_id.clone(),
        });
    }
    let mut nodes = Vec::new();
    let mut samples = Vec::new();
    for face in faces {
        nodes.extend_from_slice(&face.node_indices);
        projection
            .visit_boundary_quadrature(face, |point| {
                let normal = unit_vector(point.area_vector_weight_m2)
                    .expect("validated boundary quadrature owns a nonzero normal");
                samples.push(SurfaceFoundationSample {
                    node_indices: face.node_indices.clone(),
                    shape_values: point.shape_values().to_vec(),
                    unit_normal: normal,
                    area_weight_m2: point.area_weight_m2,
                });
            })
            .map_err(|_| invalid(connector, "surface foundation face quadrature is invalid"))?;
    }
    nodes.sort_unstable();
    nodes.dedup();
    Ok(ResolvedConnectorTarget {
        node_indices: nodes,
        surface_samples: samples,
    })
}
