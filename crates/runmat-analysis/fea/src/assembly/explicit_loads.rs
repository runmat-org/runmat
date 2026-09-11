use runmat_analysis_core::{StructuralElementKind, StructuralModel};

use super::{
    add_moment_projection_rhs, add_rhs,
    dofs::{StructuralDofKind, StructuralDofLayout},
    moment_projection::target_centroid,
    structural_target_nodes, MomentProjectionSummary, StructuralAssemblyError,
};
use crate::{
    excitation::{
        RealizedExcitationKind, RealizedExcitations, SpatialIntegrationRule,
        SpatialProjectionAccumulator, SpatialProjectionEvidence,
    },
    physics::structural::topology::{explicit_node_selector, explicit_topology_digest},
};

pub(super) struct ExplicitLoadProjection {
    pub direct_rotational_moment_count: usize,
    pub moment_projection: Vec<MomentProjectionSummary>,
    pub evidence: Vec<SpatialProjectionEvidence>,
}

pub(super) fn apply_explicit_structural_loads(
    structural: &StructuralModel,
    excitations: &RealizedExcitations,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
) -> Result<ExplicitLoadProjection, StructuralAssemblyError> {
    let projection_digest = explicit_topology_digest(structural)
        .map_err(|error| StructuralAssemblyError::ExplicitTopologyIdentity(error.to_string()))?;
    let node_coordinates_m = structural
        .nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let canonical_node_indices = (0..structural.nodes.len()).collect::<Vec<_>>();
    apply_projected_explicit_structural_loads(
        structural,
        excitations,
        layout,
        rhs,
        projection_digest,
        &node_coordinates_m,
        &canonical_node_indices,
    )
}

pub(super) fn apply_projected_explicit_structural_loads(
    structural: &StructuralModel,
    excitations: &RealizedExcitations,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    projection_digest: runmat_execution::Digest,
    canonical_node_coordinates_m: &[[f64; 3]],
    canonical_node_indices: &[usize],
) -> Result<ExplicitLoadProjection, StructuralAssemblyError> {
    if canonical_node_indices.len() != structural.nodes.len() {
        return Err(StructuralAssemblyError::InvalidCanonicalNodeMap {
            expected: structural.nodes.len(),
            actual: canonical_node_indices.len(),
        });
    }
    let mut direct_rotational_moment_count = 0;
    let mut moment_projection = Vec::new();
    let mut evidence = Vec::new();

    for load in excitations.iter() {
        if matches!(
            load.kind,
            RealizedExcitationKind::TemperatureField(_)
                | RealizedExcitationKind::TerminalVoltage(_)
        ) {
            continue;
        }
        let region_id = load.target.solver_region_id().ok_or_else(|| {
            StructuralAssemblyError::UnmappedStructuralLoad {
                load_id: load.excitation_id.0.clone(),
                region_id: "<non-region target>".to_string(),
                load_kind: super::explicit_load_kind(&load.kind),
            }
        })?;
        let explicit_target_nodes = structural_target_nodes(structural, region_id);
        if explicit_target_nodes.is_empty() {
            return Err(StructuralAssemblyError::UnmappedStructuralLoad {
                load_id: load.excitation_id.0.clone(),
                region_id: region_id.to_string(),
                load_kind: super::explicit_load_kind(&load.kind),
            });
        }
        let target_nodes = explicit_target_nodes
            .into_iter()
            .map(|node| canonical_node_indices[node])
            .collect::<Vec<_>>();
        let resolved_entity_count = explicit_target_entity_count(structural, region_id);
        let reference_point_m = target_centroid(canonical_node_coordinates_m, &target_nodes);
        let scale = 1.0 / target_nodes.len() as f64;
        match load.kind {
            RealizedExcitationKind::Force(value) => {
                let mut item = SpatialProjectionAccumulator::for_explicit_topology(
                    load,
                    projection_digest,
                    canonical_node_coordinates_m,
                    SpatialIntegrationRule::UniformResultant,
                    resolved_entity_count,
                    reference_point_m,
                );
                item.add_requested_force(reference_point_m, value.real);
                for &node_index in &target_nodes {
                    let nodal_force = value.real.map(|component| component * scale);
                    add_rhs(
                        layout,
                        rhs,
                        node_index,
                        StructuralDofKind::Ux,
                        nodal_force[0],
                    );
                    add_rhs(
                        layout,
                        rhs,
                        node_index,
                        StructuralDofKind::Uy,
                        nodal_force[1],
                    );
                    add_rhs(
                        layout,
                        rhs,
                        node_index,
                        StructuralDofKind::Uz,
                        nodal_force[2],
                    );
                    item.add_projected_force(node_index, nodal_force);
                }
                record_evidence(&mut evidence, item.finish())?;
            }
            RealizedExcitationKind::Moment(value) => {
                if target_nodes
                    .iter()
                    .all(|node_index| layout.index(*node_index, StructuralDofKind::Rx).is_some())
                {
                    direct_rotational_moment_count += 1;
                    for &node_index in &target_nodes {
                        add_rhs(
                            layout,
                            rhs,
                            node_index,
                            StructuralDofKind::Rx,
                            value.real[0] * scale,
                        );
                        add_rhs(
                            layout,
                            rhs,
                            node_index,
                            StructuralDofKind::Ry,
                            value.real[1] * scale,
                        );
                        add_rhs(
                            layout,
                            rhs,
                            node_index,
                            StructuralDofKind::Rz,
                            value.real[2] * scale,
                        );
                    }
                    let mut item = SpatialProjectionAccumulator::for_explicit_topology(
                        load,
                        projection_digest,
                        canonical_node_coordinates_m,
                        SpatialIntegrationRule::UniformResultant,
                        resolved_entity_count,
                        reference_point_m,
                    );
                    item.register_nodes(target_nodes.iter().copied());
                    item.add_requested_moment(value.real);
                    item.add_projected_moment(value.real);
                    record_evidence(&mut evidence, item.finish())?;
                } else {
                    let summary = add_moment_projection_rhs(
                        canonical_node_coordinates_m,
                        layout,
                        rhs,
                        &target_nodes,
                        value.real,
                    );
                    let mut item = SpatialProjectionAccumulator::for_explicit_topology(
                        load,
                        projection_digest,
                        canonical_node_coordinates_m,
                        SpatialIntegrationRule::UniformResultant,
                        resolved_entity_count,
                        summary.projection_point_m,
                    );
                    item.register_nodes(target_nodes.iter().copied());
                    item.add_requested_moment(value.real);
                    item.add_projected_moment(summary.applied_moment);
                    record_evidence(&mut evidence, item.finish())?;
                    moment_projection.push(MomentProjectionSummary {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        ..summary
                    });
                }
            }
            _ => unreachable!("explicit structural input validation rejects unsupported loads"),
        }
    }

    Ok(ExplicitLoadProjection {
        direct_rotational_moment_count,
        moment_projection,
        evidence,
    })
}

fn explicit_target_entity_count(structural: &StructuralModel, region_id: &str) -> usize {
    if explicit_node_selector(region_id).is_some() {
        return 1;
    }
    structural
        .elements
        .iter()
        .filter(|element| element.region_id == region_id)
        .filter(|element| {
            matches!(
                element.kind,
                StructuralElementKind::Beam(_) | StructuralElementKind::Shell(_)
            )
        })
        .count()
}

fn record_evidence(
    evidence: &mut Vec<SpatialProjectionEvidence>,
    item: SpatialProjectionEvidence,
) -> Result<(), StructuralAssemblyError> {
    item.validate().map_err(
        |error| StructuralAssemblyError::InvalidSpatialProjectionEvidence {
            load_id: item.excitation_id.clone(),
            reason: error.to_string(),
        },
    )?;
    if !item.accepted {
        let resultant_relative_error = item.resultant_relative_error();
        let moment_relative_error = item.moment_relative_error().unwrap_or(0.0);
        return Err(StructuralAssemblyError::ProjectionConservation {
            load_id: item.excitation_id,
            resultant_relative_error,
            moment_relative_error,
            relative_tolerance: item.requirements.relative_tolerance,
        });
    }
    evidence.push(item);
    Ok(())
}
