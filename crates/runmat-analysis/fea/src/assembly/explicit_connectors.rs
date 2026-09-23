use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisConnector, ConnectorKind, RegionTarget, StructuralModel};

use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    structural_target_nodes, StructuralAssemblyError,
};
use crate::{
    algebra::{ContributionDomain, ContributionId, ContributionProvenance, RealSparseAssembler},
    connectors::{
        authored_connectors_digest, CanonicalConnectorEntry, CanonicalConnectorNode,
        CanonicalStructuralConnectorPlan, SurfaceFoundationQuadrature,
    },
    physics::structural::topology::{
        explicit_surface::explicit_shell_surface_geometry, node_space::CanonicalStructuralNodeSpace,
    },
};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ExplicitConnectorSummary {
    pub stiffness_count: usize,
    pub damping_count: usize,
    pub plan: CanonicalStructuralConnectorPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConnectorTargetRole {
    Point,
    Surface,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SurfaceFoundationSample {
    pub node_indices: Vec<usize>,
    pub shape_values: Vec<f64>,
    pub unit_normal: [f64; 3],
    pub area_weight_m2: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ResolvedConnectorTarget {
    pub node_indices: Vec<usize>,
    pub surface_samples: Vec<SurfaceFoundationSample>,
}

pub(super) struct ConnectorAssemblyContext<'a> {
    pub topology_identity:
        crate::physics::structural::topology::CanonicalStructuralTopologyIdentity,
    pub layout: &'a StructuralDofLayout,
    pub coordinates_m: &'a [[f64; 3]],
    pub node_identities: &'a [runmat_meshing_core::StableDigest],
}

pub(super) fn assemble_explicit_connectors(
    connectors: &[AnalysisConnector],
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    layout: &StructuralDofLayout,
    stiffness: &mut RealSparseAssembler,
    damping: &mut RealSparseAssembler,
) -> Result<ExplicitConnectorSummary, StructuralAssemblyError> {
    let coordinates_m = topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let node_identities = topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<Vec<_>>();
    assemble_connectors_with_resolver(
        connectors,
        ConnectorAssemblyContext {
            topology_identity: topology.identity,
            layout,
            coordinates_m: &coordinates_m,
            node_identities: &node_identities,
        },
        stiffness,
        damping,
        |connector, target, role| match role {
            ConnectorTargetRole::Point => resolve_point_target(
                connector,
                target,
                structural_target_nodes(structural, &target.region_id),
            ),
            ConnectorTargetRole::Surface => {
                resolve_explicit_surface(connector, target, structural, topology)
            }
        },
    )
}

pub(super) fn assemble_connectors_with_resolver(
    connectors: &[AnalysisConnector],
    context: ConnectorAssemblyContext<'_>,
    stiffness: &mut RealSparseAssembler,
    damping: &mut RealSparseAssembler,
    mut resolve: impl FnMut(
        &AnalysisConnector,
        &RegionTarget,
        ConnectorTargetRole,
    ) -> Result<ResolvedConnectorTarget, StructuralAssemblyError>,
) -> Result<ExplicitConnectorSummary, StructuralAssemblyError> {
    let mut stiffness_count = 0;
    let mut damping_count = 0;
    let mut entries = Vec::with_capacity(connectors.len());
    for connector in connectors {
        let source_digest = crate::connectors::contract::source_digest(connector)
            .map_err(|_| invalid(connector, "connector source identity could not be encoded"))?;
        match &connector.kind {
            ConnectorKind::AxialSpring {
                primary,
                secondary,
                stiffness: value,
            } => {
                let endpoints =
                    resolve_axial_endpoints(connector, primary, secondary, &mut resolve)?;
                add_axial_operator(
                    stiffness,
                    context.layout,
                    context.coordinates_m,
                    endpoints,
                    value.newtons_per_meter,
                    connector,
                )?;
                stiffness.add_contribution(provenance(connector));
                entries.push(CanonicalConnectorEntry::AxialSpring {
                    connector_id: connector.connector_id.clone(),
                    source_digest,
                    primary: canonical_node(endpoints[0], context.node_identities, connector)?,
                    secondary: canonical_node(endpoints[1], context.node_identities, connector)?,
                    unit_axis: axial_axis(context.coordinates_m, endpoints, connector)?,
                    stiffness_newtons_per_meter: value.newtons_per_meter,
                });
                stiffness_count += 1;
            }
            ConnectorKind::AxialDashpot {
                primary,
                secondary,
                damping: value,
            } => {
                let endpoints =
                    resolve_axial_endpoints(connector, primary, secondary, &mut resolve)?;
                add_axial_operator(
                    damping,
                    context.layout,
                    context.coordinates_m,
                    endpoints,
                    value.newton_seconds_per_meter,
                    connector,
                )?;
                damping.add_contribution(provenance(connector));
                entries.push(CanonicalConnectorEntry::AxialDashpot {
                    connector_id: connector.connector_id.clone(),
                    source_digest,
                    primary: canonical_node(endpoints[0], context.node_identities, connector)?,
                    secondary: canonical_node(endpoints[1], context.node_identities, connector)?,
                    unit_axis: axial_axis(context.coordinates_m, endpoints, connector)?,
                    damping_newton_seconds_per_meter: value.newton_seconds_per_meter,
                });
                damping_count += 1;
            }
            ConnectorKind::IsotropicGroundSpring {
                target,
                stiffness: value,
            } => {
                let resolved = resolve(connector, target, ConnectorTargetRole::Point)?;
                let node = exactly_one_node(connector, target, &resolved.node_indices)?;
                add_isotropic_ground_operator(
                    stiffness,
                    context.layout,
                    node,
                    value.newtons_per_meter,
                );
                stiffness.add_contribution(provenance(connector));
                entries.push(CanonicalConnectorEntry::IsotropicGroundSpring {
                    connector_id: connector.connector_id.clone(),
                    source_digest,
                    target: canonical_node(node, context.node_identities, connector)?,
                    stiffness_newtons_per_meter: value.newtons_per_meter,
                });
                stiffness_count += 1;
            }
            ConnectorKind::SurfaceNormalFoundation { target, modulus } => {
                let resolved = resolve(connector, target, ConnectorTargetRole::Surface)?;
                if resolved.surface_samples.is_empty() {
                    return Err(invalid(
                        connector,
                        "surface foundation resolved no physical surface quadrature",
                    ));
                }
                add_surface_normal_foundation(
                    stiffness,
                    context.layout,
                    &resolved.surface_samples,
                    modulus.newtons_per_cubic_meter,
                );
                stiffness.add_contribution(provenance(connector));
                entries.push(CanonicalConnectorEntry::SurfaceNormalFoundation {
                    connector_id: connector.connector_id.clone(),
                    source_digest,
                    modulus_newtons_per_cubic_meter: modulus.newtons_per_cubic_meter,
                    quadrature: resolved
                        .surface_samples
                        .iter()
                        .map(|sample| {
                            Ok(SurfaceFoundationQuadrature {
                                nodes: sample
                                    .node_indices
                                    .iter()
                                    .map(|index| {
                                        canonical_node(*index, context.node_identities, connector)
                                    })
                                    .collect::<Result<Vec<_>, StructuralAssemblyError>>()?,
                                shape_values: sample.shape_values.clone(),
                                unit_normal: sample.unit_normal,
                                area_weight_m2: sample.area_weight_m2,
                            })
                        })
                        .collect::<Result<Vec<_>, StructuralAssemblyError>>()?,
                });
                stiffness_count += 1;
            }
        }
    }
    let plan = CanonicalStructuralConnectorPlan::new(
        context.topology_identity,
        context.coordinates_m.len(),
        authored_connectors_digest(connectors).map_err(|_| {
            StructuralAssemblyError::InvalidExplicitConnector {
                connector_id: "connector-plan".into(),
                reason: "connector source identity could not be encoded",
            }
        })?,
        entries,
    )
    .map_err(|_| StructuralAssemblyError::InvalidExplicitConnector {
        connector_id: "connector-plan".into(),
        reason: "canonical connector realization is invalid",
    })?;
    Ok(ExplicitConnectorSummary {
        stiffness_count,
        damping_count,
        plan,
    })
}

fn canonical_node(
    node_index: usize,
    node_identities: &[runmat_meshing_core::StableDigest],
    connector: &AnalysisConnector,
) -> Result<CanonicalConnectorNode, StructuralAssemblyError> {
    let node_identity = node_identities
        .get(node_index)
        .copied()
        .ok_or_else(|| invalid(connector, "connector node identity is absent"))?;
    Ok(CanonicalConnectorNode {
        node_index,
        node_identity,
    })
}

fn axial_axis(
    coordinates_m: &[[f64; 3]],
    endpoints: [usize; 2],
    connector: &AnalysisConnector,
) -> Result<[f64; 3], StructuralAssemblyError> {
    let primary = coordinates_m
        .get(endpoints[0])
        .ok_or_else(|| invalid(connector, "primary endpoint coordinate is absent"))?;
    let secondary = coordinates_m
        .get(endpoints[1])
        .ok_or_else(|| invalid(connector, "secondary endpoint coordinate is absent"))?;
    unit_vector(std::array::from_fn(|index| {
        secondary[index] - primary[index]
    }))
    .ok_or_else(|| invalid(connector, "axial connector endpoints are coincident"))
}

fn resolve_axial_endpoints(
    connector: &AnalysisConnector,
    primary: &RegionTarget,
    secondary: &RegionTarget,
    resolve: &mut impl FnMut(
        &AnalysisConnector,
        &RegionTarget,
        ConnectorTargetRole,
    ) -> Result<ResolvedConnectorTarget, StructuralAssemblyError>,
) -> Result<[usize; 2], StructuralAssemblyError> {
    let primary_nodes = resolve(connector, primary, ConnectorTargetRole::Point)?;
    let secondary_nodes = resolve(connector, secondary, ConnectorTargetRole::Point)?;
    let endpoints = [
        exactly_one_node(connector, primary, &primary_nodes.node_indices)?,
        exactly_one_node(connector, secondary, &secondary_nodes.node_indices)?,
    ];
    if endpoints[0] == endpoints[1] {
        return Err(invalid(
            connector,
            "axial connector endpoints resolve to the same node",
        ));
    }
    Ok(endpoints)
}

fn exactly_one_node(
    connector: &AnalysisConnector,
    target: &RegionTarget,
    nodes: &[usize],
) -> Result<usize, StructuralAssemblyError> {
    match nodes {
        [node] => Ok(*node),
        [] => Err(StructuralAssemblyError::UnmappedExplicitConnector {
            connector_id: connector.connector_id.clone(),
            target: "point",
            region_id: target.region_id.clone(),
        }),
        _ => Err(invalid(
            connector,
            "point connector target must resolve to exactly one node",
        )),
    }
}

fn resolve_point_target(
    connector: &AnalysisConnector,
    target: &RegionTarget,
    nodes: Vec<usize>,
) -> Result<ResolvedConnectorTarget, StructuralAssemblyError> {
    let node = exactly_one_node(connector, target, &nodes)?;
    Ok(ResolvedConnectorTarget {
        node_indices: vec![node],
        surface_samples: Vec::new(),
    })
}

pub(super) fn resolve_explicit_surface(
    connector: &AnalysisConnector,
    target: &RegionTarget,
    structural: &StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
) -> Result<ResolvedConnectorTarget, StructuralAssemblyError> {
    let surface = explicit_shell_surface_geometry(structural, topology, &target.region_id)
        .map_err(|_| {
            invalid(
                connector,
                "surface foundation target must resolve to shell faces",
            )
        })?;
    let node_index_by_identity = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index))
        .collect::<BTreeMap<_, _>>();
    let barycentric = [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ];
    let mut samples = Vec::with_capacity(surface.triangles.len() * barycentric.len());
    let mut nodes = Vec::new();
    for triangle in &surface.triangles {
        let normal = unit_vector(triangle.oriented_area_vector_m2)
            .ok_or_else(|| invalid(connector, "surface foundation face has no oriented normal"))?;
        let node_indices = triangle
            .node_identities
            .iter()
            .map(|identity| node_index_by_identity.get(identity).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                invalid(
                    connector,
                    "surface foundation references an absent canonical node",
                )
            })?;
        nodes.extend_from_slice(&node_indices);
        for shape_values in barycentric {
            samples.push(SurfaceFoundationSample {
                node_indices: node_indices.clone(),
                shape_values: shape_values.to_vec(),
                unit_normal: normal,
                area_weight_m2: triangle.area_m2 / 3.0,
            });
        }
    }
    nodes.sort_unstable();
    nodes.dedup();
    Ok(ResolvedConnectorTarget {
        node_indices: nodes,
        surface_samples: samples,
    })
}

pub(super) fn resolve_projected_surface(
    connector: &AnalysisConnector,
    target: &RegionTarget,
    projection: &crate::topology::AnalysisProjection,
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

fn add_axial_operator(
    assembler: &mut RealSparseAssembler,
    layout: &StructuralDofLayout,
    coordinates_m: &[[f64; 3]],
    endpoints: [usize; 2],
    coefficient: f64,
    connector: &AnalysisConnector,
) -> Result<(), StructuralAssemblyError> {
    let axis = axial_axis(coordinates_m, endpoints, connector)?;
    let mut terms = Vec::with_capacity(6);
    for (node, sign) in [(endpoints[0], -1.0), (endpoints[1], 1.0)] {
        for (dof, component) in [
            (StructuralDofKind::Ux, axis[0]),
            (StructuralDofKind::Uy, axis[1]),
            (StructuralDofKind::Uz, axis[2]),
        ] {
            let row = layout.index(node, dof).ok_or_else(|| {
                invalid(
                    connector,
                    "axial endpoint lacks a translational degree of freedom",
                )
            })?;
            terms.push((row, sign * component));
        }
    }
    add_rank_one(assembler, &terms, coefficient);
    Ok(())
}

fn add_isotropic_ground_operator(
    assembler: &mut RealSparseAssembler,
    layout: &StructuralDofLayout,
    node: usize,
    coefficient: f64,
) {
    for dof in [
        StructuralDofKind::Ux,
        StructuralDofKind::Uy,
        StructuralDofKind::Uz,
    ] {
        let row = layout
            .index(node, dof)
            .expect("structural connector nodes own translational degrees of freedom");
        assembler.add(row, row, coefficient);
    }
}

fn add_surface_normal_foundation(
    assembler: &mut RealSparseAssembler,
    layout: &StructuralDofLayout,
    samples: &[SurfaceFoundationSample],
    modulus: f64,
) {
    for sample in samples {
        let mut terms = Vec::with_capacity(sample.node_indices.len() * 3);
        for (&node, &shape) in sample.node_indices.iter().zip(&sample.shape_values) {
            for (dof, normal) in [
                (StructuralDofKind::Ux, sample.unit_normal[0]),
                (StructuralDofKind::Uy, sample.unit_normal[1]),
                (StructuralDofKind::Uz, sample.unit_normal[2]),
            ] {
                let row = layout
                    .index(node, dof)
                    .expect("surface foundation nodes own translational degrees of freedom");
                terms.push((row, shape * normal));
            }
        }
        add_rank_one(assembler, &terms, modulus * sample.area_weight_m2);
    }
}

fn add_rank_one(assembler: &mut RealSparseAssembler, terms: &[(usize, f64)], scale: f64) {
    for &(row, row_weight) in terms {
        for &(column, column_weight) in terms {
            assembler.add(row, column, scale * row_weight * column_weight);
        }
    }
}

fn unit_vector(vector: [f64; 3]) -> Option<[f64; 3]> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    (norm.is_finite() && norm > 1.0e-14).then(|| vector.map(|value| value / norm))
}

fn invalid(connector: &AnalysisConnector, reason: &'static str) -> StructuralAssemblyError {
    StructuralAssemblyError::InvalidExplicitConnector {
        connector_id: connector.connector_id.clone(),
        reason,
    }
}

fn provenance(connector: &AnalysisConnector) -> ContributionProvenance {
    ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId(format!("connector:{}", connector.connector_id)),
    }
}

#[cfg(test)]
mod tests;
