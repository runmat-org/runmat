use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisConnector, ConnectorKind, RegionTarget, StructuralModel};

use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    structural_target_nodes, StructuralAssemblyError,
};
use crate::{
    algebra::{ContributionDomain, ContributionId, ContributionProvenance, RealSparseAssembler},
    physics::structural::topology::{
        explicit_surface::explicit_shell_surface_geometry, node_space::CanonicalStructuralNodeSpace,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ExplicitConnectorSummary {
    pub stiffness_count: usize,
    pub damping_count: usize,
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
    assemble_connectors_with_resolver(
        connectors,
        layout,
        &coordinates_m,
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
    layout: &StructuralDofLayout,
    coordinates_m: &[[f64; 3]],
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
    for connector in connectors {
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
                    layout,
                    coordinates_m,
                    endpoints,
                    value.newtons_per_meter,
                    connector,
                )?;
                stiffness.add_contribution(provenance(connector));
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
                    layout,
                    coordinates_m,
                    endpoints,
                    value.newton_seconds_per_meter,
                    connector,
                )?;
                damping.add_contribution(provenance(connector));
                damping_count += 1;
            }
            ConnectorKind::IsotropicGroundSpring {
                target,
                stiffness: value,
            } => {
                let resolved = resolve(connector, target, ConnectorTargetRole::Point)?;
                let node = exactly_one_node(connector, target, &resolved.node_indices)?;
                add_isotropic_ground_operator(stiffness, layout, node, value.newtons_per_meter);
                stiffness.add_contribution(provenance(connector));
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
                    layout,
                    &resolved.surface_samples,
                    modulus.newtons_per_cubic_meter,
                );
                stiffness.add_contribution(provenance(connector));
                stiffness_count += 1;
            }
        }
    }
    Ok(ExplicitConnectorSummary {
        stiffness_count,
        damping_count,
    })
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
    let primary = coordinates_m
        .get(endpoints[0])
        .ok_or_else(|| invalid(connector, "primary endpoint coordinate is absent"))?;
    let secondary = coordinates_m
        .get(endpoints[1])
        .ok_or_else(|| invalid(connector, "secondary endpoint coordinate is absent"))?;
    let axis = unit_vector(std::array::from_fn(|i| secondary[i] - primary[i]))
        .ok_or_else(|| invalid(connector, "axial connector endpoints are coincident"))?;
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
mod tests {
    use super::*;
    use crate::{
        algebra::{MatrixProperties, OperatorRole},
        assembly::dofs::StructuralNodeDofSet,
    };

    #[test]
    fn axial_operator_is_objective_under_infinitesimal_rigid_rotation() {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational(),
            StructuralNodeDofSet::translational(),
        ]);
        let connector = AnalysisConnector {
            connector_id: "axial".into(),
            kind: ConnectorKind::AxialSpring {
                primary: RegionTarget::point("node:1"),
                secondary: RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(100.0),
            },
        };
        let mut assembler = RealSparseAssembler::square(
            layout.total_dof_count(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        add_axial_operator(
            &mut assembler,
            &layout,
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [0, 1],
            100.0,
            &connector,
        )
        .unwrap();
        let matrix = assembler.finish().unwrap();
        let rigid_rotation = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(quadratic_form(&matrix, &rigid_rotation), 0.0);
        let axial_extension = [0.0, 0.0, 0.0, 0.25, 0.0, 0.0];
        assert!((quadratic_form(&matrix, &axial_extension) - 6.25).abs() < 1.0e-12);
    }

    #[test]
    fn surface_foundation_scales_with_area_not_node_count() {
        let coarse = foundation_rigid_normal_stiffness(
            1,
            vec![SurfaceFoundationSample {
                node_indices: vec![0],
                shape_values: vec![1.0],
                unit_normal: [0.0, 0.0, 1.0],
                area_weight_m2: 2.0,
            }],
        );
        let refined = foundation_rigid_normal_stiffness(
            2,
            vec![
                SurfaceFoundationSample {
                    node_indices: vec![0],
                    shape_values: vec![1.0],
                    unit_normal: [0.0, 0.0, 1.0],
                    area_weight_m2: 1.0,
                },
                SurfaceFoundationSample {
                    node_indices: vec![1],
                    shape_values: vec![1.0],
                    unit_normal: [0.0, 0.0, 1.0],
                    area_weight_m2: 1.0,
                },
            ],
        );
        assert!((coarse - 20.0).abs() < 1.0e-12);
        assert!((refined - coarse).abs() < 1.0e-12);
    }

    fn foundation_rigid_normal_stiffness(
        node_count: usize,
        samples: Vec<SurfaceFoundationSample>,
    ) -> f64 {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational();
            node_count
        ]);
        let mut assembler = RealSparseAssembler::square(
            layout.total_dof_count(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        add_surface_normal_foundation(&mut assembler, &layout, &samples, 10.0);
        let matrix = assembler.finish().unwrap();
        let mut rigid_normal = vec![0.0; layout.total_dof_count()];
        for node in 0..node_count {
            rigid_normal[layout.index(node, StructuralDofKind::Uz).unwrap()] = 1.0;
        }
        quadratic_form(&matrix, &rigid_normal)
    }

    fn quadratic_form(matrix: &crate::algebra::RealSparseMatrix, vector: &[f64]) -> f64 {
        (0..vector.len())
            .map(|row| {
                (0..vector.len())
                    .map(|column| vector[row] * matrix.get(row, column) * vector[column])
                    .sum::<f64>()
            })
            .sum()
    }
}
