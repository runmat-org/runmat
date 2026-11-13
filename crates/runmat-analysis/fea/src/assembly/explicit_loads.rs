use std::collections::BTreeMap;

use runmat_analysis_core::{StructuralElementKind, StructuralModel};
use runmat_meshing_core::StableDigest;

mod context;

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
    physics::structural::{
        frame::{transform_vector, unit_vector},
        topology::{
            explicit_element_identity, explicit_node_selector,
            explicit_surface::{segment_length, triangle_area, triangle_centroid},
        },
    },
    properties::{CompiledPropertyMap, CompiledShellPropertyOwner},
};
use context::ExplicitLoadDomain;

pub(super) struct ExplicitLoadProjection {
    pub direct_rotational_moment_count: usize,
    pub moment_projection: Vec<MomentProjectionSummary>,
    pub evidence: Vec<SpatialProjectionEvidence>,
}

pub(super) fn apply_projected_explicit_structural_loads(
    structural: &StructuralModel,
    properties: &CompiledPropertyMap,
    excitations: &RealizedExcitations,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    topology: &crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
) -> Result<ExplicitLoadProjection, StructuralAssemblyError> {
    let canonical_node_indices = &topology.explicit_node_indices;
    let canonical_node_coordinates_m = topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect::<Vec<_>>();
    let projection_digest = topology.identity.discretization_digest();
    if canonical_node_indices.len() != structural.nodes.len() {
        return Err(StructuralAssemblyError::InvalidCanonicalNodeMap {
            expected: structural.nodes.len(),
            actual: canonical_node_indices.len(),
        });
    }
    let authored_node_indices = structural
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, canonical_node_indices[index]))
        .collect::<BTreeMap<_, _>>();
    let domain = ExplicitLoadDomain {
        structural,
        properties,
        topology,
        authored_node_indices: &authored_node_indices,
        layout,
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                &canonical_node_coordinates_m,
            ),
    };
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
        if let RealizedExcitationKind::BodyForce(value) = load.kind {
            let item = apply_explicit_body_force(&domain, rhs, load, value.real)?;
            record_evidence(&mut evidence, item)?;
            continue;
        }
        let region_id = load.target.solver_region_id().ok_or_else(|| {
            StructuralAssemblyError::UnmappedStructuralLoad {
                load_id: load.excitation_id.0.clone(),
                region_id: "<non-region target>".to_string(),
                load_kind: super::explicit_load_kind(&load.kind),
            }
        })?;
        let explicit_target_nodes = structural_target_nodes(structural, region_id)?;
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
        let reference_point_m = target_centroid(&canonical_node_coordinates_m, &target_nodes);
        let scale = 1.0 / target_nodes.len() as f64;
        match load.kind {
            RealizedExcitationKind::Force(value) => {
                let force = resolve_vector(
                    load,
                    value.real,
                    reference_point_m,
                    explicit_region_normal(&domain, region_id)?,
                )?;
                let mut item = SpatialProjectionAccumulator::for_explicit_topology(
                    load,
                    projection_digest,
                    &canonical_node_coordinates_m,
                    SpatialIntegrationRule::UniformResultant,
                    resolved_entity_count,
                    reference_point_m,
                );
                item.add_requested_force(reference_point_m, force);
                for &node_index in &target_nodes {
                    let nodal_force = force.map(|component| component * scale);
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
                let moment = resolve_vector(
                    load,
                    value.real,
                    reference_point_m,
                    explicit_region_normal(&domain, region_id)?,
                )?;
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
                            moment[0] * scale,
                        );
                        add_rhs(
                            layout,
                            rhs,
                            node_index,
                            StructuralDofKind::Ry,
                            moment[1] * scale,
                        );
                        add_rhs(
                            layout,
                            rhs,
                            node_index,
                            StructuralDofKind::Rz,
                            moment[2] * scale,
                        );
                    }
                    let mut item = SpatialProjectionAccumulator::for_explicit_topology(
                        load,
                        projection_digest,
                        &canonical_node_coordinates_m,
                        SpatialIntegrationRule::UniformResultant,
                        resolved_entity_count,
                        reference_point_m,
                    );
                    item.register_nodes(target_nodes.iter().copied());
                    item.add_requested_moment(moment);
                    item.add_projected_moment(moment);
                    record_evidence(&mut evidence, item.finish())?;
                } else {
                    let summary = add_moment_projection_rhs(
                        &canonical_node_coordinates_m,
                        layout,
                        rhs,
                        &target_nodes,
                        moment,
                    );
                    let mut item = SpatialProjectionAccumulator::for_explicit_topology(
                        load,
                        projection_digest,
                        &canonical_node_coordinates_m,
                        SpatialIntegrationRule::UniformResultant,
                        resolved_entity_count,
                        summary.projection_point_m,
                    );
                    item.register_nodes(target_nodes.iter().copied());
                    item.add_requested_moment(moment);
                    item.add_projected_moment(summary.applied_moment);
                    record_evidence(&mut evidence, item.finish())?;
                    moment_projection.push(MomentProjectionSummary {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        ..summary
                    });
                }
            }
            RealizedExcitationKind::Pressure(value) => {
                let item = apply_explicit_shell_surface_load(
                    &domain,
                    rhs,
                    load,
                    region_id,
                    ExplicitSurfaceLoad::Pressure(value.real),
                )?;
                record_evidence(&mut evidence, item)?;
            }
            RealizedExcitationKind::Traction(value) => {
                let item = apply_explicit_shell_surface_load(
                    &domain,
                    rhs,
                    load,
                    region_id,
                    ExplicitSurfaceLoad::Traction(value.real),
                )?;
                record_evidence(&mut evidence, item)?;
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

enum ExplicitSurfaceLoad {
    Pressure(f64),
    Traction([f64; 3]),
}

fn apply_explicit_shell_surface_load(
    domain: &ExplicitLoadDomain<'_>,
    rhs: &mut [f64],
    load: &crate::excitation::RealizedExcitation,
    region_id: &str,
    surface_load: ExplicitSurfaceLoad,
) -> Result<SpatialProjectionEvidence, StructuralAssemblyError> {
    let surface =
        crate::physics::structural::topology::explicit_surface::explicit_shell_surface_geometry(
            domain.structural,
            domain.topology,
            region_id,
        )
        .map_err(|error| StructuralAssemblyError::InvalidLoadPlan(error.to_string()))?;
    let identity_to_index = domain
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index))
        .collect::<BTreeMap<_, _>>();
    let reference_point_m = surface
        .triangles
        .iter()
        .fold([0.0; 3], |mut total, triangle| {
            let centroid = triangle_centroid(triangle.coordinates_m.canonical_value());
            for (component, value) in total.iter_mut().zip(centroid) {
                *component +=
                    value * triangle.area_m2.canonical_value() / surface.area_m2.canonical_value();
            }
            total
        });
    let mut evidence = SpatialProjectionAccumulator::for_explicit_topology(
        load,
        domain.topology.identity.discretization_digest(),
        domain.coordinates_m.canonical_value(),
        SpatialIntegrationRule::ExplicitConsistentElement,
        surface.triangles.len(),
        reference_point_m,
    );
    for triangle in &surface.triangles {
        let point = triangle_centroid(triangle.coordinates_m.canonical_value());
        let normal = unit_vector(triangle.oriented_area_vector_m2).map_err(|_| {
            StructuralAssemblyError::InvalidLoadPlan(format!(
                "shell element `{}` has no oriented normal",
                triangle.element_id
            ))
        })?;
        let integrated = match surface_load {
            ExplicitSurfaceLoad::Pressure(value) => triangle
                .oriented_area_vector_m2
                .map(|component| component * value),
            ExplicitSurfaceLoad::Traction(components) => {
                resolve_vector(load, components, point, Some(normal))?
                    .map(|component| component * triangle.area_m2.canonical_value())
            }
        };
        evidence.add_requested_force(point, integrated);
        for identity in triangle.node_identities {
            let node_index = *identity_to_index.get(&identity).ok_or_else(|| {
                StructuralAssemblyError::InvalidLoadPlan(format!(
                    "shell element `{}` references a node outside canonical topology",
                    triangle.element_id
                ))
            })?;
            let nodal_force = integrated.map(|component| component / 3.0);
            add_translation(domain.layout, rhs, node_index, nodal_force);
            evidence.add_projected_force(node_index, nodal_force);
        }
    }
    Ok(evidence.finish())
}

fn apply_explicit_body_force(
    domain: &ExplicitLoadDomain<'_>,
    rhs: &mut [f64],
    load: &crate::excitation::RealizedExcitation,
    density_components_n_per_m3: [f64; 3],
) -> Result<SpatialProjectionEvidence, StructuralAssemblyError> {
    let selected = domain
        .structural
        .elements
        .iter()
        .filter(|element| match &load.target {
            crate::excitation::RealizedExcitationTarget::WholeModel => true,
            crate::excitation::RealizedExcitationTarget::Region(target) => {
                element.region_id == target.region_id
            }
            _ => false,
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(StructuralAssemblyError::UnmappedStructuralLoad {
            load_id: load.excitation_id.0.clone(),
            region_id: load
                .target
                .solver_region_id()
                .unwrap_or("<whole_model>")
                .into(),
            load_kind: super::StructuralLoadKind::BodyForce,
        });
    }
    let topology_digest = domain
        .properties
        .identity
        .explicit_topology_digest
        .ok_or_else(|| {
            StructuralAssemblyError::InvalidLoadPlan(
                "explicit body force has no compiled topology identity".into(),
            )
        })?;
    let mut measures = Vec::with_capacity(selected.len());
    for element in selected {
        let identity = explicit_element_identity(
            topology_digest,
            &element.element_id,
            &element.region_id,
            &element.kind,
        )
        .map_err(|error| StructuralAssemblyError::InvalidLoadPlan(error.to_string()))?;
        let (node_ids, measure_m3) = match &element.kind {
            StructuralElementKind::Beam(beam) => {
                let property = domain
                    .properties
                    .beam_property(identity)
                    .ok_or_else(|| missing_property(&element.element_id, identity))?;
                let from = domain.coordinates_m.canonical_value()[explicit_node_index(
                    domain.authored_node_indices,
                    beam.node_ids[0],
                    &element.element_id,
                )?];
                let to = domain.coordinates_m.canonical_value()[explicit_node_index(
                    domain.authored_node_indices,
                    beam.node_ids[1],
                    &element.element_id,
                )?];
                (
                    beam.node_ids.to_vec(),
                    segment_length(from, to) * property.section.area_m2.canonical_value(),
                )
            }
            StructuralElementKind::Shell(shell) => {
                let property = domain
                    .properties
                    .shell_property(identity)
                    .ok_or_else(|| missing_property(&element.element_id, identity))?;
                let mut points = [[0.0; 3]; 3];
                for (point, node_id) in points.iter_mut().zip(shell.node_ids) {
                    *point = domain.coordinates_m.canonical_value()[explicit_node_index(
                        domain.authored_node_indices,
                        node_id,
                        &element.element_id,
                    )?];
                }
                (
                    shell.node_ids.to_vec(),
                    triangle_area(points) * shell_thickness(property),
                )
            }
        };
        if !measure_m3.is_finite() || measure_m3 <= 0.0 {
            return Err(StructuralAssemblyError::InvalidLoadPlan(format!(
                "element `{}` has invalid physical volume",
                element.element_id
            )));
        }
        let indices = node_ids
            .iter()
            .map(|node_id| {
                explicit_node_index(domain.authored_node_indices, *node_id, &element.element_id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let centroid = target_centroid(domain.coordinates_m.canonical_value(), &indices);
        measures.push((indices, centroid, measure_m3));
    }
    let total_measure = measures.iter().map(|(_, _, measure)| measure).sum::<f64>();
    let reference_point_m = measures
        .iter()
        .fold([0.0; 3], |mut total, (_, point, measure)| {
            for (component, value) in total.iter_mut().zip(*point) {
                *component += value * measure / total_measure;
            }
            total
        });
    let mut evidence = SpatialProjectionAccumulator::for_explicit_topology(
        load,
        domain.topology.identity.discretization_digest(),
        domain.coordinates_m.canonical_value(),
        SpatialIntegrationRule::ExplicitConsistentElement,
        measures.len(),
        reference_point_m,
    );
    for (indices, point, measure) in measures {
        let density = resolve_vector(load, density_components_n_per_m3, point, None)?;
        let integrated = density.map(|component| component * measure);
        evidence.add_requested_force(point, integrated);
        for node_index in indices.iter().copied() {
            let nodal_force = integrated.map(|component| component / indices.len() as f64);
            add_translation(domain.layout, rhs, node_index, nodal_force);
            evidence.add_projected_force(node_index, nodal_force);
        }
    }
    Ok(evidence.finish())
}

fn shell_thickness(property: &crate::properties::CompiledShellProperty) -> f64 {
    match &property.owner {
        CompiledShellPropertyOwner::Homogeneous { section, .. } => {
            section.thickness_m.canonical_value()
        }
        CompiledShellPropertyOwner::Laminate { plies, .. } => plies
            .iter()
            .map(|ply| ply.definition.thickness_m.canonical_value())
            .sum(),
    }
}

fn missing_property(element_id: &str, identity: StableDigest) -> StructuralAssemblyError {
    StructuralAssemblyError::InvalidMaterialCoverage(format!(
        "explicit element `{element_id}` has no compiled property for identity {identity:?}"
    ))
}

fn explicit_node_index(
    indices: &BTreeMap<u32, usize>,
    node_id: u32,
    element_id: &str,
) -> Result<usize, StructuralAssemblyError> {
    indices
        .get(&node_id)
        .copied()
        .ok_or_else(|| StructuralAssemblyError::UnknownNode {
            element_id: element_id.into(),
            node_id,
        })
}

fn explicit_region_normal(
    domain: &ExplicitLoadDomain<'_>,
    region_id: &str,
) -> Result<Option<[f64; 3]>, StructuralAssemblyError> {
    if !domain.structural.elements.iter().any(|element| {
        element.region_id == region_id && matches!(element.kind, StructuralElementKind::Shell(_))
    }) {
        return Ok(None);
    }
    let surface =
        crate::physics::structural::topology::explicit_surface::explicit_shell_surface_geometry(
            domain.structural,
            domain.topology,
            region_id,
        )
        .map_err(|error| StructuralAssemblyError::InvalidLoadPlan(error.to_string()))?;
    Ok(unit_vector(
        surface
            .triangles
            .iter()
            .fold([0.0; 3], |mut total, triangle| {
                for (component, value) in total.iter_mut().zip(triangle.oriented_area_vector_m2) {
                    *component += value;
                }
                total
            }),
    )
    .ok())
}

fn resolve_vector(
    load: &crate::excitation::RealizedExcitation,
    components: [f64; 3],
    point: [f64; 3],
    normal: Option<[f64; 3]>,
) -> Result<[f64; 3], StructuralAssemblyError> {
    transform_vector(&load.policy.frame, components, point, normal).map_err(|error| {
        StructuralAssemblyError::InvalidLoadPlan(format!(
            "excitation `{}` coordinate frame is undefined: {error}",
            load.excitation_id.0
        ))
    })
}

fn add_translation(
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    node_index: usize,
    force: [f64; 3],
) {
    for (kind, value) in [
        StructuralDofKind::Ux,
        StructuralDofKind::Uy,
        StructuralDofKind::Uz,
    ]
    .into_iter()
    .zip(force)
    {
        add_rhs(layout, rhs, node_index, kind, value);
    }
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
