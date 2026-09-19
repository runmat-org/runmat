use std::collections::{BTreeMap, BTreeSet};

mod geometry;

use geometry::{
    centroid, cross, edge_segments, face_triangles, least_aligned_cartesian_axis, mesh_node,
    midpoint, orientation_axis, structural_node, subtract, FormulationOrientationRole,
};
use runmat_analysis_core::{
    AnalysisModel, AuthoredOrientationDefinition, AuthoredPropertyDefinitionRef,
    AuthoredSectionDefinition, BeamElementModel, BeamSectionModel, PropertyRegionScope,
    PropertyRegionTarget, RealizedAnalysisModel, RealizedModelProperties,
    RealizedPropertyAssignment, ShellElementModel, ShellSectionModel, StructuralElement,
    StructuralElementKind, StructuralModel,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum PropertyRealizationError {
    #[error("property realization received an invalid solver mesh: {0}")]
    InvalidMesh(String),
    #[error("property definition `{0}` is duplicated")]
    DuplicateDefinition(String),
    #[error("property assignment references absent definition `{0}`")]
    MissingDefinition(String),
    #[error("property target `{0}` selects no compatible mesh entity")]
    EmptyTarget(String),
    #[error("mesh entity `{entity}` has {count} section assignments")]
    SectionCoverage { entity: String, count: usize },
    #[error("mesh entity `{entity}` has {count} orientation assignments")]
    OrientationOverlap { entity: String, count: usize },
    #[error("mesh entity `{0}` has a degenerate formulation orientation")]
    DegenerateOrientation(String),
    #[error("solver mesh node id {0} exceeds the structural topology id range")]
    NodeIdOverflow(u64),
    #[error("{entity} has unsupported {node_count}-node boundary topology")]
    UnsupportedBoundaryTopology { entity: String, node_count: usize },
    #[error("layup `{0}` requires the laminate shell formulation")]
    LaminateUnavailable(String),
    #[error("shell section `{0}` uses an offset not supported by the current shell formulation")]
    ShellOffsetUnavailable(String),
}

pub fn realize_formulation_properties(
    realized: RealizedAnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<AnalysisModel, PropertyRealizationError> {
    mesh.validate_canonical()
        .map_err(|error| PropertyRealizationError::InvalidMesh(error.to_string()))?;
    let RealizedAnalysisModel {
        mut model,
        properties,
    } = realized;
    let generated = realize_structural_topology(&properties, mesh)?;
    if !generated.elements.is_empty() {
        let structural = model
            .structural
            .get_or_insert_with(StructuralModel::default);
        merge_structural_topology(structural, generated)?;
    }
    Ok(model)
}

fn realize_structural_topology(
    properties: &RealizedModelProperties,
    mesh: &SolverMeshArtifact,
) -> Result<StructuralModel, PropertyRealizationError> {
    if let Some(layup) = properties.assignments.iter().find_map(|assignment| {
        let RealizedPropertyAssignment::Layup { layup, .. } = assignment else {
            return None;
        };
        Some(layup)
    }) {
        return Err(PropertyRealizationError::LaminateUnavailable(display_id(
            layup,
        )));
    }
    let sections = inventory(
        properties
            .sections
            .iter()
            .map(|value| (&value.definition_id, &value.definition)),
    )?;
    let orientations = inventory(
        properties
            .orientations
            .iter()
            .map(|value| (&value.definition_id, &value.definition)),
    )?;
    let section_assignments = properties
        .assignments
        .iter()
        .filter_map(|assignment| match assignment {
            RealizedPropertyAssignment::Section { section, target } => Some((section, target)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let orientation_assignments = properties
        .assignments
        .iter()
        .filter_map(|assignment| match assignment {
            RealizedPropertyAssignment::Orientation {
                orientation,
                target,
            } => Some((orientation, target)),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut topology = StructuralModel::default();
    let mut node_ids = BTreeMap::new();
    let mut used_sections = BTreeSet::new();
    let mut matched_section_assignments = BTreeSet::new();
    let mut matched_orientation_assignments = BTreeSet::new();

    for edge in &mesh.topology.boundary_edges {
        let matching = section_assignments
            .iter()
            .enumerate()
            .filter(|(_, (_, target))| {
                matches!(target, PropertyRegionTarget::Beam(_))
                    && target_matches(target, &edge.provenance)
            })
            .collect::<Vec<_>>();
        if matching.is_empty() {
            continue;
        }
        if matching.len() != 1 {
            return Err(PropertyRealizationError::SectionCoverage {
                entity: format!("boundary edge {}", edge.edge_id),
                count: matching.len(),
            });
        }
        let (assignment_index, (section_id, target)) = matching[0];
        let definition = sections
            .get(*section_id)
            .ok_or_else(|| PropertyRealizationError::MissingDefinition(display_id(section_id)))?;
        if !definition.is_beam() {
            return Err(PropertyRealizationError::SectionCoverage {
                entity: format!("boundary edge {}", edge.edge_id),
                count: 0,
            });
        }
        let orientation = matching_orientation(
            &orientation_assignments,
            &orientations,
            target,
            &edge.provenance,
            &mut matched_orientation_assignments,
            format!("boundary edge {}", edge.edge_id),
        )?;
        let entity_label = format!("boundary edge {}", edge.edge_id);
        let segments = edge_segments(&edge.node_ids, &entity_label)?;
        for (segment_index, [from, to]) in segments.into_iter().enumerate() {
            let from_id = structural_node(mesh, from, &mut topology.nodes, &mut node_ids)?;
            let to_id = structural_node(mesh, to, &mut topology.nodes, &mut node_ids)?;
            let from_coordinates = mesh_node(mesh, from)?.coordinates_m;
            let to_coordinates = mesh_node(mesh, to)?.coordinates_m;
            let midpoint = midpoint(from_coordinates, to_coordinates);
            let segment_label = format!("{entity_label} segment {segment_index}");
            let segment_axis = subtract(to_coordinates, from_coordinates);
            let reference_axis = orientation_axis(
                orientation,
                midpoint,
                least_aligned_cartesian_axis(segment_axis),
                segment_axis,
                &segment_label,
                FormulationOrientationRole::Beam,
            )?;
            topology.elements.push(StructuralElement {
                element_id: format!("mesh-edge:{}:{segment_index}", edge.edge_id),
                region_id: formulation_region_id(target),
                kind: StructuralElementKind::Beam(BeamElementModel {
                    node_ids: [from_id, to_id],
                    section_id: display_id(section_id),
                    reference_axis,
                }),
            });
        }
        matched_section_assignments.insert(assignment_index);
        used_sections.insert((*section_id).clone());
    }

    for face in &mesh.topology.boundary_faces {
        let matching = section_assignments
            .iter()
            .enumerate()
            .filter(|(_, (_, target))| {
                matches!(target, PropertyRegionTarget::Shell(_))
                    && target_matches(target, &face.provenance)
            })
            .collect::<Vec<_>>();
        if matching.is_empty() {
            continue;
        }
        if matching.len() != 1 {
            return Err(PropertyRealizationError::SectionCoverage {
                entity: format!("boundary face {}", face.face_id),
                count: matching.len(),
            });
        }
        let (assignment_index, (section_id, target)) = matching[0];
        let definition = sections
            .get(*section_id)
            .ok_or_else(|| PropertyRealizationError::MissingDefinition(display_id(section_id)))?;
        let AuthoredSectionDefinition::HomogeneousShell(section) = definition else {
            return Err(PropertyRealizationError::SectionCoverage {
                entity: format!("boundary face {}", face.face_id),
                count: 0,
            });
        };
        if section.offset_fraction != 0.0 {
            return Err(PropertyRealizationError::ShellOffsetUnavailable(
                display_id(section_id),
            ));
        }
        let orientation = matching_orientation(
            &orientation_assignments,
            &orientations,
            target,
            &face.provenance,
            &mut matched_orientation_assignments,
            format!("boundary face {}", face.face_id),
        )?;
        let entity_label = format!("boundary face {}", face.face_id);
        for (triangle_index, triangle) in face_triangles(&face.node_ids, &entity_label)?
            .into_iter()
            .enumerate()
        {
            let node_ids = [
                structural_node(mesh, triangle[0], &mut topology.nodes, &mut node_ids)?,
                structural_node(mesh, triangle[1], &mut topology.nodes, &mut node_ids)?,
                structural_node(mesh, triangle[2], &mut topology.nodes, &mut node_ids)?,
            ];
            let points = [
                mesh_node(mesh, triangle[0])?.coordinates_m,
                mesh_node(mesh, triangle[1])?.coordinates_m,
                mesh_node(mesh, triangle[2])?.coordinates_m,
            ];
            let centroid = centroid(points);
            let first_edge = subtract(points[1], points[0]);
            let face_normal = cross(first_edge, subtract(points[2], points[0]));
            let triangle_label = format!("{entity_label} triangle {triangle_index}");
            let reference_axis = orientation_axis(
                orientation,
                centroid,
                first_edge,
                face_normal,
                &triangle_label,
                FormulationOrientationRole::Shell,
            )?;
            topology.elements.push(StructuralElement {
                element_id: format!("mesh-face:{}:{triangle_index}", face.face_id),
                region_id: formulation_region_id(target),
                kind: StructuralElementKind::Shell(ShellElementModel {
                    node_ids,
                    section_id: display_id(section_id),
                    reference_axis,
                }),
            });
        }
        matched_section_assignments.insert(assignment_index);
        used_sections.insert((*section_id).clone());
    }

    reject_unmatched(&section_assignments, &matched_section_assignments)?;
    reject_unmatched(&orientation_assignments, &matched_orientation_assignments)?;
    for id in used_sections {
        match sections[&id] {
            AuthoredSectionDefinition::HomogeneousShell(value) => {
                topology.shell_sections.push(ShellSectionModel {
                    section_id: display_id(&id),
                    thickness_m: value.thickness_m,
                    integration_points: value.integration_points,
                    shear_correction: value.shear_correction,
                    drilling_stiffness_scale: value.drilling_stiffness_scale,
                });
            }
            value => topology.beam_sections.push(beam_section(&id, value)),
        }
    }
    Ok(topology)
}

fn inventory<'a, T>(
    values: impl Iterator<Item = (&'a AuthoredPropertyDefinitionRef, &'a T)>,
) -> Result<BTreeMap<AuthoredPropertyDefinitionRef, &'a T>, PropertyRealizationError> {
    let mut result = BTreeMap::new();
    for (id, value) in values {
        if result.insert(id.clone(), value).is_some() {
            return Err(PropertyRealizationError::DuplicateDefinition(display_id(
                id,
            )));
        }
    }
    Ok(result)
}

fn target_matches(
    target: &PropertyRegionTarget,
    provenance: &[runmat_geometry_core::PersistentEntityId],
) -> bool {
    match target.scope() {
        PropertyRegionScope::All => true,
        PropertyRegionScope::Region(selection) => provenance
            .iter()
            .any(|entity| selection.matches_persistent_entity(entity)),
    }
}

fn same_formulation_role(left: &PropertyRegionTarget, right: &PropertyRegionTarget) -> bool {
    matches!(
        (left, right),
        (PropertyRegionTarget::Beam(_), PropertyRegionTarget::Beam(_))
            | (
                PropertyRegionTarget::Shell(_),
                PropertyRegionTarget::Shell(_)
            )
            | (
                PropertyRegionTarget::Solid(_),
                PropertyRegionTarget::Solid(_)
            )
            | (PropertyRegionTarget::Cell(_), PropertyRegionTarget::Cell(_))
    )
}

fn matching_orientation<'a>(
    assignments: &'a [(&AuthoredPropertyDefinitionRef, &PropertyRegionTarget)],
    definitions: &BTreeMap<AuthoredPropertyDefinitionRef, &'a AuthoredOrientationDefinition>,
    section_target: &PropertyRegionTarget,
    provenance: &[runmat_geometry_core::PersistentEntityId],
    matched: &mut BTreeSet<usize>,
    entity: String,
) -> Result<Option<&'a AuthoredOrientationDefinition>, PropertyRealizationError> {
    let matches = assignments
        .iter()
        .enumerate()
        .filter(|(_, (_, target))| {
            same_formulation_role(target, section_target) && target_matches(target, provenance)
        })
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return Err(PropertyRealizationError::OrientationOverlap {
            entity,
            count: matches.len(),
        });
    }
    let Some((index, (id, _))) = matches.first().copied() else {
        return Ok(None);
    };
    matched.insert(index);
    definitions
        .get(id)
        .copied()
        .map(Some)
        .ok_or_else(|| PropertyRealizationError::MissingDefinition(display_id(id)))
}

fn reject_unmatched<T>(
    assignments: &[(&AuthoredPropertyDefinitionRef, &T)],
    matched: &BTreeSet<usize>,
) -> Result<(), PropertyRealizationError> {
    if let Some((_, (id, _))) = assignments
        .iter()
        .enumerate()
        .find(|(index, _)| !matched.contains(index))
    {
        return Err(PropertyRealizationError::EmptyTarget(display_id(id)));
    }
    Ok(())
}

fn formulation_region_id(target: &PropertyRegionTarget) -> String {
    target.selected_region_id().map_or_else(
        || {
            match target {
                PropertyRegionTarget::Beam(_) => "all_beams",
                PropertyRegionTarget::Shell(_) => "all_shells",
                PropertyRegionTarget::Solid(_) => "all_solids",
                PropertyRegionTarget::Cell(_) => "all_cells",
            }
            .to_owned()
        },
        str::to_owned,
    )
}

fn beam_section(
    id: &AuthoredPropertyDefinitionRef,
    value: &AuthoredSectionDefinition,
) -> BeamSectionModel {
    let (area, iy, iz, torsion, outer_y, outer_z, radius) = match value {
        AuthoredSectionDefinition::GeneralBeam(value) => (
            value.area_m2,
            value.iy_m4,
            value.iz_m4,
            value.torsion_j_m4,
            value.outer_fiber_y_m,
            value.outer_fiber_z_m,
            value.torsion_outer_radius_m,
        ),
        AuthoredSectionDefinition::RectangularBeam(value) => {
            let (w, h) = (value.width_m, value.height_m);
            let (a, b) = if w >= h { (w, h) } else { (h, w) };
            let torsion =
                a * b.powi(3) * (1.0 / 3.0 - 0.21 * b / a * (1.0 - b.powi(4) / (12.0 * a.powi(4))));
            (
                w * h,
                w * h.powi(3) / 12.0,
                h * w.powi(3) / 12.0,
                torsion,
                w / 2.0,
                h / 2.0,
                w.hypot(h) / 2.0,
            )
        }
        AuthoredSectionDefinition::CircularBeam(value) => {
            let r = value.radius_m;
            (
                std::f64::consts::PI * r * r,
                std::f64::consts::PI * r.powi(4) / 4.0,
                std::f64::consts::PI * r.powi(4) / 4.0,
                std::f64::consts::PI * r.powi(4) / 2.0,
                r,
                r,
                r,
            )
        }
        AuthoredSectionDefinition::TubularBeam(value) => {
            let r4 = value.outer_radius_m.powi(4) - value.inner_radius_m.powi(4);
            (
                std::f64::consts::PI
                    * (value.outer_radius_m.powi(2) - value.inner_radius_m.powi(2)),
                std::f64::consts::PI * r4 / 4.0,
                std::f64::consts::PI * r4 / 4.0,
                std::f64::consts::PI * r4 / 2.0,
                value.outer_radius_m,
                value.outer_radius_m,
                value.outer_radius_m,
            )
        }
        AuthoredSectionDefinition::HomogeneousShell(_) => unreachable!(),
    };
    BeamSectionModel {
        section_id: display_id(id),
        area_m2: area,
        iy_m4: iy,
        iz_m4: iz,
        torsion_j_m4: torsion,
        outer_fiber_y_m: outer_y,
        outer_fiber_z_m: outer_z,
        torsion_outer_radius_m: radius,
    }
}

fn display_id(id: &AuthoredPropertyDefinitionRef) -> String {
    format!("{}::{}::{}", id.module_path, id.model_id, id.property_id)
}

fn merge_structural_topology(
    target: &mut StructuralModel,
    generated: StructuralModel,
) -> Result<(), PropertyRealizationError> {
    if !target.elements.is_empty()
        || !target.nodes.is_empty()
        || !target.beam_sections.is_empty()
        || !target.shell_sections.is_empty()
    {
        return Err(PropertyRealizationError::InvalidMesh(
            "authored explicit topology cannot be combined with geometry-derived property topology"
                .into(),
        ));
    }
    *target = generated;
    Ok(())
}
