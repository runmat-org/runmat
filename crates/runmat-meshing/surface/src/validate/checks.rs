use std::collections::{BTreeMap, BTreeSet};

use crate::{
    math::{dot, norm, point_triangle_distance, triangle_area, MeshingTolerance},
    SurfaceDiscretization, SurfaceElement, INTERNAL_SOURCE_EDGE_ID,
};
use runmat_meshing_cad::{CadFace, CadTopologyModel, SourceTopologyEdge, SourceTopologyModel};

use super::{
    geometry::{surface_element_points, topology_face_points, unit_normal},
    source_edges::{
        count_closed_source_edge_loops, source_edge_is_recovered_by_chain, surface_edge_source_ids,
    },
    types::{SurfaceValidationError, SurfaceValidationOptions, SurfaceValidationReport},
};

pub fn validate_surface_discretization(
    topology: &SourceTopologyModel,
    surface: &SurfaceDiscretization,
    options: SurfaceValidationOptions,
) -> Result<SurfaceValidationReport, SurfaceValidationError> {
    validate_options(options)?;
    if surface.elements.is_empty() {
        return Err(SurfaceValidationError::EmptySurface);
    }

    let tolerance = MeshingTolerance::from_bounds(topology.bounds_min_m, topology.bounds_max_m);
    let source_faces = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    let source_edges = topology
        .edges
        .iter()
        .map(|edge| (edge.edge_id, edge))
        .collect::<BTreeMap<_, _>>();
    let surface_edges = surface_edge_source_ids(surface);

    let mut covered_source_faces = BTreeSet::<u32>::new();
    let mut conforming_source_edges = BTreeSet::<u32>::new();
    let mut max_projection_error_m = 0.0_f64;
    let mut min_orientation_alignment = f64::INFINITY;

    for element in &surface.elements {
        let source_face = source_faces.get(&element.source_face_id).ok_or(
            SurfaceValidationError::MissingSourceFace {
                source_face_id: element.source_face_id,
            },
        )?;
        let points = surface_element_points(surface, element)?;
        if triangle_area(points) <= tolerance.length_epsilon(1.0).powi(2) {
            return Err(SurfaceValidationError::DegenerateElement {
                element_id: element.element_id,
            });
        }
        let source_points = topology_face_points(topology, source_face.node_ids)?;
        let projection_error_m = points
            .into_iter()
            .map(|point| point_triangle_distance(point, source_points))
            .fold(0.0_f64, f64::max);
        max_projection_error_m = max_projection_error_m.max(projection_error_m);
        if projection_error_m > options.max_projection_error_m.max(tolerance.absolute_m) {
            return Err(SurfaceValidationError::ProjectionError {
                element_id: element.element_id,
                error_m: projection_error_m,
                max_error_m: options.max_projection_error_m,
            });
        }

        let surface_normal =
            unit_normal(points).ok_or(SurfaceValidationError::DegenerateElement {
                element_id: element.element_id,
            })?;
        let element_normal = validate_element_unit_normal(element)?;
        let alignment = dot(surface_normal, element_normal);
        min_orientation_alignment = min_orientation_alignment.min(alignment);
        if alignment < options.min_orientation_alignment {
            return Err(SurfaceValidationError::OrientationMismatch {
                element_id: element.element_id,
                source_face_id: element.source_face_id,
                alignment,
                min_alignment: options.min_orientation_alignment,
            });
        }

        covered_source_faces.insert(element.source_face_id);
        for source_edge_id in element.source_edge_ids {
            if source_edge_id == INTERNAL_SOURCE_EDGE_ID {
                continue;
            }
            source_edges
                .get(&source_edge_id)
                .ok_or(SurfaceValidationError::MissingSourceEdge { source_edge_id })?;
        }
    }

    for (source_edge_id, source_edge) in &source_edges {
        if source_edge_is_recovered_by_chain(
            surface_edges
                .get(source_edge_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            source_edge,
        ) {
            conforming_source_edges.insert(*source_edge_id);
        }
    }

    if options.require_source_edge_conformity {
        for edge in &topology.edges {
            if !conforming_source_edges.contains(&edge.edge_id) {
                return Err(SurfaceValidationError::EdgeConformityFailed {
                    source_edge_id: edge.edge_id,
                    source_edge_node_ids: edge.node_ids,
                    recovered_segment_count: surface_edges
                        .get(&edge.edge_id)
                        .map(Vec::len)
                        .unwrap_or_default(),
                });
            }
        }
    }

    for source_face in source_faces.keys() {
        if !covered_source_faces.contains(source_face) {
            return Err(SurfaceValidationError::UncoveredSourceFace {
                source_face_id: *source_face,
            });
        }
    }

    let (source_edge_loop_count, closed_source_edge_loop_count) =
        count_closed_source_edge_loops(&topology.edges)?;

    Ok(SurfaceValidationReport {
        source_face_count: topology.faces.len(),
        surface_element_count: surface.elements.len(),
        source_edge_loop_count,
        closed_source_edge_loop_count,
        conforming_source_edge_count: conforming_source_edges.len(),
        missing_source_edge_count: topology
            .edges
            .len()
            .saturating_sub(conforming_source_edges.len()),
        max_projection_error_m,
        min_orientation_alignment: if min_orientation_alignment.is_finite() {
            min_orientation_alignment
        } else {
            1.0
        },
        face_coverage_ratio: covered_source_faces.len() as f64 / topology.faces.len() as f64,
    })
}

pub fn validate_cad_topology_surface_discretization(
    cad_topology: &CadTopologyModel,
    topology: &SourceTopologyModel,
    surface: &SurfaceDiscretization,
    options: SurfaceValidationOptions,
) -> Result<SurfaceValidationReport, SurfaceValidationError> {
    validate_options(options)?;
    if surface.elements.is_empty() {
        return Err(SurfaceValidationError::EmptySurface);
    }

    let tolerance = MeshingTolerance::from_bounds(topology.bounds_min_m, topology.bounds_max_m);
    let source_edges = topology
        .edges
        .iter()
        .map(|edge| (edge.edge_id, edge))
        .collect::<BTreeMap<_, _>>();
    let cad_faces = cad_topology
        .faces
        .iter()
        .map(|face| (face.entity_id.id.clone(), face))
        .collect::<BTreeMap<_, _>>();
    let surface_edges = surface_edge_source_ids(surface);

    let mut cad_face_loop_edges = BTreeMap::<String, BTreeSet<u32>>::new();
    let mut all_loop_source_edges = BTreeSet::<u32>::new();
    for cad_face in &cad_topology.faces {
        let loop_edge_ids = cad_face_loop_source_edge_ids(cad_face)?;
        all_loop_source_edges.extend(loop_edge_ids.iter().copied());
        cad_face_loop_edges.insert(cad_face.entity_id.id.clone(), loop_edge_ids);
    }

    let mut covered_cad_faces = BTreeSet::<String>::new();
    let mut conforming_source_edges = BTreeSet::<u32>::new();
    let mut max_projection_error_m = 0.0_f64;
    let mut min_orientation_alignment = f64::INFINITY;

    for element in &surface.elements {
        let cad_face_id =
            element
                .cad_face_id
                .as_ref()
                .ok_or(SurfaceValidationError::MissingCadFaceId {
                    element_id: element.element_id,
                })?;
        let _cad_face =
            cad_faces
                .get(cad_face_id)
                .ok_or_else(|| SurfaceValidationError::MissingCadFace {
                    cad_face_id: cad_face_id.clone(),
                })?;
        let loop_edges = cad_face_loop_edges.get(cad_face_id).ok_or_else(|| {
            SurfaceValidationError::MissingCadFace {
                cad_face_id: cad_face_id.clone(),
            }
        })?;

        let points = surface_element_points(surface, element)?;
        if triangle_area(points) <= tolerance.length_epsilon(1.0).powi(2) {
            return Err(SurfaceValidationError::DegenerateElement {
                element_id: element.element_id,
            });
        }
        max_projection_error_m = max_projection_error_m.max(element.max_projection_error_m);
        if element.max_projection_error_m > options.max_projection_error_m.max(tolerance.absolute_m)
        {
            return Err(SurfaceValidationError::ProjectionError {
                element_id: element.element_id,
                error_m: element.max_projection_error_m,
                max_error_m: options.max_projection_error_m,
            });
        }

        let surface_normal =
            unit_normal(points).ok_or(SurfaceValidationError::DegenerateElement {
                element_id: element.element_id,
            })?;
        let element_normal = validate_element_unit_normal(element)?;
        let alignment = dot(surface_normal, element_normal);
        min_orientation_alignment = min_orientation_alignment.min(alignment);
        if alignment < options.min_orientation_alignment {
            return Err(SurfaceValidationError::OrientationMismatch {
                element_id: element.element_id,
                source_face_id: element.source_face_id,
                alignment,
                min_alignment: options.min_orientation_alignment,
            });
        }

        covered_cad_faces.insert(cad_face_id.clone());
        for source_edge_id in element.source_edge_ids {
            if source_edge_id == INTERNAL_SOURCE_EDGE_ID {
                continue;
            }
            source_edges
                .get(&source_edge_id)
                .ok_or(SurfaceValidationError::MissingSourceEdge { source_edge_id })?;
            if !loop_edges.contains(&source_edge_id) {
                return Err(SurfaceValidationError::SourceEdgeOutsideCadFace {
                    cad_face_id: cad_face_id.clone(),
                    source_edge_id,
                });
            }
        }
    }

    for source_edge_id in &all_loop_source_edges {
        let source_edge =
            source_edges
                .get(source_edge_id)
                .ok_or(SurfaceValidationError::MissingSourceEdge {
                    source_edge_id: *source_edge_id,
                })?;
        if source_edge_is_recovered_by_chain(
            surface_edges
                .get(source_edge_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            source_edge,
        ) {
            conforming_source_edges.insert(*source_edge_id);
        }
    }

    if options.require_source_edge_conformity {
        for source_edge_id in &all_loop_source_edges {
            if !conforming_source_edges.contains(source_edge_id) {
                let source_edge = source_edges.get(source_edge_id).ok_or(
                    SurfaceValidationError::MissingSourceEdge {
                        source_edge_id: *source_edge_id,
                    },
                )?;
                return Err(SurfaceValidationError::EdgeConformityFailed {
                    source_edge_id: *source_edge_id,
                    source_edge_node_ids: source_edge.node_ids,
                    recovered_segment_count: surface_edges
                        .get(source_edge_id)
                        .map(Vec::len)
                        .unwrap_or_default(),
                });
            }
        }
    }

    for cad_face_id in cad_faces.keys() {
        if !covered_cad_faces.contains(cad_face_id) {
            return Err(SurfaceValidationError::UncoveredCadFace {
                cad_face_id: cad_face_id.clone(),
            });
        }
    }

    let loop_source_edges = all_loop_source_edges
        .iter()
        .map(|source_edge_id| {
            source_edges
                .get(source_edge_id)
                .map(|edge| (*edge).clone())
                .ok_or(SurfaceValidationError::MissingSourceEdge {
                    source_edge_id: *source_edge_id,
                })
        })
        .collect::<Result<Vec<SourceTopologyEdge>, SurfaceValidationError>>()?;
    let (source_edge_loop_count, closed_source_edge_loop_count) =
        count_closed_source_edge_loops(&loop_source_edges)?;

    Ok(SurfaceValidationReport {
        source_face_count: cad_topology.faces.len(),
        surface_element_count: surface.elements.len(),
        source_edge_loop_count,
        closed_source_edge_loop_count,
        conforming_source_edge_count: conforming_source_edges.len(),
        missing_source_edge_count: all_loop_source_edges
            .len()
            .saturating_sub(conforming_source_edges.len()),
        max_projection_error_m,
        min_orientation_alignment: if min_orientation_alignment.is_finite() {
            min_orientation_alignment
        } else {
            1.0
        },
        face_coverage_ratio: covered_cad_faces.len() as f64 / cad_topology.faces.len() as f64,
    })
}

fn cad_face_loop_source_edge_ids(
    cad_face: &CadFace,
) -> Result<BTreeSet<u32>, SurfaceValidationError> {
    cad_face
        .loop_edge_ids
        .iter()
        .map(|loop_edge_id| {
            loop_edge_id
                .strip_prefix("cad_edge_")
                .and_then(|edge_id| edge_id.parse::<u32>().ok())
                .ok_or_else(|| SurfaceValidationError::InvalidCadLoopEdgeId {
                    cad_face_id: cad_face.entity_id.id.clone(),
                    loop_edge_id: loop_edge_id.clone(),
                })
        })
        .collect()
}

fn validate_options(options: SurfaceValidationOptions) -> Result<(), SurfaceValidationError> {
    if !options.max_projection_error_m.is_finite()
        || options.max_projection_error_m < 0.0
        || !options.min_orientation_alignment.is_finite()
        || !(0.0..=1.0).contains(&options.min_orientation_alignment)
    {
        return Err(SurfaceValidationError::InvalidOptions);
    }
    Ok(())
}

fn validate_element_unit_normal(
    element: &SurfaceElement,
) -> Result<[f64; 3], SurfaceValidationError> {
    if !element
        .unit_normal
        .iter()
        .all(|component| component.is_finite())
    {
        return Err(SurfaceValidationError::InvalidElementNormal {
            element_id: element.element_id,
        });
    }
    let normal_length = norm(element.unit_normal);
    if (normal_length - 1.0).abs() > 1.0e-8 {
        return Err(SurfaceValidationError::InvalidElementNormal {
            element_id: element.element_id,
        });
    }
    Ok(element.unit_normal)
}
