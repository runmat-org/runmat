use std::collections::{BTreeMap, BTreeSet};

use crate::{
    math::{dot, norm, triangle_area, MeshingTolerance},
    SurfaceDiscretization,
};
use runmat_meshing_cad::SourceTopologyModel;

use super::{
    edges::triangle_edges,
    geometry::{surface_element_points, triangle_unit_normal},
    types::{SurfaceRecoveryError, SurfaceRecoveryOptions, SurfaceRecoveryReport},
};

pub fn validate_surface_recovery(
    topology: &SourceTopologyModel,
    surface: &SurfaceDiscretization,
    options: SurfaceRecoveryOptions,
) -> Result<SurfaceRecoveryReport, SurfaceRecoveryError> {
    validate_options(options)?;
    if surface.elements.is_empty() {
        return Err(SurfaceRecoveryError::EmptySurface);
    }

    let tolerance = MeshingTolerance::from_bounds(topology.bounds_min_m, topology.bounds_max_m);
    let source_faces = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    let mut covered_source_faces = BTreeSet::<u32>::new();
    let mut edge_incidence = BTreeMap::<[u32; 2], usize>::new();
    let mut recovered_area_by_source_face = BTreeMap::<u32, f64>::new();
    let mut max_area_relative_error = 0.0_f64;
    let mut min_normal_alignment = f64::INFINITY;

    for element in &surface.elements {
        let source_face = source_faces.get(&element.source_face_id).ok_or(
            SurfaceRecoveryError::MissingSourceFace {
                source_face_id: element.source_face_id,
            },
        )?;
        let points = surface_element_points(surface, element)?;
        let area_m2 = triangle_area(points);
        if !area_m2.is_finite() || area_m2 <= tolerance.length_epsilon(1.0).powi(2) {
            return Err(SurfaceRecoveryError::DegenerateElement {
                element_id: element.element_id,
            });
        }

        let expected_area = element.area_m2;
        let relative_error = if expected_area > 0.0 && expected_area.is_finite() {
            (area_m2 - expected_area).abs() / expected_area
        } else {
            0.0
        };
        max_area_relative_error = max_area_relative_error.max(relative_error);
        if relative_error > options.max_area_relative_error {
            return Err(SurfaceRecoveryError::AreaMismatch {
                element_id: element.element_id,
                relative_error,
                max_relative_error: options.max_area_relative_error,
            });
        }

        *recovered_area_by_source_face
            .entry(element.source_face_id)
            .or_default() += area_m2;
        let actual_normal =
            triangle_unit_normal(points).ok_or(SurfaceRecoveryError::DegenerateElement {
                element_id: element.element_id,
            })?;
        let normal_alignment = dot(actual_normal, source_face.unit_normal).abs();
        let normal_alignment = if normal_alignment.is_finite() {
            normal_alignment
        } else {
            -1.0
        };
        min_normal_alignment = min_normal_alignment.min(normal_alignment);
        if normal_alignment < options.min_normal_alignment {
            return Err(SurfaceRecoveryError::NormalMismatch {
                element_id: element.element_id,
                alignment: normal_alignment,
                min_alignment: options.min_normal_alignment,
            });
        }

        if norm(source_face.unit_normal) <= tolerance.length_epsilon(1.0) {
            return Err(SurfaceRecoveryError::NormalMismatch {
                element_id: element.element_id,
                alignment: 0.0,
                min_alignment: options.min_normal_alignment,
            });
        }

        covered_source_faces.insert(element.source_face_id);
        for edge in triangle_edges(element.node_ids) {
            *edge_incidence.entry(edge).or_default() += 1;
        }
    }

    for source_face in source_faces.keys() {
        if !covered_source_faces.contains(source_face) {
            return Err(SurfaceRecoveryError::UncoveredSourceFace {
                source_face_id: *source_face,
            });
        }
    }

    for source_face in topology.faces.iter() {
        let recovered_area = recovered_area_by_source_face
            .get(&source_face.face_id)
            .copied()
            .unwrap_or(0.0);
        let expected_area = source_face.area_m2;
        let relative_error = if expected_area > 0.0 && expected_area.is_finite() {
            (recovered_area - expected_area).abs() / expected_area
        } else {
            0.0
        };
        max_area_relative_error = max_area_relative_error.max(relative_error);
        if relative_error > options.max_area_relative_error {
            return Err(SurfaceRecoveryError::SourceFaceAreaMismatch {
                source_face_id: source_face.face_id,
                relative_error,
                max_relative_error: options.max_area_relative_error,
            });
        }
    }

    let mut open_edge_count = 0_usize;
    let mut nonmanifold_edge_count = 0_usize;
    let mut recovered_edge_count = 0_usize;
    for (edge, count) in &edge_incidence {
        match *count {
            2 => recovered_edge_count += 1,
            0 | 1 => {
                open_edge_count += 1;
                if options.require_closed {
                    return Err(SurfaceRecoveryError::OpenEdge {
                        edge: *edge,
                        count: *count,
                    });
                }
            }
            _ => {
                nonmanifold_edge_count += 1;
                if options.require_closed {
                    return Err(SurfaceRecoveryError::NonManifoldEdge {
                        edge: *edge,
                        count: *count,
                    });
                }
            }
        }
    }

    Ok(SurfaceRecoveryReport {
        surface_element_count: surface.elements.len(),
        recovered_edge_count,
        open_edge_count,
        nonmanifold_edge_count,
        max_area_relative_error,
        min_normal_alignment: if min_normal_alignment.is_finite() {
            min_normal_alignment
        } else {
            1.0
        },
        source_face_coverage_ratio: covered_source_faces.len() as f64 / source_faces.len() as f64,
    })
}

fn validate_options(options: SurfaceRecoveryOptions) -> Result<(), SurfaceRecoveryError> {
    if !options.max_area_relative_error.is_finite()
        || options.max_area_relative_error < 0.0
        || !options.min_normal_alignment.is_finite()
        || !(0.0..=1.0).contains(&options.min_normal_alignment)
    {
        return Err(SurfaceRecoveryError::InvalidOptions);
    }
    Ok(())
}
