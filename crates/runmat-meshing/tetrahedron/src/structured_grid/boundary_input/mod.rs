use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{EntityKind, GeometryAsset, UnitSystem};
use runmat_meshing_core::{
    BoundaryMeshInput, BoundaryMeshTriangle, MeshEntityProvenance, SourceEntityKind,
};

mod error;
mod weld;

pub(super) use error::BoundaryMeshInputError;
use weld::weld_surface_vertices;

pub(super) fn boundary_input_from_geometry(
    geometry: &GeometryAsset,
) -> Result<BoundaryMeshInput, BoundaryMeshInputError> {
    let surface = geometry
        .surface_meshes
        .iter()
        .min_by(|left, right| left.mesh_id.cmp(&right.mesh_id))
        .ok_or(BoundaryMeshInputError::NoSurfaceMeshes)?;
    if surface.vertices.is_empty() || surface.triangles.is_empty() {
        return Err(BoundaryMeshInputError::EmptySurfaceMesh {
            mesh_id: surface.mesh_id.clone(),
        });
    }

    let coordinate_scale = geometry_unit_scale_to_meters(geometry.units);
    let vertices_m = surface
        .vertices
        .iter()
        .map(|vertex| {
            [
                vertex[0] * coordinate_scale,
                vertex[1] * coordinate_scale,
                vertex[2] * coordinate_scale,
            ]
        })
        .collect::<Vec<_>>();

    let mut min = vertices_m[0];
    let mut max = vertices_m[0];
    for (vertex_index, vertex) in vertices_m.iter().copied().enumerate() {
        if vertex.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(BoundaryMeshInputError::NonFiniteVertex {
                mesh_id: surface.mesh_id.clone(),
                vertex_index,
            });
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    if (0..3).any(|axis| max[axis] <= min[axis]) {
        return Err(BoundaryMeshInputError::DegenerateBounds {
            mesh_id: surface.mesh_id.clone(),
        });
    }

    let (vertices, vertex_map) = weld_surface_vertices(&vertices_m, min, max);
    let mut welded_triangles = Vec::with_capacity(surface.triangles.len());
    let mut edge_incidence = BTreeMap::<[u32; 2], u32>::new();
    for (triangle_id, triangle) in surface.triangles.iter().copied().enumerate() {
        if triangle
            .iter()
            .any(|node_id| *node_id as usize >= surface.vertices.len())
        {
            return Err(BoundaryMeshInputError::TriangleIndexOutOfBounds {
                mesh_id: surface.mesh_id.clone(),
                triangle_id: triangle_id as u32,
            });
        }
        let triangle = [
            vertex_map[triangle[0] as usize],
            vertex_map[triangle[1] as usize],
            vertex_map[triangle[2] as usize],
        ];
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0] {
            continue;
        }
        for edge in triangle_edges(triangle) {
            *edge_incidence.entry(edge).or_insert(0) += 1;
        }
        welded_triangles.push((triangle_id, triangle));
    }
    if welded_triangles.is_empty() {
        return Err(BoundaryMeshInputError::EmptySurfaceMesh {
            mesh_id: surface.mesh_id.clone(),
        });
    }
    for (edge, count) in edge_incidence {
        if count != 2 {
            return Err(BoundaryMeshInputError::OpenBoundaryEdge {
                mesh_id: surface.mesh_id.clone(),
                edge,
                count,
            });
        }
    }

    let mut all_region_ids = geometry
        .regions
        .iter()
        .map(|region| region.region_id.clone())
        .collect::<Vec<_>>();
    all_region_ids.sort();
    all_region_ids.dedup();

    let mut triangles = Vec::with_capacity(welded_triangles.len());
    for (triangle_id, node_ids) in welded_triangles {
        let mut region_ids = geometry
            .region_entity_mappings
            .iter()
            .filter(|mapping| {
                mapping.mesh_id == surface.mesh_id
                    && matches!(mapping.entity_kind, EntityKind::Face | EntityKind::Element)
                    && mapping.contains_entity(triangle_id as u64)
            })
            .map(|mapping| mapping.region_id.clone())
            .collect::<Vec<_>>();
        region_ids.sort();
        region_ids.dedup();
        if region_ids.is_empty() {
            region_ids = all_region_ids.clone();
        }
        triangles.push(BoundaryMeshTriangle {
            triangle_id: triangle_id as u32,
            node_ids,
            region_ids: region_ids.clone(),
            provenance: vec![MeshEntityProvenance {
                source_geometry_id: geometry.geometry_id.clone(),
                source_geometry_revision: geometry.revision,
                source_entity_kind: SourceEntityKind::Face,
                source_entity_id: triangle_id.to_string(),
                region_ids,
            }],
        });
    }

    let region_ids = triangles
        .iter()
        .flat_map(|triangle| triangle.region_ids.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    Ok(BoundaryMeshInput {
        mesh_id: surface.mesh_id.clone(),
        source_geometry_id: geometry.geometry_id.clone(),
        source_geometry_revision: geometry.revision,
        source_geometry_sha256: Some(geometry.source.sha256.clone()),
        vertices,
        triangles,
        bounds_min_m: min,
        bounds_max_m: max,
        region_ids,
    })
}

fn geometry_unit_scale_to_meters(units: UnitSystem) -> f64 {
    match units {
        UnitSystem::Meter | UnitSystem::Unspecified => 1.0,
        UnitSystem::Millimeter => 0.001,
        UnitSystem::Inch => 0.0254,
    }
}

fn triangle_edges(triangle: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_edge(triangle[0], triangle[1]),
        sorted_edge(triangle[1], triangle[2]),
        sorted_edge(triangle[2], triangle[0]),
    ]
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    [left.min(right), left.max(right)]
}

#[cfg(test)]
mod tests;
