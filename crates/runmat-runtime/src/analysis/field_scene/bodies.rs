//! A result surface split by solved body: each body's triangles form their own chunks, tagged
//! with the body's study region (a scene region and an owner node), drawn in field colour or as
//! translucent grey context; hidden bodies are left out.
use super::regions::{RegionDraw, ViewRegions, BODY_REGION_TAG};
use super::{field_magnitude, FieldData};
use crate::analysis::{AnalysisFieldLocation, AnalysisRenderMesh};
use glam::Vec4;
use runmat_plot::core::{AlphaMode, Material, Vertex};
use runmat_plot::geometry_scene::{
    GeometrySceneOverlay, GeometrySceneRegion, GeometrySceneRegionSummary,
    GeometrySceneTriangleRange,
};
use runmat_plot::GeometrySceneChunk;

/// The colour of a surface with no field value.
pub(super) const UNCOLORED: [f32; 4] = [0.66, 0.73, 0.83, 1.0];
/// A context body: translucent grey, no field colour.
pub(super) const CONTEXT_COLOR: [f32; 4] = [0.62, 0.64, 0.68, 0.3];

/// The triangles of one mesh that one body contributes, in mesh order.
pub(super) struct BodyGroup {
    pub draw: RegionDraw,
    /// The body's region name; `None` for triangles the study names no body for.
    pub region: Option<String>,
    pub triangles: Vec<usize>,
}

/// The drawn triangles of `mesh` grouped by body (triangles of no named body first), with
/// `draws` giving each triangle's draw. Without bodies, one group holds every triangle.
pub(super) fn body_groups(
    mesh: &AnalysisRenderMesh,
    draws: &[RegionDraw],
    bodies: Option<&super::regions::BodyRegions>,
) -> Vec<BodyGroup> {
    let Some(bodies) = bodies.filter(|bodies| !bodies.is_empty()) else {
        return vec![BodyGroup {
            draw: RegionDraw::Field,
            region: None,
            triangles: (0..mesh.triangles.len()).collect(),
        }];
    };
    let mut groups: Vec<BodyGroup> = std::iter::once(None)
        .chain(bodies.names.iter().map(Some))
        .map(|region| BodyGroup {
            draw: RegionDraw::Field,
            region: region.cloned(),
            triangles: Vec::new(),
        })
        .collect();
    for (triangle, draw) in draws.iter().enumerate() {
        if *draw == RegionDraw::Hidden {
            continue;
        }
        let body = mesh
            .triangle_volume_element_indices
            .get(triangle)
            .copied()
            .flatten()
            .and_then(|element| bodies.of_element(element));
        let group = &mut groups[body.map_or(0, |index| index + 1)];
        group.draw = *draw;
        group.triangles.push(triangle);
    }
    groups.retain(|group| !group.triangles.is_empty());
    groups
}

impl BodyGroup {
    /// One chunk of this group's triangles, batch `batch` of `mesh`.
    pub(super) fn chunk(
        &self,
        mesh: &AnalysisRenderMesh,
        batch: usize,
        vertices: Vec<Vertex>,
    ) -> GeometrySceneChunk {
        let context = self.draw == RegionDraw::Context;
        let material = Material {
            albedo: Vec4::ONE,
            double_sided: true,
            alpha_mode: if context {
                AlphaMode::Blend
            } else {
                Material::default().alpha_mode
            },
            ..Material::default()
        };
        let triangles = (vertices.len() / 3) as u32;
        let indices = (0..vertices.len() as u32).collect();
        let chunk_id = match &self.region {
            Some(region) => format!("{}:surface:{region}:{batch}", mesh.mesh_id),
            None => format!("{}:surface:{batch}", mesh.mesh_id),
        };
        let mut chunk =
            GeometrySceneChunk::indexed_triangles(chunk_id, vertices, indices, material);
        chunk.mesh_id = Some(mesh.mesh_id.clone());
        if let Some(region) = &self.region {
            chunk.label = Some(region.clone());
            chunk.regions = vec![GeometrySceneRegion::new(
                region.clone(),
                Some(region.clone()),
                Some(BODY_REGION_TAG.into()),
                vec![GeometrySceneTriangleRange::new(0, triangles)],
            )];
            chunk.owner_node_ids = vec![region.clone()];
        }
        chunk
    }
}

/// The field magnitudes a drawn triangle contributes to the legend's extremes: its corner nodes'
/// values for a nodal field, its element's value for an element field.
pub(super) fn triangle_samples(
    mesh: &AnalysisRenderMesh,
    field: &FieldData,
    triangle_index: usize,
    triangle: &[u32; 3],
    element_stride: usize,
) -> Result<Vec<f64>, String> {
    Ok(match field.location {
        AnalysisFieldLocation::Element => {
            triangle_corner_samples(mesh, field, triangle_index, triangle, element_stride)?[0]
                .into_iter()
                .collect()
        }
        _ => triangle_corner_samples(mesh, field, triangle_index, triangle, element_stride)?
            .into_iter()
            .flatten()
            .collect(),
    })
}

/// The field magnitude drawn at each corner of a triangle.
pub(super) fn triangle_corner_samples(
    mesh: &AnalysisRenderMesh,
    field: &FieldData,
    triangle_index: usize,
    triangle: &[u32; 3],
    element_stride: usize,
) -> Result<[Option<f64>; 3], String> {
    let mut samples = [None; 3];
    for corner in 0..3 {
        let entity = match field.location {
            AnalysisFieldLocation::Node => mesh
                .vertex_volume_node_indices
                .get(triangle[corner] as usize)
                .copied()
                .flatten(),
            AnalysisFieldLocation::Element => mesh
                .triangle_volume_element_indices
                .get(triangle_index)
                .copied()
                .flatten(),
            _ => None,
        };
        samples[corner] = entity
            .map(|index| field_magnitude(field, index, element_stride))
            .transpose()?;
    }
    Ok(samples)
}

/// List the solved bodies a scene draws in its `overlay`, with how each is drawn, and warn of a
/// hidden or context region no solved body carries.
pub(super) fn summarize(
    overlay: &mut GeometrySceneOverlay,
    chunks: &[GeometrySceneChunk],
    regions: &ViewRegions,
) {
    let mut summaries: Vec<GeometrySceneRegionSummary> = Vec::new();
    for chunk in chunks {
        for region in &chunk.regions {
            let count = region
                .triangle_ranges
                .iter()
                .map(|range| range.count as usize)
                .sum::<usize>();
            match summaries
                .iter_mut()
                .find(|summary| summary.region_id == region.region_id)
            {
                Some(summary) => summary.triangle_count += count,
                None => summaries.push(GeometrySceneRegionSummary {
                    region_id: region.region_id.clone(),
                    label: region.label.clone().unwrap_or(region.region_id.clone()),
                    tag: region.tag.clone(),
                    kind: Some(
                        if regions.context.contains(&region.region_id) {
                            "context"
                        } else {
                            "body"
                        }
                        .into(),
                    ),
                    triangle_count: count,
                }),
            }
        }
    }
    overlay.region_count = summaries.len();
    overlay.mapped_region_count = summaries.len();
    overlay.regions = summaries;
    for name in regions.unknown_regions() {
        overlay.warnings.push(format!(
            "the study's display names region '{name}', which no solved body carries"
        ));
    }
}
