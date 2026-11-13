//! Section cuts with one or two planes, capped or open, and hidden regions left out of a chunk.
//!
//! A section keeps what lies on the positive side of its plane. With a second plane the view
//! removes only what lies on the removed side of both planes, a corner cutaway, so the kept part
//! is the union of the two kept half-spaces. It is cut as two disjoint pieces: what the first
//! plane keeps, and what the first removes but the second keeps. Each plane's cap is the solid's
//! cut face on that plane, trimmed to where the removed corner meets it (the other plane's
//! removed side). Without caps the cut stays open and shows the model's inside.

use super::section_cap::section_cap;
use super::{
    bounds_from_vertices, push_clipped_line, push_clipped_triangle, triangles_of,
    GeometrySceneTriangleRange,
};
use crate::core::{DrawCall, PipelineType, RenderData, Vertex};
use glam::Vec3;

type Plane = (Vec3, Vec3);

fn flipped((normal, origin): Plane) -> Plane {
    (-normal, origin)
}

/// `render_data` cut by `first` (and `second`, when given), with caps when `cap` is set. The
/// caps are cut from `cap_source` when given: the same faces in the colors a cap takes.
pub(super) fn cut(
    render_data: RenderData,
    cap_source: Option<&RenderData>,
    first: Plane,
    second: Option<Plane>,
    cap: bool,
) -> RenderData {
    match render_data.pipeline_type {
        PipelineType::Triangles => {
            let triangles = triangles_of(&render_data);
            let cap_triangles = cap_source.map(triangles_of);
            let cap_triangles = cap_triangles.as_deref().unwrap_or(triangles.as_slice());
            let mut kept = clip_triangles(&triangles, first);
            if let Some(second) = second {
                let removed_by_first = as_triangles(&clip_triangles(&triangles, flipped(first)));
                kept.extend(clip_triangles(&removed_by_first, second));
            }
            if cap {
                kept.extend(caps(cap_triangles, first, second));
            }
            with_vertices(render_data, kept)
        }
        PipelineType::Lines | PipelineType::LinesNoDepth => {
            let lines = lines_of(&render_data);
            let mut kept = clip_lines(&lines, first);
            if let Some(second) = second {
                let removed_by_first = as_lines(&clip_lines(&lines, flipped(first)));
                kept.extend(clip_lines(&removed_by_first, second));
            }
            with_vertices(render_data, kept)
        }
        _ => render_data,
    }
}

/// The caps where `first` (and `second`, when given) cut the closed meshes among `triangles`:
/// with two planes, each plane's cap trimmed to where the removed corner meets it.
pub(super) fn caps(triangles: &[[Vertex; 3]], first: Plane, second: Option<Plane>) -> Vec<Vertex> {
    let mut caps = section_cap(triangles, first);
    if let Some(second) = second {
        caps = clip_triangles(&as_triangles(&caps), flipped(second));
        let second_caps = section_cap(triangles, second);
        caps.extend(clip_triangles(&as_triangles(&second_caps), flipped(first)));
    }
    caps
}

impl super::GeometryScene {
    /// The section's caps, one node per body (mesh): each cut from the faces of all the body's
    /// chunks together, since a CAD import gives each face its own chunk and a single face is an
    /// open surface with no inside to fill. `cap_sources` holds each body's faces in the colors a
    /// cap takes (body or material color, never setup tints or selection).
    pub(super) fn section_cap_nodes(
        &self,
        presentation: &super::GeometryScenePresentation,
        cap_sources: &std::collections::BTreeMap<Option<String>, (crate::core::Material, Vec<[Vertex; 3]>)>,
    ) -> Vec<crate::core::SceneNode> {
        let Some(section) = presentation.active_section().filter(|section| section.cap) else {
            return Vec::new();
        };
        let Some(first) = super::section_plane(&section.plane) else {
            return Vec::new();
        };
        let second = section.second_plane.as_ref().and_then(super::section_plane);
        cap_sources
            .iter()
            .enumerate()
            .filter_map(|(index, (mesh_id, (material, triangles)))| {
                let vertices = caps(triangles, first, second);
                if vertices.is_empty() {
                    return None;
                }
                let chunk_id = format!("__section_cap:{}", mesh_id.as_deref().unwrap_or("scene"));
                let vertex_count = vertices.len();
                let bounds = bounds_from_vertices(&vertices);
                Some(crate::core::SceneNode {
                    id: super::stable_node_id(
                        &self.scene_id,
                        self.revision,
                        self.chunks.len() + 3 + index,
                        &chunk_id,
                    ),
                    name: "Section cap".to_string(),
                    transform: glam::Mat4::IDENTITY,
                    visible: true,
                    cast_shadows: false,
                    receive_shadows: false,
                    axes_index: 0,
                    parent: None,
                    children: Vec::new(),
                    render_data: Some(RenderData {
                        pipeline_type: PipelineType::Triangles,
                        vertices,
                        indices: None,
                        gpu_vertices: None,
                        bounds: Some(bounds),
                        material: material.clone(),
                        draw_calls: vec![DrawCall {
                            vertex_offset: 0,
                            vertex_count,
                            index_offset: None,
                            index_count: None,
                            instance_count: 1,
                        }],
                        image: None,
                    }),
                    bounds,
                    lod_levels: Vec::new(),
                    current_lod: 0,
                    instances: None,
                })
            })
            .collect()
    }
}

/// `render_data` less the primitives (`per` vertices each: 3 for triangles, 2 for segments)
/// whose index falls in one of `ranges`.
pub(super) fn without_primitives(
    render_data: RenderData,
    per: usize,
    ranges: &[GeometrySceneTriangleRange],
) -> RenderData {
    let hidden = |index: usize| {
        ranges.iter().any(|range| {
            index >= range.start as usize
                && index < (range.start as usize).saturating_add(range.count as usize)
        })
    };
    let mut kept = Vec::new();
    match render_data.indices.as_ref() {
        Some(indices) => {
            for (index, primitive) in indices.chunks_exact(per).enumerate() {
                if hidden(index) {
                    continue;
                }
                let vertices: Option<Vec<Vertex>> = primitive
                    .iter()
                    .map(|i| render_data.vertices.get(*i as usize).copied())
                    .collect();
                if let Some(vertices) = vertices {
                    kept.extend(vertices);
                }
            }
        }
        None => {
            for (index, primitive) in render_data.vertices.chunks_exact(per).enumerate() {
                if !hidden(index) {
                    kept.extend_from_slice(primitive);
                }
            }
        }
    }
    with_vertices(render_data, kept)
}

fn clip_triangles(triangles: &[[Vertex; 3]], plane: Plane) -> Vec<Vertex> {
    let mut out = Vec::new();
    for triangle in triangles {
        push_clipped_triangle(*triangle, plane, &mut out);
    }
    out
}

fn clip_lines(lines: &[[Vertex; 2]], plane: Plane) -> Vec<Vertex> {
    let mut out = Vec::new();
    for [a, b] in lines {
        push_clipped_line(*a, *b, plane, &mut out);
    }
    out
}

fn as_triangles(soup: &[Vertex]) -> Vec<[Vertex; 3]> {
    soup.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect()
}

fn as_lines(soup: &[Vertex]) -> Vec<[Vertex; 2]> {
    soup.chunks_exact(2).map(|l| [l[0], l[1]]).collect()
}

fn lines_of(render_data: &RenderData) -> Vec<[Vertex; 2]> {
    match render_data.indices.as_ref() {
        Some(indices) => indices
            .chunks_exact(2)
            .filter_map(|l| {
                Some([
                    *render_data.vertices.get(l[0] as usize)?,
                    *render_data.vertices.get(l[1] as usize)?,
                ])
            })
            .collect(),
        None => as_lines(&render_data.vertices),
    }
}

/// `render_data` drawing `vertices` as an unindexed list. A chunk whose draw calls were cleared
/// (shaded faces in a wireframe view) stays undrawn.
fn with_vertices(mut render_data: RenderData, vertices: Vec<Vertex>) -> RenderData {
    let drawn = !render_data.draw_calls.is_empty();
    let vertex_count = vertices.len();
    render_data.bounds = Some(bounds_from_vertices(&vertices));
    render_data.vertices = vertices;
    render_data.indices = None;
    render_data.draw_calls = if drawn {
        vec![DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: None,
            index_count: None,
            instance_count: 1,
        }]
    } else {
        Vec::new()
    };
    render_data
}
