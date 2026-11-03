//! Volume views baked into ordinary scene chunks: a section through the part colored by the
//! field (the boundary clipped to the kept side, the cut capped), streamlines and arrow glyphs
//! of a vector field inside a see-through boundary. The geometry and values come from
//! [`runmat_plot::field_volume`], evaluated on the run's solver mesh with each element's own
//! value or basis; nothing is averaged. A viewer that draws triangles and lines shows them.

use super::super::{
    analysis_field_page_by_run_id_op, AnalysisFieldLocation, AnalysisFieldPageValues,
    AnalysisFieldRequestOptions, AnalysisFieldSelection, AnalysisRenderTopology,
};
use super::{append_feature_edges, scalar_color, FEATURE_EDGE_CREASE_DEGREES};
use crate::operations::OperationContext;
use glam::Vec4;
use runmat_plot::core::{AlphaMode, DrawCall, Material, PipelineType, RenderData, Vertex};
use runmat_plot::field_volume::{
    arrow_glyphs, section_cap, streamlines, FieldSupport, GlyphOptions, Reduction, SectionOptions,
    SolvedField, StreamlineOptions, TetMesh, TetOrder, TetrahedronLocator,
};
use runmat_plot::geometry_scene::{
    GeometrySceneCompleteness, GeometrySceneOverlay, GeometrySceneScalarLegend,
};
use runmat_plot::{GeometryScene, GeometrySceneChunk};
use serde::Serialize;

/// The poster camera looks from this direction (snapshot `camera.rs`); a section removes the
/// half of the part on this side so the cut faces the viewer.
const VIEW_DIRECTION: [f64; 3] = [0.35, -1.0, 0.45];
/// Triangles a section cap may hold before its refinement is reduced.
const CAP_TRIANGLE_BUDGET: usize = 600_000;
/// Opacity of the boundary drawn around streamlines and glyphs.
const CONTEXT_ALPHA: f32 = 0.18;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeViewKind {
    /// The part cut by a plane: the kept boundary and the cap colored by the field.
    Section,
    /// Streamlines of a vector field through a see-through boundary.
    Streamlines,
    /// Arrows of a vector field on the section plane through a see-through boundary.
    Glyphs,
}

/// What a volume view draws.
#[derive(Debug, Clone)]
pub struct VolumeViewRequest {
    pub field: AnalysisFieldSelection,
    pub kind: VolumeViewKind,
    pub reduction: Reduction,
    /// Legend range (the field's true extrema); the drawn values' range when absent.
    pub range: Option<(f64, f64)>,
    pub legend: String,
    pub unit: String,
}

/// What a volume view leaves out of the solution, stated with the view.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct VolumeViewError {
    /// Largest difference between the field and the linear colors drawn between evaluated
    /// points of the section, in field units.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_error: Option<f64>,
    /// Largest departure of a curved quadratic element used by the view from the straight-sided
    /// tetrahedron its cut or point location uses, in metres.
    pub geometric_error_m: f64,
    /// Largest estimated accumulated RK4 error of a streamline, in metres.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integration_error_m: Option<f64>,
    /// Elements cut, lines traced or arrows drawn.
    pub count: usize,
    /// The plane of a section or glyph view: origin and the normal pointing to the kept side.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plane: Option<[[f64; 3]; 2]>,
}

/// One view of `request` for the run, from its canonical solver mesh artifact.
pub fn build_volume_scene(
    run_id: &str,
    solver_mesh: &str,
    request: &VolumeViewRequest,
) -> Result<(GeometryScene, VolumeViewError), String> {
    let artifact = super::super::solver_mesh_artifact::load_solver_mesh_artifact(
        std::path::Path::new(solver_mesh),
    )
    .map_err(|error| error.to_string())?;
    let mesh = tet_mesh(&artifact)?;
    let topology = super::super::render_topology_from_solver_mesh(Some(&artifact))
        .ok_or_else(|| format!("FEA run '{run_id}' has no boundary to draw"))?;
    drop(artifact);
    let field = load_solved_field(run_id, &request.field, &mesh)?;
    let (origin, normal) = section_plane(&mesh);
    let mut error = VolumeViewError::default();
    let mut chunks = Vec::new();
    let mut drawn = Vec::new();
    match request.kind {
        VolumeViewKind::Section => {
            let cap = refined_cap(&mesh, &field, request.reduction, origin, normal);
            error.display_error = Some(cap.display_error);
            error.geometric_error_m = cap.geometric_error_m;
            error.count = cap.cut_elements;
            error.plane = Some([origin, normal]);
            if cap.triangles.is_empty() {
                return Err("the section plane cuts no element".into());
            }
            drawn.extend(cap.triangles.values.iter().copied());
            let surface = surface_values(&topology, &mesh, &field, request.reduction);
            drawn.extend(surface.iter().flatten().copied());
            let range = legend_range(request.range, &drawn)?;
            chunks.push(clipped_surface(
                &topology,
                field.support,
                &surface,
                range,
                origin,
                normal,
            )?);
            chunks.push(triangles_chunk(
                "section-cap",
                "Section",
                &cap.triangles,
                range,
                1.0,
            ));
        }
        VolumeViewKind::Streamlines | VolumeViewKind::Glyphs => {
            if field.components != 3 {
                return Err(format!(
                    "field '{}' is not a vector field",
                    request.field.field_id
                ));
            }
            let locator = TetrahedronLocator::new(&mesh);
            match request.kind {
                VolumeViewKind::Streamlines => {
                    let options = StreamlineOptions::for_mesh(&locator);
                    let lines = streamlines(&locator, &field, request.reduction, 0.0, &options);
                    if lines.lines.is_empty() {
                        return Err("no streamline could be traced".into());
                    }
                    error.integration_error_m = Some(lines.integration_error_m);
                    error.geometric_error_m = lines.geometric_error_m;
                    error.count = lines.lines.len();
                    drawn.extend(
                        lines
                            .lines
                            .iter()
                            .flat_map(|line| line.values.iter().copied()),
                    );
                    let range = legend_range(request.range, &drawn)?;
                    chunks.push(context_surface(&topology)?);
                    chunks.push(lines_chunk(&lines.lines, range));
                }
                _ => {
                    let (min, max) = mesh.bounds();
                    let extent = (0..3).map(|axis| max[axis] - min[axis]).fold(0.0, f64::max);
                    let options = GlyphOptions {
                        origin,
                        normal,
                        spacing: extent / 24.0,
                        min_relative_magnitude: 0.02,
                        segments: 8,
                    };
                    let glyphs = arrow_glyphs(&locator, &field, request.reduction, 0.0, &options);
                    if glyphs.arrows == 0 {
                        return Err("no arrow lies inside the part on the section plane".into());
                    }
                    error.geometric_error_m = glyphs.geometric_error_m;
                    error.count = glyphs.arrows;
                    error.plane = Some([origin, normal]);
                    drawn.extend(glyphs.triangles.values.iter().copied());
                    let range = legend_range(request.range, &drawn)?;
                    chunks.push(context_surface(&topology)?);
                    chunks.push(triangles_chunk(
                        "glyphs",
                        "Arrows",
                        &glyphs.triangles,
                        range,
                        1.0,
                    ));
                }
            }
        }
    }
    let range = legend_range(request.range, &drawn)?;
    let mut scene = GeometryScene::new(
        format!("fea:{run_id}:{}", request.field.field_id),
        1,
        chunks,
    );
    if request.kind == VolumeViewKind::Section {
        append_feature_edges(&mut scene, FEATURE_EDGE_CREASE_DEGREES);
    }
    let triangle_count = scene
        .chunks
        .iter()
        .filter_map(|chunk| chunk.indices.as_ref().map(|indices| indices.len() / 3))
        .sum();
    let vertex_count = scene.chunks.iter().map(|chunk| chunk.vertices.len()).sum();
    scene.overlay = Some(GeometrySceneOverlay {
        source_name: None,
        status: GeometrySceneCompleteness::Complete,
        quality_label: None,
        format: None,
        source_label: None,
        allow_create_fea_study: false,
        byte_count: None,
        mesh_count: topology.meshes.len(),
        vertex_count,
        triangle_count,
        progress_percent: None,
        scalar_legend: Some(GeometrySceneScalarLegend {
            label: request.legend.clone(),
            unit: request.unit.clone(),
            minimum: range.0,
            maximum: range.1,
        }),
        region_count: 0,
        mapped_region_count: 0,
        assembly_nodes: Vec::new(),
        regions: Vec::new(),
        warnings: Vec::new(),
    });
    Ok((scene, error))
}

/// The solver mesh's volume elements in solver node order. Every element shares one order.
fn tet_mesh(artifact: &runmat_meshing_core::SolverMeshArtifact) -> Result<TetMesh, String> {
    let topology = &artifact.topology;
    let index_by_id: std::collections::HashMap<u64, u32> = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index as u32))
        .collect();
    let order = match topology
        .volume_elements
        .first()
        .map(|element| element.order)
    {
        Some(runmat_meshing_core::ElementOrder::Tet4) => TetOrder::Linear,
        Some(runmat_meshing_core::ElementOrder::Tet10) => TetOrder::Quadratic,
        None => return Err("the solver mesh has no volume elements".into()),
    };
    let mut elements = Vec::with_capacity(topology.volume_elements.len() * order.node_count());
    for element in &topology.volume_elements {
        if element.node_ids.len() != order.node_count() {
            return Err("the solver mesh mixes element orders".into());
        }
        for id in &element.node_ids {
            elements.push(*index_by_id.get(id).ok_or("element node is missing")?);
        }
    }
    let positions = topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect();
    TetMesh::new(positions, order, elements).map_err(|error| error.to_string())
}

/// Every stored value of the field, real and imaginary parts kept apart (a phasor is drawn from
/// its interpolated parts, never from interpolated magnitudes).
fn load_solved_field(
    run_id: &str,
    selection: &AnalysisFieldSelection,
    mesh: &TetMesh,
) -> Result<SolvedField, String> {
    const PAGE: usize = 4096;
    let mut real = Vec::new();
    let mut imaginary: Option<Vec<f64>> = None;
    let mut offset = 0;
    let mut descriptor = None;
    loop {
        let page = analysis_field_page_by_run_id_op(
            run_id,
            selection.clone(),
            AnalysisFieldRequestOptions {
                offset: Some(offset),
                limit: Some(PAGE),
            },
            OperationContext::new(None, None),
        )
        .map_err(|error| format!("{}: {}", error.error_code, error.message))?
        .data;
        if page.offset != offset {
            return Err(format!(
                "field '{}' page is out of order",
                selection.field_id
            ));
        }
        match page.values {
            AnalysisFieldPageValues::Real(values) => real.extend(values),
            AnalysisFieldPageValues::ComplexInterleaved(values) => {
                let parts = imaginary.get_or_insert_with(Vec::new);
                for pair in values.chunks_exact(2) {
                    real.push(pair[0]);
                    parts.push(pair[1]);
                }
            }
        }
        offset += page.count;
        let total = page.total_count;
        descriptor.get_or_insert(page.descriptor);
        if page.count == 0 || offset >= total {
            break;
        }
    }
    let descriptor = descriptor.ok_or("field has no pages")?;
    let support = match descriptor.location {
        AnalysisFieldLocation::Node => FieldSupport::Node,
        AnalysisFieldLocation::Element => FieldSupport::Element,
        other => {
            return Err(format!(
                "field '{}' lives at {other:?}; volume views draw node and element fields",
                selection.field_id
            ))
        }
    };
    let field = SolvedField {
        support,
        components: descriptor.component_count.unwrap_or(1),
        real,
        imaginary,
    };
    // An element field with several values per element (integration points) is refused rather
    // than averaged.
    field
        .validate(mesh)
        .map_err(|error| format!("field '{}': {error}", selection.field_id))?;
    Ok(field)
}

/// The plane through the centre of the bounds across the part's thinnest axis: along a pipe,
/// duct or busbar it shows the profile over the length. Its normal points away from the
/// viewer, so the removed half is the one in front.
fn section_plane(mesh: &TetMesh) -> ([f64; 3], [f64; 3]) {
    let (min, max) = mesh.bounds();
    let origin = [0, 1, 2].map(|axis| 0.5 * (min[axis] + max[axis]));
    let axis = (0..3)
        .min_by(|a, b| (max[*a] - min[*a]).total_cmp(&(max[*b] - min[*b])))
        .unwrap_or(2);
    let mut normal = [0.0; 3];
    normal[axis] = if VIEW_DIRECTION[axis] > 0.0 {
        -1.0
    } else {
        1.0
    };
    (origin, normal)
}

/// The cap at the finest grid within the triangle budget (each element's polygon is evaluated
/// on an n x n grid per triangle; element fields need one).
fn refined_cap(
    mesh: &TetMesh,
    field: &SolvedField,
    reduction: Reduction,
    origin: [f64; 3],
    normal: [f64; 3],
) -> runmat_plot::field_volume::SectionCap {
    let mut subdivisions = match (field.support, mesh.order()) {
        (FieldSupport::Element, _) => 1,
        (FieldSupport::Node, TetOrder::Quadratic) => 4,
        (FieldSupport::Node, TetOrder::Linear) => 2,
    };
    loop {
        let cap = section_cap(
            mesh,
            field,
            reduction,
            &SectionOptions {
                origin,
                normal,
                subdivisions,
            },
        );
        if cap.triangles.triangle_count() <= CAP_TRIANGLE_BUDGET || subdivisions == 1 {
            return cap;
        }
        subdivisions /= 2;
    }
}

/// The reduced field at each boundary vertex (nodal) or triangle (element), per mesh.
fn surface_values(
    topology: &AnalysisRenderTopology,
    mesh: &TetMesh,
    field: &SolvedField,
    reduction: Reduction,
) -> Vec<Vec<f64>> {
    topology
        .meshes
        .iter()
        .map(|render| match field.support {
            FieldSupport::Node => render
                .vertex_volume_node_indices
                .iter()
                .map(|node| {
                    node.map_or(f64::NAN, |node| reduction.apply(&field.entity_value(node)))
                })
                .collect(),
            FieldSupport::Element => render
                .triangle_volume_element_indices
                .iter()
                .map(|element| {
                    element
                        .filter(|element| *element < mesh.element_count())
                        .map_or(f64::NAN, |element| {
                            reduction.apply(&field.entity_value(element))
                        })
                })
                .collect(),
        })
        .collect()
}

fn legend_range(declared: Option<(f64, f64)>, drawn: &[f64]) -> Result<(f64, f64), String> {
    if let Some(range) = declared {
        return Ok(range);
    }
    let finite = drawn.iter().copied().filter(|value| value.is_finite());
    let (min, max) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    if min <= max {
        Ok((min, max))
    } else {
        Err("the view draws no finite value".into())
    }
}

fn vertex(position: [f64; 3], normal: [f32; 3], color: [f32; 4]) -> Vertex {
    Vertex {
        position: position.map(|v| v as f32),
        color,
        normal,
        tex_coords: [0.0, 0.0],
    }
}

/// The boundary on the kept side of the plane, colored by the field, each triangle that crosses
/// the plane clipped exactly at it (positions and colors interpolated along the cut edges).
fn clipped_surface(
    topology: &AnalysisRenderTopology,
    support: FieldSupport,
    values: &[Vec<f64>],
    range: (f64, f64),
    origin: [f64; 3],
    normal: [f64; 3],
) -> Result<GeometrySceneChunk, String> {
    let mut vertices = Vec::new();
    for (render, values) in topology.meshes.iter().zip(values) {
        for (index, triangle) in render.triangles.iter().enumerate() {
            let corners = triangle.map(|node| render.vertices[node as usize]);
            // Nodal values are indexed by vertex, element values by triangle.
            let value_at = |corner: usize| match support {
                FieldSupport::Node => values[triangle[corner] as usize],
                FieldSupport::Element => values[index],
            };
            let colors = [0, 1, 2].map(|corner| match value_at(corner) {
                value if value.is_finite() => scalar_color(value, range.0, range.1),
                // A boundary vertex outside the field's support (another body).
                _ => [0.66, 0.73, 0.83, 1.0],
            });
            let face = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
            let face_normal = normalize(face).map(|v| v as f32);
            let side = corners.map(|p| dot(sub(p, origin), normal));
            let mut polygon: Vec<([f64; 3], [f32; 4])> = Vec::with_capacity(4);
            for k in 0..3 {
                let next = (k + 1) % 3;
                if side[k] >= 0.0 {
                    polygon.push((corners[k], colors[k]));
                }
                if (side[k] >= 0.0) != (side[next] >= 0.0) {
                    let t = side[k] / (side[k] - side[next]);
                    let position =
                        [0, 1, 2].map(|a| corners[k][a] + t * (corners[next][a] - corners[k][a]));
                    let color: [f32; 4] = std::array::from_fn(|c| {
                        colors[k][c] + t as f32 * (colors[next][c] - colors[k][c])
                    });
                    polygon.push((position, color));
                }
            }
            for k in 1..polygon.len().saturating_sub(1) {
                for (position, color) in [polygon[0], polygon[k], polygon[k + 1]] {
                    vertices.push(vertex(position, face_normal, color));
                }
            }
        }
    }
    if vertices.is_empty() {
        return Err("the section removes the whole boundary".into());
    }
    let indices = (0..vertices.len() as u32).collect();
    let mut chunk = GeometrySceneChunk::indexed_triangles(
        "section-surface",
        vertices,
        indices,
        Material {
            albedo: Vec4::ONE,
            double_sided: true,
            ..Material::default()
        },
    );
    chunk.label = Some("Boundary".into());
    Ok(chunk)
}

/// The whole boundary, light and see-through, as context for lines and arrows inside.
fn context_surface(topology: &AnalysisRenderTopology) -> Result<GeometrySceneChunk, String> {
    let color = [0.70, 0.76, 0.84, CONTEXT_ALPHA];
    let mut vertices = Vec::new();
    for render in &topology.meshes {
        for triangle in &render.triangles {
            let corners = triangle.map(|node| render.vertices[node as usize]);
            let face_normal = normalize(cross(
                sub(corners[1], corners[0]),
                sub(corners[2], corners[0]),
            ))
            .map(|v| v as f32);
            for corner in corners {
                vertices.push(vertex(corner, face_normal, color));
            }
        }
    }
    if vertices.is_empty() {
        return Err("FEA run has no boundary to draw".into());
    }
    let indices = (0..vertices.len() as u32).collect();
    let mut chunk = GeometrySceneChunk::indexed_triangles(
        "context-surface",
        vertices,
        indices,
        Material {
            albedo: Vec4::new(1.0, 1.0, 1.0, CONTEXT_ALPHA),
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            ..Material::default()
        },
    );
    chunk.label = Some("Boundary".into());
    Ok(chunk)
}

fn triangles_chunk(
    id: &str,
    label: &str,
    triangles: &runmat_plot::field_volume::FieldTriangles,
    range: (f64, f64),
    alpha: f32,
) -> GeometrySceneChunk {
    let vertices = triangles
        .positions
        .iter()
        .zip(&triangles.normals)
        .zip(&triangles.values)
        .map(|((position, normal), value)| {
            let mut color = scalar_color(*value, range.0, range.1);
            color[3] = alpha;
            Vertex {
                position: *position,
                color,
                normal: *normal,
                tex_coords: [0.0, 0.0],
            }
        })
        .collect();
    let mut chunk = GeometrySceneChunk::indexed_triangles(
        id,
        vertices,
        triangles.indices.clone(),
        Material {
            albedo: Vec4::ONE,
            double_sided: true,
            ..Material::default()
        },
    );
    chunk.label = Some(label.into());
    chunk
}

fn lines_chunk(
    lines: &[runmat_plot::field_volume::FieldPolyline],
    range: (f64, f64),
) -> GeometrySceneChunk {
    let mut vertices = Vec::new();
    for line in lines {
        for k in 1..line.points.len() {
            for j in [k - 1, k] {
                vertices.push(Vertex {
                    position: line.points[j],
                    color: scalar_color(line.values[j], range.0, range.1),
                    normal: [0.0, 0.0, 0.0],
                    tex_coords: [0.0, 0.0],
                });
            }
        }
    }
    let vertex_count = vertices.len();
    let mut chunk = GeometrySceneChunk::from_render_data(
        "streamlines",
        RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices: None,
            bounds: None,
            material: Material {
                albedo: Vec4::ONE,
                ..Material::default()
            },
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count,
                index_offset: None,
                index_count: None,
                instance_count: 1,
            }],
            image: None,
        },
    );
    chunk.label = Some("Streamlines".into());
    chunk
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let length = dot(a, a).sqrt();
    if length > 0.0 {
        a.map(|v| v / length)
    } else {
        [0.0, 0.0, 1.0]
    }
}
