//! `ScenePlot`: one plot of a figure scene, as it is serialized.

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScenePlot {
    Line {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        marker_style: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        marker_size: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        marker_face_color: Option<[f32; 4]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        marker_edge_color: Option<[f32; 4]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        marker_filled: Option<bool>,
        /// "off" for a line hidden from the legend; omitted when "on".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        handle_visibility: Option<String>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    ReferenceLine {
        orientation: String,
        #[serde(deserialize_with = "deserialize_f64_lossy")]
        value: f64,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        label: Option<String>,
        display_name: Option<String>,
        label_orientation: String,
        axes_index: u32,
        visible: bool,
        #[serde(default, skip_serializing_if = "is_false")]
        color_auto: bool,
    },
    Scatter {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        color_rgba: [f32; 4],
        marker_size: f32,
        marker_style: String,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Bar {
        labels: Vec<String>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        values: Vec<f64>,
        #[serde(default, deserialize_with = "deserialize_option_vec_f64_lossy")]
        histogram_bin_edges: Option<Vec<f64>>,
        #[serde(default)]
        polar_histogram: bool,
        #[serde(default)]
        polar_histogram_display_style: Option<String>,
        color_rgba: [f32; 4],
        #[serde(default)]
        outline_color_rgba: Option<[f32; 4]>,
        bar_width: f32,
        outline_width: f32,
        orientation: String,
        group_index: u32,
        group_count: u32,
        #[serde(default, deserialize_with = "deserialize_option_vec_f64_lossy")]
        stack_offsets: Option<Vec<f64>>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    ErrorBar {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        err_low: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        err_high: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x_err_low: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x_err_high: Vec<f64>,
        orientation: String,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        cap_width: f32,
        marker_style: Option<String>,
        marker_size: Option<f32>,
        marker_face_color: Option<[f32; 4]>,
        marker_edge_color: Option<[f32; 4]>,
        marker_filled: Option<bool>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Stairs {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        color_rgba: [f32; 4],
        line_width: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Stem {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        #[serde(deserialize_with = "deserialize_f64_lossy")]
        baseline: f64,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        baseline_color_rgba: [f32; 4],
        baseline_visible: bool,
        marker_color_rgba: [f32; 4],
        marker_size: f32,
        marker_filled: bool,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Area {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(default, deserialize_with = "deserialize_option_vec_f64_lossy")]
        lower_y: Option<Vec<f64>>,
        #[serde(deserialize_with = "deserialize_f64_lossy")]
        baseline: f64,
        color_rgba: [f32; 4],
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Quiver {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(default, deserialize_with = "deserialize_option_vec_f64_lossy")]
        z: Option<Vec<f64>>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        u: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        v: Vec<f64>,
        #[serde(default, deserialize_with = "deserialize_option_vec_f64_lossy")]
        w: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        z_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        u_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        v_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        w_source: Option<SceneNumericData>,
        color_rgba: [f32; 4],
        line_width: f32,
        scale: f32,
        head_size: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Surface {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(deserialize_with = "deserialize_matrix_f64_lossy")]
        z: Vec<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        z_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        c_source: Option<SceneNumericData>,
        #[serde(default)]
        x_grid: Option<Vec<Vec<f64>>>,
        #[serde(default)]
        y_grid: Option<Vec<Vec<f64>>>,
        colormap: String,
        shading_mode: String,
        wireframe: bool,
        alpha: f32,
        flatten_z: bool,
        #[serde(default)]
        image_mode: bool,
        /// Heatmap: the text printed in each cell, indexed like `z`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cell_labels: Option<Vec<Vec<String>>>,
        #[serde(default)]
        color_mapping: crate::plots::surface::SurfaceColorMapping,
        #[serde(default)]
        color_grid_rgba: Option<Vec<Vec<[f32; 4]>>>,
        #[serde(default, deserialize_with = "deserialize_option_pair_f64_lossy")]
        color_limits: Option<[f64; 2]>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Patch {
        #[serde(deserialize_with = "deserialize_vec_xyz_f32_lossy")]
        vertices: Vec<[f32; 3]>,
        faces: Vec<Vec<u32>>,
        face_color_rgba: [f32; 4],
        edge_color_rgba: [f32; 4],
        face_color_mode: String,
        edge_color_mode: String,
        face_alpha: f32,
        edge_alpha: f32,
        line_width: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
        #[serde(default)]
        force_3d: bool,
        #[serde(default)]
        color_mapping: crate::plots::surface::SurfaceColorMapping,
        #[serde(default)]
        color_source_kind: crate::plots::PatchColorSourceKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        z_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        c_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        faces_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vertices_source: Option<SceneNumericData>,
    },
    Mesh {
        #[serde(deserialize_with = "deserialize_vec_xyz_f32_lossy")]
        vertices: Vec<[f32; 3]>,
        triangles: Vec<[u32; 3]>,
        mesh_id: Option<String>,
        face_color_rgba: [f32; 4],
        edge_color_rgba: [f32; 4],
        face_alpha: f32,
        edge_alpha: f32,
        edge_width: f32,
        #[serde(default)]
        edge_mode: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        feature_edge_groups: Vec<u64>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        vertex_colors_rgba: Vec<[f32; 4]>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        triangle_colors_rgba: Vec<[f32; 4]>,
        /// Per-vertex surface normals from the source (CAD kernels' exact normals); empty
        /// shades with area-weighted triangle normals.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        vertex_normals: Vec<[f32; 3]>,
        axes_index: u32,
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        regions: Vec<SerializedMeshRegion>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        highlighted_region_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        highlight_color_rgba: Option<[f32; 4]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scalar_field: Option<Box<SerializedMeshScalarField>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vector_field: Option<Box<SerializedMeshVectorField>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deformation: Option<Box<SerializedMeshDeformation>>,
        /// Placements of the mesh (a CAD prototype drawn at each occurrence); the vertices are
        /// then the prototype's own coordinates. See `geometry_scene::instances`.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        instances: Vec<crate::geometry_scene::GeometrySceneInstance>,
        /// The projected diameters (device px) placements draw at: one level of detail.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lod_pixels: Option<[f32; 2]>,
        /// The mesh packed (base64), in place of `vertices`, `triangles` and `vertex_normals`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        packed: Option<packed::PackedMesh>,
        visible: bool,
    },
    Line3 {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        x: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        y: Vec<f64>,
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        z: Vec<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        z_source: Option<SceneNumericData>,
        color_rgba: [f32; 4],
        line_width: f32,
        line_style: String,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Scatter3 {
        #[serde(deserialize_with = "deserialize_vec_xyz_f32_lossy")]
        points: Vec<[f32; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y_source: Option<SceneNumericData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        z_source: Option<SceneNumericData>,
        #[serde(default, deserialize_with = "deserialize_vec_rgba_f32_lossy")]
        colors_rgba: Vec<[f32; 4]>,
        point_size: f32,
        #[serde(default, deserialize_with = "deserialize_option_vec_f32_lossy")]
        point_sizes: Option<Vec<f32>>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Contour {
        vertices: Vec<SerializedVertex>,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        base_z: f32,
        line_width: f32,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
        #[serde(default)]
        force_3d: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value_range: Option<[f64; 2]>,
    },
    ContourFill {
        vertices: Vec<SerializedVertex>,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        axes_index: u32,
        label: Option<String>,
        visible: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value_range: Option<[f64; 2]>,
    },
    Pie {
        #[serde(deserialize_with = "deserialize_vec_f64_lossy")]
        values: Vec<f64>,
        colors_rgba: Vec<[f32; 4]>,
        slice_labels: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        legend_labels: Vec<String>,
        label_format: Option<String>,
        explode: Vec<bool>,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
    Unsupported {
        plot_kind: PlotKind,
        axes_index: u32,
        label: Option<String>,
        visible: bool,
    },
}
