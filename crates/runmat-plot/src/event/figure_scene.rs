//! Figure snapshots and scenes: building them from a figure, and restoring a figure from them.

use super::*;

impl FigureSnapshot {
    /// Capture a snapshot from a [`Figure`] reference.
    pub fn capture(figure: &Figure) -> Self {
        let (rows, cols) = figure.axes_grid();
        let mut axes_grid_cells = (0..figure.axes_count())
            .map(|axes| figure.axes_grid_cell(axes))
            .collect::<Vec<_>>();
        if axes_grid_cells.iter().all(Option::is_none) {
            axes_grid_cells.clear();
        }
        let layout = FigureLayout {
            axes_rows: rows as u32,
            axes_cols: cols as u32,
            axes_indices: figure
                .plot_axes_indices()
                .iter()
                .map(|idx| *idx as u32)
                .collect(),
            axes_grid_cells,
        };

        let metadata = FigureMetadata::from_figure(figure);

        let plots = figure
            .plots()
            .enumerate()
            .map(|(idx, plot)| PlotDescriptor::from_plot(plot, figure_axis_index(figure, idx)))
            .collect();

        Self {
            layout,
            metadata,
            plots,
        }
    }

    pub fn fingerprint(&self) -> String {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let bytes = serde_json::to_vec(self).unwrap_or_default();
        let mut hash = FNV_OFFSET_BASIS;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        format!("fig:{hash:016x}")
    }
}

impl FigureScene {
    pub const SCHEMA_VERSION: u32 = 7;

    pub fn capture(figure: &Figure) -> Self {
        let snapshot = FigureSnapshot::capture(figure);
        let plots = figure
            .plots()
            .enumerate()
            .map(|(idx, plot)| ScenePlot::from_plot(plot, figure_axis_index(figure, idx)))
            .collect();

        Self {
            schema_version: Self::SCHEMA_VERSION,
            geometry_overlay: None,
            geometry_physical_context: None,
            geometry_lines: Vec::new(),
            geometry_orientation_triad: true,
            layout: snapshot.layout,
            metadata: snapshot.metadata,
            plots,
        }
    }

    pub async fn capture_for_export(
        figure: &Figure,
        policy: SceneExportPolicy,
    ) -> Result<Self, SceneExportError> {
        Self::capture_for_export_with_cause(figure, policy)
            .await
            .map_err(|failure| failure.error)
    }

    /// Capture an export with the original plot-data failure attached for callers that need it.
    pub async fn capture_for_export_with_cause(
        figure: &Figure,
        policy: SceneExportPolicy,
    ) -> Result<Self, SceneExportFailure> {
        let snapshot = FigureSnapshot::capture(figure);
        let mut budget = SceneExportBudget::new(policy);
        let mut plots = Vec::new();

        for (idx, plot) in figure.plots().enumerate() {
            let scene_plot =
                ScenePlot::from_plot_for_export(plot, figure_axis_index(figure, idx)).await?;
            budget.reserve_plot(&scene_plot)?;
            plots.push(scene_plot);
        }

        Ok(Self {
            schema_version: Self::SCHEMA_VERSION,
            geometry_overlay: None,
            geometry_physical_context: None,
            geometry_lines: Vec::new(),
            geometry_orientation_triad: true,
            layout: snapshot.layout,
            metadata: snapshot.metadata,
            plots,
        })
    }

    /// The largest distance between a saved position and the scene's own position when
    /// [`Self::from_geometry_scene`] exports it (the export's position quantization).
    pub fn geometry_position_error_m(scene: &crate::geometry_scene::GeometryScene) -> f64 {
        compact_mesh::PositionGrid::for_diagonal((scene.bounds.max - scene.bounds.min).length())
            .maximum_error()
    }

    pub fn from_geometry_scene(scene: &crate::geometry_scene::GeometryScene) -> Self {
        let mut figure = Figure::new()
            .with_grid(scene.show_grid)
            .with_legend(false)
            .with_axis_equal(scene.axis_equal);
        figure.title = scene.title.clone();
        figure.x_label = Some("X".to_string());
        figure.y_label = Some("Y".to_string());
        figure.z_label = Some("Z".to_string());
        figure.set_axes_view(0, -38.0, 24.0);
        let snapshot = FigureSnapshot::capture(&figure);
        let grid = compact_mesh::PositionGrid::for_diagonal(
            (scene.bounds.max - scene.bounds.min).length(),
        );
        let plots = scene
            .chunks
            .iter()
            .filter_map(|chunk| scene_chunk_to_mesh_plot(chunk, grid))
            .collect::<Vec<_>>();

        Self {
            schema_version: Self::SCHEMA_VERSION,
            geometry_overlay: scene.overlay.clone(),
            geometry_physical_context: scene.physical_context.clone(),
            geometry_lines: scene
                .chunks
                .iter()
                .filter_map(|chunk| GeometryLines::capture(chunk, grid))
                .collect(),
            geometry_orientation_triad: scene.show_orientation_triad,
            layout: FigureLayout {
                axes_rows: 1,
                axes_cols: 1,
                axes_indices: vec![0; plots.len()],
                axes_grid_cells: Vec::new(),
            },
            metadata: snapshot.metadata,
            plots,
        }
    }

    pub fn into_figure(self) -> Result<Figure, String> {
        self.validate_schema_version()?;

        let mut figure = Figure::new();
        let default_map = self
            .metadata
            .default_colormap
            .as_deref()
            .map(parse_colormap_name)
            .unwrap_or(ColorMap::Parula);
        figure.set_figure_colormap(default_map, self.metadata.default_colormap_length);
        figure.set_subplot_grid(
            self.layout.axes_rows as usize,
            self.layout.axes_cols as usize,
        );
        figure.active_axes_index = self.metadata.active_axes_index as usize;
        if let Some(axes_metadata) = self.metadata.axes_metadata.clone() {
            figure.axes_metadata = axes_metadata.into_iter().map(AxesMetadata::from).collect();
            figure.set_active_axes_index(figure.active_axes_index);
        } else {
            figure.set_active_axes_index(figure.active_axes_index);
            let axes = &mut figure.axes_metadata[figure.active_axes_index];
            axes.title = self.metadata.title;
            axes.x_label = self.metadata.x_label;
            axes.y_label = self.metadata.y_label;
            figure.set_axes_legend_enabled(figure.active_axes_index, self.metadata.legend_enabled);
            figure.set_axes_data_aspect_ratio(
                figure.active_axes_index,
                self.metadata.data_aspect_ratio,
                self.metadata.data_aspect_ratio_mode.clone(),
            );
            if self.metadata.axis_equal {
                figure.set_axes_axis_equal(figure.active_axes_index, true);
            }
        }
        figure.restore_axes_grid_cells(self.layout.axes_grid_cells)?;
        figure.name = self.metadata.name;
        figure.number_title = self.metadata.number_title;
        figure.visible = self.metadata.visible;
        if let Some(position) = self.metadata.position {
            validate_figure_position(position)?;
            figure.set_position(position);
        }
        figure.sg_title = self.metadata.sg_title;
        figure.sg_title_style = self
            .metadata
            .sg_title_style
            .map(TextStyle::from)
            .unwrap_or_default();
        figure.grid_enabled = self.metadata.grid_enabled;
        figure.minor_grid_enabled = self.metadata.minor_grid_enabled;
        figure.z_limits = self.metadata.z_limits.map(|[lo, hi]| (lo, hi));
        figure.colorbar_enabled = self.metadata.colorbar_enabled;
        figure.axis_equal = self.metadata.axis_equal;
        figure.background_color = rgba_to_vec4(self.metadata.background_rgba);

        for plot in self.plots {
            plot.apply_to_figure(&mut figure)?;
        }

        Ok(figure)
    }

    pub fn into_geometry_scene(
        self,
        scene_id: impl Into<String>,
        revision: u64,
    ) -> Result<crate::GeometryScene, String> {
        self.validate_schema_version()?;
        let scene_id = scene_id.into();
        let mut chunks = Vec::new();
        for (plot_index, plot) in self.plots.into_iter().enumerate() {
            append_geometry_scene_chunks(&scene_id, plot_index, plot, &mut chunks)?;
        }
        for (index, lines) in self.geometry_lines.into_iter().enumerate() {
            chunks.push(lines.restore(format!("{scene_id}:lines:{index}"))?);
        }
        if chunks.is_empty() {
            return Err("figure scene does not contain renderable mesh plots".to_string());
        }
        let mut scene = crate::GeometryScene::new(scene_id, revision, chunks);
        scene.title = self.metadata.title;
        scene.overlay = self.geometry_overlay;
        scene.physical_context = self.geometry_physical_context;
        scene.show_grid = self.metadata.grid_enabled;
        scene.show_orientation_triad = self.geometry_orientation_triad;
        scene.axis_equal = self.metadata.axis_equal;
        Ok(scene)
    }

    pub(super) fn validate_schema_version(&self) -> Result<(), String> {
        if self.schema_version == 0 || self.schema_version > FigureScene::SCHEMA_VERSION {
            return Err(format!(
                "unsupported figure scene schema version {} (supported 1..={})",
                self.schema_version,
                FigureScene::SCHEMA_VERSION
            ));
        }
        if self.schema_version < 2
            && self
                .plots
                .iter()
                .any(|plot| matches!(plot, ScenePlot::Patch { .. }))
        {
            return Err(format!(
                "patch plots require figure scene schema version {}",
                2
            ));
        }
        if self.schema_version < 3
            && self
                .plots
                .iter()
                .any(|plot| matches!(plot, ScenePlot::Mesh { .. }))
        {
            return Err(format!(
                "mesh plots require figure scene schema version {}",
                3
            ));
        }
        if self.schema_version < 5 && !self.layout.axes_grid_cells.is_empty() {
            return Err("axes grid-cell identity requires figure scene schema version 5".into());
        }
        Ok(())
    }
}

pub(super) fn validate_figure_position(position: [f64; 4]) -> Result<(), String> {
    if !position.iter().all(|value| value.is_finite()) {
        return Err("figure scene metadata Position values must be finite".to_string());
    }
    if position[2] <= 0.0 || position[3] <= 0.0 {
        return Err("figure scene metadata Position width and height must be positive".to_string());
    }
    Ok(())
}

pub(super) fn append_geometry_scene_chunks(
    scene_id: &str,
    plot_index: usize,
    plot: ScenePlot,
    chunks: &mut Vec<crate::GeometrySceneChunk>,
) -> Result<(), String> {
    let ScenePlot::Mesh {
        vertices,
        triangles,
        mesh_id,
        face_color_rgba,
        edge_color_rgba,
        face_alpha,
        edge_alpha,
        edge_width,
        edge_mode,
        feature_edge_groups,
        vertex_colors_rgba,
        triangle_colors_rgba,
        vertex_normals,
        axes_index: _,
        label,
        regions,
        highlighted_region_id,
        highlight_color_rgba,
        scalar_field,
        vector_field,
        deformation,
        instances,
        lod_pixels,
        packed,
        visible,
    } = plot
    else {
        return Ok(());
    };
    let (vertices, triangles, vertex_normals) = match packed {
        Some(packed) => packed.decode()?,
        None => (vertices, triangles, vertex_normals),
    };

    if !visible {
        return Ok(());
    }

    let region_metadata = regions
        .iter()
        .cloned()
        .map(crate::geometry_scene::GeometrySceneRegion::from)
        .collect::<Vec<_>>();
    let mesh_id_for_chunk = mesh_id
        .clone()
        .unwrap_or_else(|| format!("mesh_{}", plot_index + 1));
    let mut mesh = MeshPlot::new(vertices.into_iter().map(xyz_to_vec3).collect(), triangles)?;
    mesh.set_mesh_id(mesh_id.clone());
    mesh.set_face_color(rgba_to_vec4(face_color_rgba));
    mesh.set_edge_color(rgba_to_vec4(edge_color_rgba));
    mesh.set_face_alpha(face_alpha);
    mesh.set_edge_alpha(edge_alpha);
    mesh.set_edge_width(edge_width);
    mesh.set_edge_mode(parse_mesh_edge_mode(&edge_mode));
    if !feature_edge_groups.is_empty() {
        mesh.set_feature_edge_groups(Some(feature_edge_groups))?;
    }
    if !vertex_colors_rgba.is_empty() {
        mesh.set_vertex_colors(Some(
            vertex_colors_rgba.into_iter().map(rgba_to_vec4).collect(),
        ))?;
    }
    if !vertex_normals.is_empty() {
        mesh.set_vertex_normals(Some(
            vertex_normals.into_iter().map(Vec3::from_array).collect(),
        ))?;
    }
    if !triangle_colors_rgba.is_empty() {
        mesh.set_triangle_colors(Some(
            triangle_colors_rgba.into_iter().map(rgba_to_vec4).collect(),
        ))?;
    }
    mesh.set_label(label.clone());
    mesh.set_regions(regions.into_iter().map(Into::into).collect());
    mesh.set_highlighted_region_id(highlighted_region_id);
    if let Some(color) = highlight_color_rgba {
        mesh.set_highlight_color(rgba_to_vec4(color));
    }
    if let Some(field) = scalar_field {
        mesh.set_scalar_field(Some((*field).try_into()?))?;
    }
    if let Some(field) = vector_field {
        mesh.set_vector_field(Some((*field).try_into()?))?;
    }
    if let Some(field) = deformation {
        mesh.set_deformation(Some((*field).into()))?;
    }

    let face_render_data = mesh.render_data();
    chunks.push(
        crate::GeometrySceneChunk::from_render_data(
            format!("{scene_id}:{mesh_id_for_chunk}:faces:{plot_index}"),
            face_render_data,
        )
        .with_mesh_id(mesh_id_for_chunk.clone())
        .with_label(label.clone().unwrap_or_else(|| mesh_id_for_chunk.clone()))
        .with_regions(region_metadata)
        .with_instances(instances.clone(), lod_pixels),
    );

    if let Some(edge_render_data) = mesh.edge_render_data() {
        chunks.push(
            crate::GeometrySceneChunk::from_render_data(
                format!("{scene_id}:{mesh_id_for_chunk}:edges:{plot_index}"),
                edge_render_data,
            )
            .with_mesh_id(mesh_id_for_chunk.clone())
            .with_label(format!(
                "{} edges",
                label.clone().unwrap_or_else(|| mesh_id_for_chunk.clone())
            ))
            .with_instances(instances.clone(), lod_pixels),
        );
    }

    if let Some(vector_render_data) = mesh.vector_render_data() {
        chunks.push(
            crate::GeometrySceneChunk::from_render_data(
                format!("{scene_id}:{mesh_id_for_chunk}:vectors:{plot_index}"),
                vector_render_data,
            )
            .with_mesh_id(mesh_id_for_chunk.clone())
            .with_label(format!(
                "{} vectors",
                label.unwrap_or_else(|| mesh_id_for_chunk.clone())
            )),
        );
    }

    Ok(())
}

pub(super) fn scene_chunk_to_mesh_plot(
    chunk: &crate::geometry_scene::GeometrySceneChunk,
    grid: compact_mesh::PositionGrid,
) -> Option<ScenePlot> {
    if chunk.render_data.pipeline_type != crate::core::PipelineType::Triangles {
        return None;
    }
    let indices = chunk.indices.as_ref()?;
    if indices.len() < 3 {
        return None;
    }
    let compact_mesh::CompactSurface {
        vertices,
        triangles,
        vertex_colors,
        triangle_colors,
    } = compact_mesh::compact_surface(&chunk.vertices, indices, grid)?;
    Some(ScenePlot::Mesh {
        vertices,
        triangles,
        mesh_id: chunk.mesh_id.clone(),
        face_color_rgba: chunk.material.albedo.to_array(),
        edge_color_rgba: [0.08, 0.10, 0.13, 1.0],
        // The chunk's opacity is already in its face and vertex colours; a face alpha would
        // apply it a second time when the mesh is decoded.
        face_alpha: 1.0,
        edge_alpha: 0.0,
        edge_width: 0.0,
        edge_mode: "none".to_string(),
        feature_edge_groups: Vec::new(),
        vertex_colors_rgba: vertex_colors,
        triangle_colors_rgba: triangle_colors,
        vertex_normals: Vec::new(),
        axes_index: 0,
        label: chunk.label.clone(),
        regions: chunk.regions.iter().map(Into::into).collect(),
        highlighted_region_id: None,
        highlight_color_rgba: Some([0.98, 0.78, 0.22, 1.0]),
        scalar_field: None,
        vector_field: None,
        deformation: None,
        instances: chunk.instances.clone(),
        lod_pixels: chunk.lod_pixels,
        packed: None,
        visible: chunk.visible,
    })
}

pub(super) fn figure_axis_index(figure: &Figure, plot_index: usize) -> u32 {
    figure
        .plot_axes_indices()
        .get(plot_index)
        .copied()
        .unwrap_or(0) as u32
}
