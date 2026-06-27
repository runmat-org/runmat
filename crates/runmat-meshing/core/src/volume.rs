use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::{
    artifact::{
        AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode, AnalysisVolumeElement,
        ANALYSIS_MESH_SCHEMA_VERSION,
    },
    backend::{select_volume_backend, MeshBackendKind},
    boundary::{BoundaryMeshInput, BoundaryMeshInputError},
    options::{MeshKindRequest, MeshProfile, MeshTargetSize, VolumeMeshingOptions},
    production::generate_production_analysis_mesh,
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind},
    quality::{AnalysisMeshQualityReport, ElementQuality, QualityThresholds},
    sizing::{MeshSizingField, SizingSample, SizingSampleApplication, SizingSampleRejection},
    topology::{BoundaryElementKind, VolumeElementKind},
};

pub trait VolumeMesher {
    fn mesh(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
    ) -> Result<AnalysisMeshArtifact, MeshingError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshingError {
    BoundaryInput(String),
    UnsupportedMeshKind(MeshKindRequest),
    UnsupportedElementKind(VolumeElementKind),
    InvalidElementBudget,
    InvalidTargetSize,
    EmptyBoundaryRegions,
    ProductionBackend(String),
}

impl std::fmt::Display for MeshingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundaryInput(message) => write!(formatter, "invalid boundary mesh: {message}"),
            Self::UnsupportedMeshKind(kind) => {
                write!(formatter, "unsupported analysis mesh kind: {kind:?}")
            }
            Self::UnsupportedElementKind(kind) => {
                write!(formatter, "unsupported volume element kind: {kind:?}")
            }
            Self::InvalidElementBudget => write!(formatter, "max_elements must be greater than 0"),
            Self::InvalidTargetSize => {
                write!(
                    formatter,
                    "target_size must be auto or a finite positive length"
                )
            }
            Self::EmptyBoundaryRegions => write!(formatter, "boundary mesh has no region ids"),
            Self::ProductionBackend(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for MeshingError {}

impl From<BoundaryMeshInputError> for MeshingError {
    fn from(value: BoundaryMeshInputError) -> Self {
        Self::BoundaryInput(value.to_string())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StructuredTetMesher;

impl VolumeMesher for StructuredTetMesher {
    fn mesh(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
    ) -> Result<AnalysisMeshArtifact, MeshingError> {
        self.mesh_with_sizing(input, options, None)
    }
}

impl StructuredTetMesher {
    pub fn mesh_with_sizing(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
        sizing: Option<&MeshSizingField>,
    ) -> Result<AnalysisMeshArtifact, MeshingError> {
        if !matches!(options.kind, MeshKindRequest::Solid) {
            return Err(MeshingError::UnsupportedMeshKind(options.kind));
        }
        if !matches!(options.element, VolumeElementKind::Tet4) {
            return Err(MeshingError::UnsupportedElementKind(options.element));
        }
        if options.max_elements == 0 {
            return Err(MeshingError::InvalidElementBudget);
        }
        if input.region_ids.is_empty() {
            return Err(MeshingError::EmptyBoundaryRegions);
        }

        let mut mesh_sizing = sizing.cloned().unwrap_or_default();
        append_geometry_focus_sizing_samples(input, options, &mut mesh_sizing);

        let grid = structured_grid(input, options, Some(&mut mesh_sizing))?;
        let nodes = grid_nodes(&grid);
        let node_id_at = |i: usize, j: usize, k: usize| -> u32 {
            (1 + i + grid.x.len() * (j + grid.y.len() * k)) as u32
        };

        let material_region_id = input
            .region_ids
            .first()
            .cloned()
            .unwrap_or_else(|| "region_default".to_string());
        let provenance = vec![MeshEntityProvenance {
            source_geometry_id: input.source_geometry_id.clone(),
            source_geometry_revision: input.source_geometry_revision,
            source_entity_kind: SourceEntityKind::Mesh,
            source_entity_id: input.mesh_id.clone(),
            region_ids: input.region_ids.clone(),
        }];

        let occupied_cells = occupied_cells(input, &grid);
        let mut cell_tet_ids = vec![None; grid.cell_count()];
        let mut volume_elements = Vec::<AnalysisVolumeElement>::new();
        let mut element_quality = Vec::<ElementQuality>::new();
        for k in 0..grid.nz() {
            for j in 0..grid.ny() {
                for i in 0..grid.nx() {
                    let cell_index = grid.cell_index(i, j, k);
                    if !occupied_cells[cell_index] {
                        continue;
                    }
                    let cell_nodes = [
                        node_id_at(i, j, k),
                        node_id_at(i + 1, j, k),
                        node_id_at(i, j + 1, k),
                        node_id_at(i + 1, j + 1, k),
                        node_id_at(i, j, k + 1),
                        node_id_at(i + 1, j, k + 1),
                        node_id_at(i, j + 1, k + 1),
                        node_id_at(i + 1, j + 1, k + 1),
                    ];
                    for tet in [
                        [cell_nodes[0], cell_nodes[1], cell_nodes[3], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[3], cell_nodes[2], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[2], cell_nodes[6], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[6], cell_nodes[4], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[4], cell_nodes[5], cell_nodes[7]],
                        [cell_nodes[0], cell_nodes[5], cell_nodes[1], cell_nodes[7]],
                    ] {
                        let element_id = format!("tet_{}", volume_elements.len() + 1);
                        let oriented = orient_tet(tet, &nodes);
                        let volume_m3 = tet_volume(oriented, &nodes).abs();
                        let aspect_ratio = tet_aspect_ratio(oriented, &nodes);
                        element_quality.push(ElementQuality {
                            element_id: element_id.clone(),
                            scaled_jacobian: 1.0 / aspect_ratio.max(1.0),
                            aspect_ratio,
                            volume_m3,
                        });
                        volume_elements.push(AnalysisVolumeElement {
                            element_id: element_id.clone(),
                            kind: VolumeElementKind::Tet4,
                            node_ids: oriented.to_vec(),
                            material_region_id: material_region_id.clone(),
                            provenance: provenance.clone(),
                        });
                        cell_tet_ids[cell_index]
                            .get_or_insert_with(Vec::new)
                            .push(element_id);
                    }
                }
            }
        }

        let boundary_faces =
            grid_boundary_faces(input, &grid, &occupied_cells, &cell_tet_ids, &node_id_at);
        let original_quality = quality_report(
            element_quality,
            boundary_projection_errors(input, &boundary_faces, &nodes),
        );
        let (nodes, quality) = project_boundary_nodes_if_quality_improves(
            input,
            nodes,
            &volume_elements,
            &boundary_faces,
            original_quality,
        );
        mesh_sizing.global_target_size_m = target_size_m(input, options, &grid);

        let mesh = AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: format!("analysis_{}", input.mesh_id),
            nodes,
            volume_elements,
            boundary_faces,
            boundary_edges: Vec::new(),
            quality,
            sizing: mesh_sizing,
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "structured_bbox_tet/v1".to_string(),
                source_geometry_id: input.source_geometry_id.clone(),
                source_geometry_revision: input.source_geometry_revision,
                source_geometry_sha256: input.source_geometry_sha256.clone(),
            },
        };
        Ok(compact_analysis_mesh_nodes(mesh))
    }
}

pub fn generate_analysis_mesh(
    geometry: &runmat_geometry_core::GeometryAsset,
    options: VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, MeshingError> {
    let input = BoundaryMeshInput::from_geometry(geometry)?;
    match select_volume_backend(&options).selected {
        MeshBackendKind::StructuredTetFallback => StructuredTetMesher.mesh(&input, &options),
        MeshBackendKind::Production => generate_production_analysis_mesh(geometry, &options)
            .map_err(|err| MeshingError::ProductionBackend(err.to_string())),
        MeshBackendKind::Auto => unreachable!("backend selection must resolve auto"),
    }
}

pub fn generate_analysis_mesh_with_sizing(
    geometry: &runmat_geometry_core::GeometryAsset,
    options: VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, MeshingError> {
    let input = BoundaryMeshInput::from_geometry(geometry)?;
    match select_volume_backend(&options).selected {
        MeshBackendKind::StructuredTetFallback => {
            StructuredTetMesher.mesh_with_sizing(&input, &options, Some(sizing))
        }
        MeshBackendKind::Production => generate_production_analysis_mesh(geometry, &options)
            .map_err(|err| MeshingError::ProductionBackend(err.to_string())),
        MeshBackendKind::Auto => unreachable!("backend selection must resolve auto"),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct StructuredGrid {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

impl StructuredGrid {
    fn uniform(input: &BoundaryMeshInput, divisions: usize) -> Self {
        Self {
            x: uniform_axis(input.bounds_min_m[0], input.bounds_max_m[0], divisions),
            y: uniform_axis(input.bounds_min_m[1], input.bounds_max_m[1], divisions),
            z: uniform_axis(input.bounds_min_m[2], input.bounds_max_m[2], divisions),
        }
    }

    fn nx(&self) -> usize {
        self.x.len().saturating_sub(1)
    }

    fn ny(&self) -> usize {
        self.y.len().saturating_sub(1)
    }

    fn nz(&self) -> usize {
        self.z.len().saturating_sub(1)
    }

    fn element_count(&self) -> usize {
        6 * self.nx() * self.ny() * self.nz()
    }

    fn cell_count(&self) -> usize {
        self.nx() * self.ny() * self.nz()
    }

    fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx() * (j + self.ny() * k)
    }

    fn cell_coordinates(&self, index: usize) -> (usize, usize, usize) {
        let nx = self.nx().max(1);
        let ny = self.ny().max(1);
        let k = index / (nx * ny);
        let rem = index % (nx * ny);
        let j = rem / nx;
        let i = rem % nx;
        (i, j, k)
    }

    fn cell_neighbors(&self, i: usize, j: usize, k: usize) -> Vec<usize> {
        let mut neighbors = Vec::with_capacity(6);
        if i > 0 {
            neighbors.push(self.cell_index(i - 1, j, k));
        }
        if i + 1 < self.nx() {
            neighbors.push(self.cell_index(i + 1, j, k));
        }
        if j > 0 {
            neighbors.push(self.cell_index(i, j - 1, k));
        }
        if j + 1 < self.ny() {
            neighbors.push(self.cell_index(i, j + 1, k));
        }
        if k > 0 {
            neighbors.push(self.cell_index(i, j, k - 1));
        }
        if k + 1 < self.nz() {
            neighbors.push(self.cell_index(i, j, k + 1));
        }
        neighbors
    }

    fn min_cell_size(&self) -> Option<f64> {
        [
            min_axis_spacing(&self.x),
            min_axis_spacing(&self.y),
            min_axis_spacing(&self.z),
        ]
        .into_iter()
        .flatten()
        .reduce(f64::min)
    }

    fn max_cell_aspect_ratio(&self) -> Option<f64> {
        let mut max_ratio = 0.0_f64;
        let mut saw_cell = false;
        for dx in axis_spacings(&self.x) {
            for dy in axis_spacings(&self.y) {
                for dz in axis_spacings(&self.z) {
                    let min_edge = dx.min(dy).min(dz);
                    if !min_edge.is_finite() || min_edge <= 0.0 {
                        continue;
                    }
                    let diagonal = (dx * dx + dy * dy + dz * dz).sqrt();
                    max_ratio = max_ratio.max(diagonal / min_edge);
                    saw_cell = true;
                }
            }
        }
        saw_cell.then_some(max_ratio)
    }
}

fn structured_grid(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    sizing: Option<&mut MeshSizingField>,
) -> Result<StructuredGrid, MeshingError> {
    let max_by_budget = ((options.max_elements / 6).max(1) as f64)
        .cbrt()
        .floor()
        .max(1.0) as usize;
    let requested = match options.target_size {
        MeshTargetSize::Auto => match options.profile {
            MeshProfile::Coarse => 1,
            MeshProfile::AnalysisReady => 2,
            MeshProfile::Adaptive | MeshProfile::Fine => 3,
        },
        MeshTargetSize::LengthM(length_m) => {
            if !length_m.is_finite() || length_m <= 0.0 {
                return Err(MeshingError::InvalidTargetSize);
            }
            let max_span = (0..3)
                .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            (max_span / length_m).ceil().max(1.0) as usize
        }
    };
    let requested = sizing
        .as_deref()
        .and_then(global_sizing_target_size_m)
        .map(|length_m| requested.max(divisions_for_target_size(input, length_m)))
        .unwrap_or(requested);
    let divisions = requested.clamp(1, max_by_budget);
    let mut grid = StructuredGrid::uniform(input, divisions);
    if let Some(sizing) = sizing {
        insert_local_sizing_breakpoints(input, options.max_elements, sizing, &mut grid);
    }
    Ok(grid)
}

fn divisions_for_target_size(input: &BoundaryMeshInput, length_m: f64) -> usize {
    let max_span = (0..3)
        .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    (max_span / length_m).ceil().max(1.0) as usize
}

fn global_sizing_target_size_m(sizing: &MeshSizingField) -> Option<f64> {
    [sizing.min_size_m, sizing.global_target_size_m]
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite() && *value > 0.0)
        .reduce(f64::min)
}

fn insert_local_sizing_breakpoints(
    input: &BoundaryMeshInput,
    max_elements: usize,
    sizing: &mut MeshSizingField,
    grid: &mut StructuredGrid,
) {
    let mut samples = Vec::new();
    for sample in sizing.samples.clone() {
        let Some(target_size_m) = clamped_sample_target_size(sample.target_size_m, sizing) else {
            sizing.rejected_samples.push(sizing_rejection(
                sample.position_m,
                sample.target_size_m,
                sample.reason,
                "skipped_invalid",
                "sizing sample target size was not finite and positive after bounds were applied",
            ));
            continue;
        };
        if !sample.position_m.iter().all(|value| value.is_finite()) {
            sizing.rejected_samples.push(sizing_rejection(
                sample.position_m,
                target_size_m,
                sample.reason,
                "skipped_invalid",
                "sizing sample position contained a non-finite coordinate",
            ));
            continue;
        }
        samples.push((sample.position_m, target_size_m, sample.reason));
    }
    samples.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0[0].total_cmp(&right.0[0]))
            .then_with(|| left.0[1].total_cmp(&right.0[1]))
            .then_with(|| left.0[2].total_cmp(&right.0[2]))
    });

    for (position_m, target_size_m, reason) in samples {
        let mut inserted_breakpoint_count = 0_usize;
        let mut duplicate_or_boundary_count = 0_usize;
        for axis in 0..3 {
            for coordinate in
                local_breakpoint_candidates(input, axis, position_m[axis], target_size_m)
            {
                let mut candidate = grid.clone();
                if !candidate.insert_axis_coordinate(axis, coordinate) {
                    duplicate_or_boundary_count += 1;
                    continue;
                }
                if candidate.element_count() > max_elements {
                    sizing.rejected_samples.push(sizing_rejection(
                        position_m,
                        target_size_m,
                        reason.clone(),
                        "skipped_budget",
                        "element budget prevented local sizing breakpoint",
                    ));
                } else if !candidate.satisfies_quality_guard() {
                    sizing.rejected_samples.push(sizing_rejection(
                        position_m,
                        target_size_m,
                        reason.clone(),
                        "skipped_quality",
                        "mesh quality guard prevented local sizing breakpoint",
                    ));
                } else {
                    *grid = candidate;
                    inserted_breakpoint_count += 1;
                }
            }
        }
        if inserted_breakpoint_count > 0 {
            sizing.applied_samples.push(sizing_application(
                position_m,
                target_size_m,
                inserted_breakpoint_count,
                reason.clone(),
                duplicate_or_boundary_count,
            ));
        } else if duplicate_or_boundary_count > 0 {
            sizing.rejected_samples.push(sizing_rejection(
                position_m,
                target_size_m,
                reason,
                "skipped_duplicate",
                "sizing sample only produced duplicate or boundary-clamped breakpoints",
            ));
        }
    }
}

fn sizing_application(
    position_m: [f64; 3],
    target_size_m: f64,
    inserted_breakpoint_count: usize,
    reason: Option<String>,
    duplicate_or_boundary_count: usize,
) -> SizingSampleApplication {
    let detail = if duplicate_or_boundary_count > 0 {
        Some(format!(
            "inserted {inserted_breakpoint_count} local sizing breakpoints; skipped {duplicate_or_boundary_count} duplicate or boundary-clamped candidates"
        ))
    } else {
        Some(format!(
            "inserted {inserted_breakpoint_count} local sizing breakpoints"
        ))
    };
    SizingSampleApplication {
        position_m,
        target_size_m,
        inserted_breakpoint_count,
        reason,
        detail,
    }
}

fn sizing_rejection(
    position_m: [f64; 3],
    target_size_m: f64,
    reason: Option<String>,
    status: &str,
    detail: &str,
) -> SizingSampleRejection {
    SizingSampleRejection {
        position_m,
        target_size_m,
        status: status.to_string(),
        reason,
        detail: Some(detail.to_string()),
    }
}

fn clamped_sample_target_size(target_size_m: f64, sizing: &MeshSizingField) -> Option<f64> {
    if !target_size_m.is_finite() || target_size_m <= 0.0 {
        return None;
    }
    let mut target_size_m = target_size_m;
    if let Some(min_size_m) = sizing
        .min_size_m
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        target_size_m = target_size_m.max(min_size_m);
    }
    if let Some(max_size_m) = sizing
        .max_size_m
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        target_size_m = target_size_m.min(max_size_m);
    }
    (target_size_m.is_finite() && target_size_m > 0.0).then_some(target_size_m)
}

fn local_breakpoint_candidates(
    input: &BoundaryMeshInput,
    axis: usize,
    coordinate: f64,
    target_size_m: f64,
) -> [f64; 3] {
    [
        coordinate,
        coordinate - target_size_m,
        coordinate + target_size_m,
    ]
    .map(|value| value.clamp(input.bounds_min_m[axis], input.bounds_max_m[axis]))
}

fn append_geometry_focus_sizing_samples(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    sizing: &mut MeshSizingField,
) {
    if options.refinement.focus.curvature {
        sizing.samples.extend(curvature_sizing_samples(input));
    }
    if options.refinement.focus.small_features {
        sizing.samples.extend(small_feature_sizing_samples(input));
    }
}

fn curvature_sizing_samples(input: &BoundaryMeshInput) -> Vec<SizingSample> {
    let mut triangles_by_edge = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (triangle_index, triangle) in input.triangles.iter().enumerate() {
        for edge in triangle_edges(triangle.node_ids) {
            triangles_by_edge
                .entry(edge)
                .or_default()
                .push(triangle_index);
        }
    }

    triangles_by_edge
        .into_iter()
        .filter_map(|(edge, triangle_indices)| {
            if triangle_indices.len() != 2 {
                return None;
            }
            let left = input.triangles.get(triangle_indices[0])?;
            let right = input.triangles.get(triangle_indices[1])?;
            let left_normal = triangle_unit_normal(input, left.node_ids)?;
            let right_normal = triangle_unit_normal(input, right.node_ids)?;
            let normal_dot = dot(left_normal, right_normal).clamp(-1.0, 1.0);
            if 1.0 - normal_dot.abs() <= 0.05 {
                return None;
            }
            let left_vertex = *input.vertices.get(edge[0] as usize)?;
            let right_vertex = *input.vertices.get(edge[1] as usize)?;
            let edge_length = distance(left_vertex, right_vertex);
            (edge_length.is_finite() && edge_length > 0.0).then_some(SizingSample {
                position_m: midpoint(left_vertex, right_vertex),
                target_size_m: edge_length * 0.5,
                reason: Some("geometry.curvature".to_string()),
            })
        })
        .collect()
}

fn small_feature_sizing_samples(input: &BoundaryMeshInput) -> Vec<SizingSample> {
    let max_span = boundary_max_span(input);
    if !max_span.is_finite() || max_span <= 0.0 {
        return Vec::new();
    }
    let threshold = max_span * 0.35;
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            let min_edge = triangle_min_edge(vertices);
            if !min_edge.is_finite() || min_edge <= 0.0 || min_edge > threshold {
                return None;
            }
            Some(SizingSample {
                position_m: triangle_centroid(vertices),
                target_size_m: min_edge * 0.5,
                reason: Some("geometry.small_features".to_string()),
            })
        })
        .collect()
}

impl StructuredGrid {
    fn insert_axis_coordinate(&mut self, axis: usize, coordinate: f64) -> bool {
        let coordinates = match axis {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        };
        if !coordinate.is_finite() {
            return false;
        }
        let span = coordinates
            .last()
            .zip(coordinates.first())
            .map(|(max, min)| max - min)
            .unwrap_or(0.0)
            .abs();
        let tolerance = span.max(1.0) * 1.0e-10;
        if coordinates
            .iter()
            .any(|existing| (*existing - coordinate).abs() <= tolerance)
        {
            return false;
        }
        coordinates.push(coordinate);
        coordinates.sort_by(f64::total_cmp);
        true
    }

    fn satisfies_quality_guard(&self) -> bool {
        let thresholds = QualityThresholds::default();
        let aspect_limit = thresholds
            .max_aspect_ratio
            .min(1.0 / thresholds.min_scaled_jacobian.max(f64::EPSILON));
        self.max_cell_aspect_ratio()
            .is_some_and(|ratio| ratio.is_finite() && ratio <= aspect_limit)
    }
}

fn uniform_axis(min: f64, max: f64, divisions: usize) -> Vec<f64> {
    (0..=divisions)
        .map(|index| lerp(min, max, index as f64 / divisions as f64))
        .collect()
}

fn min_axis_spacing(axis: &[f64]) -> Option<f64> {
    axis_spacings(axis)
        .filter(|value| value.is_finite() && *value > 0.0)
        .reduce(f64::min)
}

fn axis_spacings(axis: &[f64]) -> impl Iterator<Item = f64> + '_ {
    axis.windows(2).map(|pair| pair[1] - pair[0])
}

fn grid_nodes(grid: &StructuredGrid) -> Vec<AnalysisMeshNode> {
    let mut nodes = Vec::with_capacity(grid.x.len() * grid.y.len() * grid.z.len());
    for z in &grid.z {
        for y in &grid.y {
            for x in &grid.x {
                nodes.push(AnalysisMeshNode {
                    node_id: nodes.len() as u32 + 1,
                    coordinates_m: [*x, *y, *z],
                    provenance: Vec::new(),
                });
            }
        }
    }
    nodes
}

fn compact_analysis_mesh_nodes(mut mesh: AnalysisMeshArtifact) -> AnalysisMeshArtifact {
    let mut referenced_node_ids = BTreeMap::<u32, u32>::new();
    for element in &mesh.volume_elements {
        for node_id in &element.node_ids {
            referenced_node_ids.entry(*node_id).or_default();
        }
    }
    for face in &mesh.boundary_faces {
        for node_id in &face.node_ids {
            referenced_node_ids.entry(*node_id).or_default();
        }
    }

    if referenced_node_ids.len() == mesh.nodes.len() {
        return mesh;
    }

    let nodes_by_id = mesh
        .nodes
        .into_iter()
        .map(|node| (node.node_id, node))
        .collect::<BTreeMap<_, _>>();
    let mut compact_nodes = Vec::with_capacity(referenced_node_ids.len());
    for (new_index, (old_node_id, new_node_id)) in referenced_node_ids.iter_mut().enumerate() {
        *new_node_id = new_index as u32 + 1;
        if let Some(mut node) = nodes_by_id.get(old_node_id).cloned() {
            node.node_id = *new_node_id;
            compact_nodes.push(node);
        }
    }

    for element in &mut mesh.volume_elements {
        remap_node_ids(&mut element.node_ids, &referenced_node_ids);
    }
    for face in &mut mesh.boundary_faces {
        remap_node_ids(&mut face.node_ids, &referenced_node_ids);
    }
    mesh.nodes = compact_nodes;
    mesh
}

fn remap_node_ids(node_ids: &mut [u32], node_id_map: &BTreeMap<u32, u32>) {
    for node_id in node_ids {
        if let Some(new_node_id) = node_id_map.get(node_id) {
            *node_id = *new_node_id;
        }
    }
}

fn grid_boundary_faces(
    input: &BoundaryMeshInput,
    grid: &StructuredGrid,
    occupied_cells: &[bool],
    cell_tet_ids: &[Option<Vec<String>>],
    node_id_at: &impl Fn(usize, usize, usize) -> u32,
) -> Vec<AnalysisBoundaryFace> {
    let mut faces = Vec::new();
    for k in 0..grid.nz() {
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let cell_index = grid.cell_index(i, j, k);
                if !occupied_cells[cell_index] {
                    continue;
                }
                let adjacent_volume_element_ids =
                    cell_tet_ids[cell_index].clone().unwrap_or_default();
                for side in BoundarySide::ALL {
                    if neighbor_is_occupied(grid, occupied_cells, i, j, k, side) {
                        continue;
                    }
                    let quad = boundary_side_quad(side, i, j, k, node_id_at);
                    let centroid = quad_centroid(quad, grid, node_id_at);
                    let region_ids = nearest_boundary_triangle_regions(input, centroid)
                        .unwrap_or_else(|| input.region_ids.clone());
                    for tri in [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]] {
                        faces.push(AnalysisBoundaryFace {
                            face_id: format!("bf_{}", faces.len() + 1),
                            kind: BoundaryElementKind::Tri3,
                            node_ids: tri.to_vec(),
                            adjacent_volume_element_ids: adjacent_volume_element_ids.clone(),
                            region_ids: region_ids.clone(),
                            provenance: vec![MeshEntityProvenance {
                                source_geometry_id: input.source_geometry_id.clone(),
                                source_geometry_revision: input.source_geometry_revision,
                                source_entity_kind: SourceEntityKind::Region,
                                source_entity_id: region_ids.join(","),
                                region_ids: region_ids.clone(),
                            }],
                        });
                    }
                }
            }
        }
    }
    faces
}

fn neighbor_is_occupied(
    grid: &StructuredGrid,
    occupied_cells: &[bool],
    i: usize,
    j: usize,
    k: usize,
    side: BoundarySide,
) -> bool {
    let neighbor = match side {
        BoundarySide::XMin => i.checked_sub(1).map(|i| (i, j, k)),
        BoundarySide::XMax => (i + 1 < grid.nx()).then_some((i + 1, j, k)),
        BoundarySide::YMin => j.checked_sub(1).map(|j| (i, j, k)),
        BoundarySide::YMax => (j + 1 < grid.ny()).then_some((i, j + 1, k)),
        BoundarySide::ZMin => k.checked_sub(1).map(|k| (i, j, k)),
        BoundarySide::ZMax => (k + 1 < grid.nz()).then_some((i, j, k + 1)),
    };
    neighbor
        .map(|(i, j, k)| occupied_cells[grid.cell_index(i, j, k)])
        .unwrap_or(false)
}

fn boundary_side_quad(
    side: BoundarySide,
    i: usize,
    j: usize,
    k: usize,
    node_id_at: &impl Fn(usize, usize, usize) -> u32,
) -> [u32; 4] {
    match side {
        BoundarySide::XMin => [
            node_id_at(i, j, k),
            node_id_at(i, j + 1, k),
            node_id_at(i, j + 1, k + 1),
            node_id_at(i, j, k + 1),
        ],
        BoundarySide::XMax => [
            node_id_at(i + 1, j, k),
            node_id_at(i + 1, j, k + 1),
            node_id_at(i + 1, j + 1, k + 1),
            node_id_at(i + 1, j + 1, k),
        ],
        BoundarySide::YMin => [
            node_id_at(i, j, k),
            node_id_at(i, j, k + 1),
            node_id_at(i + 1, j, k + 1),
            node_id_at(i + 1, j, k),
        ],
        BoundarySide::YMax => [
            node_id_at(i, j + 1, k),
            node_id_at(i + 1, j + 1, k),
            node_id_at(i + 1, j + 1, k + 1),
            node_id_at(i, j + 1, k + 1),
        ],
        BoundarySide::ZMin => [
            node_id_at(i, j, k),
            node_id_at(i + 1, j, k),
            node_id_at(i + 1, j + 1, k),
            node_id_at(i, j + 1, k),
        ],
        BoundarySide::ZMax => [
            node_id_at(i, j, k + 1),
            node_id_at(i, j + 1, k + 1),
            node_id_at(i + 1, j + 1, k + 1),
            node_id_at(i + 1, j, k + 1),
        ],
    }
}

fn quad_centroid(
    quad: [u32; 4],
    grid: &StructuredGrid,
    node_id_at: &impl Fn(usize, usize, usize) -> u32,
) -> [f64; 3] {
    let mut centroid = [0.0; 3];
    for node_id in quad {
        let index = node_id - 1;
        let x_index = index as usize % grid.x.len();
        let yz = index as usize / grid.x.len();
        let y_index = yz % grid.y.len();
        let z_index = yz / grid.y.len();
        debug_assert_eq!(
            node_id,
            node_id_at(x_index, y_index, z_index),
            "structured node id mapping changed"
        );
        centroid[0] += grid.x[x_index] * 0.25;
        centroid[1] += grid.y[y_index] * 0.25;
        centroid[2] += grid.z[z_index] * 0.25;
    }
    centroid
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BoundarySide {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

impl BoundarySide {
    const ALL: [Self; 6] = [
        Self::XMin,
        Self::XMax,
        Self::YMin,
        Self::YMax,
        Self::ZMin,
        Self::ZMax,
    ];
}

fn occupied_cells(input: &BoundaryMeshInput, grid: &StructuredGrid) -> Vec<bool> {
    let mut occupied = vec![false; grid.cell_count()];
    let boundary_cells = boundary_triangle_centroid_cells(input, grid);
    for k in 0..grid.nz() {
        for j in 0..grid.ny() {
            for i in 0..grid.nx() {
                let cell_index = grid.cell_index(i, j, k);
                if boundary_cells[cell_index] {
                    occupied[cell_index] = true;
                    continue;
                }
                let center = [
                    (grid.x[i] + grid.x[i + 1]) * 0.5,
                    (grid.y[j] + grid.y[j + 1]) * 0.5,
                    (grid.z[k] + grid.z[k + 1]) * 0.5,
                ];
                occupied[cell_index] = point_inside_closed_surface(input, center)
                    || cell_corners(i, j, k, grid)
                        .into_iter()
                        .any(|corner| point_inside_closed_surface(input, corner));
            }
        }
    }
    if occupied.iter().any(|cell| *cell) {
        largest_connected_occupied_component(grid, occupied)
    } else {
        vec![true; grid.cell_count()]
    }
}

fn boundary_triangle_centroid_cells(input: &BoundaryMeshInput, grid: &StructuredGrid) -> Vec<bool> {
    let mut cells = vec![false; grid.cell_count()];
    if grid.cell_count() == 0 {
        return cells;
    }
    for triangle in &input.triangles {
        let Some(vertices) = triangle_vertices(input, triangle.node_ids) else {
            continue;
        };
        let centroid = triangle_centroid(vertices);
        let Some(i) = axis_cell_index(&grid.x, centroid[0]) else {
            continue;
        };
        let Some(j) = axis_cell_index(&grid.y, centroid[1]) else {
            continue;
        };
        let Some(k) = axis_cell_index(&grid.z, centroid[2]) else {
            continue;
        };
        cells[grid.cell_index(i, j, k)] = true;
    }
    cells
}

fn axis_cell_index(axis: &[f64], value: f64) -> Option<usize> {
    if axis.len() < 2 || !value.is_finite() {
        return None;
    }
    let first = *axis.first()?;
    let last = *axis.last()?;
    if value < first || value > last {
        return None;
    }
    if value == last {
        return Some(axis.len() - 2);
    }
    let upper = axis.partition_point(|breakpoint| *breakpoint <= value);
    upper.checked_sub(1).filter(|index| *index + 1 < axis.len())
}

fn cell_corners(i: usize, j: usize, k: usize, grid: &StructuredGrid) -> [[f64; 3]; 8] {
    [
        [grid.x[i], grid.y[j], grid.z[k]],
        [grid.x[i + 1], grid.y[j], grid.z[k]],
        [grid.x[i], grid.y[j + 1], grid.z[k]],
        [grid.x[i + 1], grid.y[j + 1], grid.z[k]],
        [grid.x[i], grid.y[j], grid.z[k + 1]],
        [grid.x[i + 1], grid.y[j], grid.z[k + 1]],
        [grid.x[i], grid.y[j + 1], grid.z[k + 1]],
        [grid.x[i + 1], grid.y[j + 1], grid.z[k + 1]],
    ]
}

fn largest_connected_occupied_component(
    grid: &StructuredGrid,
    occupied_cells: Vec<bool>,
) -> Vec<bool> {
    let mut visited = vec![false; occupied_cells.len()];
    let mut largest_component = Vec::<usize>::new();
    for cell_index in 0..occupied_cells.len() {
        if !occupied_cells[cell_index] || visited[cell_index] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([cell_index]);
        visited[cell_index] = true;
        while let Some(current) = queue.pop_front() {
            component.push(current);
            let (i, j, k) = grid.cell_coordinates(current);
            for neighbor in grid.cell_neighbors(i, j, k) {
                if occupied_cells[neighbor] && !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        if component.len() > largest_component.len() {
            largest_component = component;
        }
    }
    if largest_component.is_empty() {
        return occupied_cells;
    }

    let mut retained = vec![false; occupied_cells.len()];
    for cell_index in largest_component {
        retained[cell_index] = true;
    }
    retained
}

fn point_inside_closed_surface(input: &BoundaryMeshInput, point: [f64; 3]) -> bool {
    let epsilon = boundary_max_span(input).max(1.0) * 1.0e-10;
    let probes = [
        ([1.0, 0.0, 0.0], [-0.37, 0.19, 0.11]),
        ([0.0, 1.0, 0.0], [0.13, -0.41, 0.23]),
        ([0.0, 0.0, 1.0], [0.17, 0.29, -0.43]),
    ];
    probes
        .into_iter()
        .filter(|(direction, jitter)| {
            ray_odd_intersection_count(input, point, *direction, *jitter, epsilon)
        })
        .count()
        >= 2
}

fn ray_odd_intersection_count(
    input: &BoundaryMeshInput,
    point: [f64; 3],
    direction: [f64; 3],
    jitter: [f64; 3],
    epsilon: f64,
) -> bool {
    let origin = [
        point[0] + epsilon * jitter[0],
        point[1] + epsilon * jitter[1],
        point[2] + epsilon * jitter[2],
    ];
    let mut intersections = Vec::<f64>::new();
    for triangle in &input.triangles {
        let Some(vertices) = triangle_vertices(input, triangle.node_ids) else {
            continue;
        };
        let Some(distance) = ray_triangle_intersection(origin, direction, vertices, epsilon) else {
            continue;
        };
        if distance > epsilon {
            intersections.push(distance);
        }
    }
    intersections.sort_by(f64::total_cmp);
    intersections.dedup_by(|left, right| (*left - *right).abs() <= epsilon);
    intersections.len() % 2 == 1
}

fn ray_triangle_intersection(
    origin: [f64; 3],
    direction: [f64; 3],
    vertices: [[f64; 3]; 3],
    epsilon: f64,
) -> Option<f64> {
    let edge1 = sub(vertices[1], vertices[0]);
    let edge2 = sub(vertices[2], vertices[0]);
    let h = cross(direction, edge2);
    let determinant = dot(edge1, h);
    if determinant.abs() <= epsilon {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let s = sub(origin, vertices[0]);
    let u = inverse_determinant * dot(s, h);
    if u < -epsilon || u > 1.0 + epsilon {
        return None;
    }
    let q = cross(s, edge1);
    let v = inverse_determinant * dot(direction, q);
    if v < -epsilon || u + v > 1.0 + epsilon {
        return None;
    }
    let distance = inverse_determinant * dot(edge2, q);
    distance.is_finite().then_some(distance)
}

fn nearest_boundary_triangle_regions(
    input: &BoundaryMeshInput,
    point: [f64; 3],
) -> Option<Vec<String>> {
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            let centroid = triangle_centroid(vertices);
            Some((distance(point, centroid), triangle.region_ids.clone()))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, mut region_ids)| {
            region_ids.sort();
            region_ids.dedup();
            region_ids
        })
        .filter(|region_ids| !region_ids.is_empty())
}

fn target_size_m(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    grid: &StructuredGrid,
) -> Option<f64> {
    match options.target_size {
        MeshTargetSize::LengthM(value) => Some(value),
        MeshTargetSize::Auto => grid.min_cell_size().or_else(|| {
            let max_span = (0..3)
                .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            Some(max_span)
        }),
    }
}

fn quality_report(
    elements: Vec<ElementQuality>,
    boundary_projection_errors_m: Vec<f64>,
) -> AnalysisMeshQualityReport {
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let max_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let mean_aspect_ratio = if elements.is_empty() {
        0.0
    } else {
        elements
            .iter()
            .map(|element| element.aspect_ratio)
            .sum::<f64>()
            / elements.len() as f64
    };
    let mean_boundary_projection_error_m = if boundary_projection_errors_m.is_empty() {
        0.0
    } else {
        boundary_projection_errors_m.iter().sum::<f64>() / boundary_projection_errors_m.len() as f64
    };
    let max_boundary_projection_error_m = boundary_projection_errors_m
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    AnalysisMeshQualityReport {
        min_scaled_jacobian,
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: 0,
        mean_boundary_projection_error_m,
        max_boundary_projection_error_m,
        elements,
    }
}

fn boundary_projection_errors(
    input: &BoundaryMeshInput,
    boundary_faces: &[AnalysisBoundaryFace],
    nodes: &[AnalysisMeshNode],
) -> Vec<f64> {
    boundary_faces
        .iter()
        .filter_map(|face| {
            let centroid = element_centroid(nodes, &face.node_ids)?;
            nearest_boundary_triangle_distance(input, centroid)
        })
        .filter(|distance_m| distance_m.is_finite())
        .collect()
}

fn project_boundary_nodes_if_quality_improves(
    input: &BoundaryMeshInput,
    nodes: Vec<AnalysisMeshNode>,
    volume_elements: &[AnalysisVolumeElement],
    boundary_faces: &[AnalysisBoundaryFace],
    original_quality: AnalysisMeshQualityReport,
) -> (Vec<AnalysisMeshNode>, AnalysisMeshQualityReport) {
    let mut projection_targets = Vec::<(u32, [f64; 3], [f64; 3])>::new();
    for node_id in boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let Some(node) = nodes.get(node_id.saturating_sub(1) as usize) else {
            continue;
        };
        let Some(projected) = nearest_boundary_triangle_point(input, node.coordinates_m) else {
            continue;
        };
        if distance(node.coordinates_m, projected) > 0.0 {
            projection_targets.push((node_id, node.coordinates_m, projected));
        }
    }
    if projection_targets.is_empty() {
        return (nodes, original_quality);
    }

    let thresholds = QualityThresholds::default();
    let mut best_nodes = nodes.clone();
    let mut best_quality = original_quality.clone();
    for relaxation in [1.0_f64, 0.75, 0.5, 0.25, 0.125] {
        let mut candidate_nodes = nodes.clone();
        for (node_id, original, projected) in &projection_targets {
            let Some(node) = candidate_nodes.get_mut(node_id.saturating_sub(1) as usize) else {
                continue;
            };
            node.coordinates_m = [
                original[0] + relaxation * (projected[0] - original[0]),
                original[1] + relaxation * (projected[1] - original[1]),
                original[2] + relaxation * (projected[2] - original[2]),
            ];
        }

        let Some(candidate_element_quality) =
            element_quality_for_nodes(volume_elements, &candidate_nodes)
        else {
            continue;
        };
        let candidate_quality = quality_report(
            candidate_element_quality,
            boundary_projection_errors(input, boundary_faces, &candidate_nodes),
        );
        let improved_projection = candidate_quality.max_boundary_projection_error_m
            < best_quality.max_boundary_projection_error_m;
        let quality_ok = candidate_quality.min_scaled_jacobian.is_finite()
            && candidate_quality.min_scaled_jacobian >= thresholds.min_scaled_jacobian
            && candidate_quality.max_aspect_ratio.is_finite()
            && candidate_quality.max_aspect_ratio <= thresholds.max_aspect_ratio
            && candidate_quality.inverted_element_count == 0;
        if improved_projection && quality_ok {
            best_nodes = candidate_nodes;
            best_quality = candidate_quality;
        }
    }
    (best_nodes, best_quality)
}

fn element_quality_for_nodes(
    volume_elements: &[AnalysisVolumeElement],
    nodes: &[AnalysisMeshNode],
) -> Option<Vec<ElementQuality>> {
    let mut qualities = Vec::with_capacity(volume_elements.len());
    for element in volume_elements {
        let node_ids: [u32; 4] = element.node_ids.as_slice().try_into().ok()?;
        let volume_m3 = tet_volume(node_ids, nodes);
        if !volume_m3.is_finite() || volume_m3 <= 0.0 {
            return None;
        }
        let aspect_ratio = tet_aspect_ratio(node_ids, nodes);
        qualities.push(ElementQuality {
            element_id: element.element_id.clone(),
            scaled_jacobian: 1.0 / aspect_ratio.max(1.0),
            aspect_ratio,
            volume_m3,
        });
    }
    Some(qualities)
}

fn nearest_boundary_triangle_distance(input: &BoundaryMeshInput, point: [f64; 3]) -> Option<f64> {
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            Some(point_triangle_distance(point, vertices))
        })
        .filter(|distance_m| distance_m.is_finite())
        .min_by(f64::total_cmp)
}

fn nearest_boundary_triangle_point(input: &BoundaryMeshInput, point: [f64; 3]) -> Option<[f64; 3]> {
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            let closest = closest_point_on_triangle(point, vertices);
            Some((distance(point, closest), closest))
        })
        .filter(|(distance_m, _)| distance_m.is_finite())
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, closest)| closest)
}

fn point_triangle_distance(point: [f64; 3], vertices: [[f64; 3]; 3]) -> f64 {
    distance(point, closest_point_on_triangle(point, vertices))
}

fn closest_point_on_triangle(point: [f64; 3], vertices: [[f64; 3]; 3]) -> [f64; 3] {
    let [a, b, c] = vertices;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(point, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = sub(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }

    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let bc = sub(c, b);
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(bc, w));
    }

    let normal = cross(ab, ac);
    let normal_dot = dot(normal, normal);
    if normal_dot <= f64::EPSILON {
        return [a, b, c]
            .into_iter()
            .min_by(|left, right| distance(point, *left).total_cmp(&distance(point, *right)))
            .unwrap_or(a);
    }
    let signed_distance_scale = dot(ap, normal) / normal_dot;
    sub(point, scale(normal, signed_distance_scale))
}

fn orient_tet(mut node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> [u32; 4] {
    if tet_volume(node_ids, nodes) < 0.0 {
        node_ids.swap(0, 1);
    }
    node_ids
}

fn tet_volume(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> f64 {
    let Some([a, b, c, d]) = tet_points(node_ids, nodes) else {
        return 0.0;
    };
    dot(sub(b, a), cross(sub(c, a), sub(d, a))) / 6.0
}

fn tet_aspect_ratio(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> f64 {
    let Some(points) = tet_points(node_ids, nodes) else {
        return f64::INFINITY;
    };
    let mut min_edge = f64::INFINITY;
    let mut max_edge = 0.0_f64;
    for (left, right) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let length = norm(sub(points[left], points[right]));
        min_edge = min_edge.min(length);
        max_edge = max_edge.max(length);
    }
    max_edge / min_edge.max(f64::EPSILON)
}

fn tet_points(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> Option<[[f64; 3]; 4]> {
    Some([
        nodes
            .get(node_ids[0].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[1].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[2].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[3].checked_sub(1)? as usize)?
            .coordinates_m,
    ])
}

fn element_centroid(nodes: &[AnalysisMeshNode], node_ids: &[u32]) -> Option<[f64; 3]> {
    if node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0; 3];
    for node_id in node_ids {
        let coordinates = nodes.get(node_id.checked_sub(1)? as usize)?.coordinates_m;
        centroid[0] += coordinates[0];
        centroid[1] += coordinates[1];
        centroid[2] += coordinates[2];
    }
    let scale = 1.0 / node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}

fn lerp(left: f64, right: f64, t: f64) -> f64 {
    left + (right - left) * t
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    norm(sub(left, right))
}

fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        (left[0] + right[0]) * 0.5,
        (left[1] + right[1]) * 0.5,
        (left[2] + right[2]) * 0.5,
    ]
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

fn triangle_vertices(input: &BoundaryMeshInput, node_ids: [u32; 3]) -> Option<[[f64; 3]; 3]> {
    Some([
        *input.vertices.get(node_ids[0] as usize)?,
        *input.vertices.get(node_ids[1] as usize)?,
        *input.vertices.get(node_ids[2] as usize)?,
    ])
}

fn triangle_unit_normal(input: &BoundaryMeshInput, node_ids: [u32; 3]) -> Option<[f64; 3]> {
    let [a, b, c] = triangle_vertices(input, node_ids)?;
    let normal = cross(sub(b, a), sub(c, a));
    let length = norm(normal);
    (length > 0.0).then_some([normal[0] / length, normal[1] / length, normal[2] / length])
}

fn triangle_min_edge(vertices: [[f64; 3]; 3]) -> f64 {
    distance(vertices[0], vertices[1])
        .min(distance(vertices[1], vertices[2]))
        .min(distance(vertices[2], vertices[0]))
}

fn triangle_centroid(vertices: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3.0,
        (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3.0,
        (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3.0,
    ]
}

fn boundary_max_span(input: &BoundaryMeshInput) -> f64 {
    (0..3)
        .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        validate_analysis_mesh, BoundaryMeshTriangle, MeshBackendKind, MeshKindRequest,
        SizingSample,
    };
    use runmat_geometry_core::{
        GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping,
        SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
    };

    fn cube_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_tet_cube".to_string(),
            source: GeometrySource {
                path: "/fixtures/generic_cube.step".to_string(),
                sha256: "generic-cube".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 3,
            meshes: vec![MeshDescriptor {
                mesh_id: "cube_surface".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 8,
                element_count: 12,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "cube_surface",
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                vec![
                    [0, 2, 1],
                    [0, 3, 2],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ],
            )],
            regions: vec![
                Region {
                    region_id: "region_fixed".to_string(),
                    name: "fixed".to_string(),
                    tag: Some("fixed".to_string()),
                    cad_ownership: None,
                },
                Region {
                    region_id: "region_load".to_string(),
                    name: "load".to_string(),
                    tag: Some("load".to_string()),
                    cad_ownership: None,
                },
            ],
            region_entity_mappings: vec![
                RegionEntityMapping::all_faces("region_fixed", "cube_surface", 2),
                RegionEntityMapping::new(
                    "region_load",
                    "cube_surface",
                    runmat_geometry_core::EntityKind::Face,
                    vec![runmat_geometry_core::EntityIdRange::new(2, 2)],
                ),
            ],
            diagnostics: Vec::new(),
        }
    }

    fn thin_box_geometry() -> GeometryAsset {
        let mut geometry = cube_geometry();
        geometry.geometry_id = "geo_tet_thin_box".to_string();
        for vertex in geometry.surface_meshes[0].vertices.iter_mut().skip(4) {
            vertex[2] = 0.2;
        }
        geometry
    }

    fn tetrahedron_geometry() -> GeometryAsset {
        let mut geometry = cube_geometry();
        geometry.geometry_id = "geo_tet_tetrahedron".to_string();
        geometry.meshes[0].vertex_count = 4;
        geometry.meshes[0].element_count = 4;
        geometry.surface_meshes = vec![SurfaceMesh::new(
            "cube_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )];
        geometry.region_entity_mappings = vec![
            RegionEntityMapping::all_faces("region_fixed", "cube_surface", 4),
            RegionEntityMapping::new(
                "region_load",
                "cube_surface",
                runmat_geometry_core::EntityKind::Face,
                vec![runmat_geometry_core::EntityIdRange::new(2, 2)],
            ),
        ];
        geometry
    }

    #[test]
    fn structured_tet_mesher_generates_valid_analysis_mesh() {
        let geometry = cube_geometry();
        let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
            .expect("cube should produce an analysis mesh");

        validate_analysis_mesh(&mesh, Default::default()).expect("analysis mesh should validate");
        assert_eq!(mesh.schema_version, ANALYSIS_MESH_SCHEMA_VERSION);
        assert_eq!(mesh.nodes.len(), 27);
        assert_eq!(mesh.volume_elements.len(), 48);
        assert_eq!(mesh.boundary_faces.len(), 48);
        assert!(mesh.quality.mean_boundary_projection_error_m <= 1.0e-12);
        assert!(mesh.quality.max_boundary_projection_error_m <= 1.0e-12);
        assert!(mesh.boundary_faces.iter().any(|face| {
            face.region_ids
                .iter()
                .any(|region| region == "region_fixed")
        }));
        assert!(mesh
            .boundary_faces
            .iter()
            .any(|face| face.region_ids.iter().any(|region| region == "region_load")));
        assert!(mesh
            .quality
            .elements
            .iter()
            .all(|quality| quality.volume_m3 > 0.0));
        assert!(mesh
            .boundary_faces
            .iter()
            .all(|face| !face.adjacent_volume_element_ids.is_empty()));
        assert!(mesh.boundary_faces.iter().all(|face| {
            face.adjacent_volume_element_ids.iter().all(|element_id| {
                mesh.volume_elements
                    .iter()
                    .any(|element| element.element_id == *element_id)
            })
        }));
        assert!(mesh.volume_elements.iter().all(|element| {
            tet_volume(
                [
                    element.node_ids[0],
                    element.node_ids[1],
                    element.node_ids[2],
                    element.node_ids[3],
                ],
                &mesh.nodes,
            ) > 0.0
        }));
    }

    #[test]
    fn target_size_controls_structured_tet_density() {
        let geometry = cube_geometry();
        let coarse = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                kind: MeshKindRequest::Solid,
                target_size: MeshTargetSize::LengthM(1.0),
                ..VolumeMeshingOptions::default()
            },
        )
        .expect("coarse mesh should generate");
        let fine = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                kind: MeshKindRequest::Solid,
                target_size: MeshTargetSize::LengthM(0.25),
                max_elements: 10_000,
                ..VolumeMeshingOptions::default()
            },
        )
        .expect("fine mesh should generate");

        assert!(fine.volume_elements.len() > coarse.volume_elements.len());
        assert!(fine.nodes.len() > coarse.nodes.len());
    }

    #[test]
    fn structured_tet_mesher_carves_cells_outside_closed_surface() {
        let geometry = tetrahedron_geometry();
        let mesh = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                kind: MeshKindRequest::Solid,
                target_size: MeshTargetSize::LengthM(0.25),
                max_elements: 10_000,
                ..VolumeMeshingOptions::default()
            },
        )
        .expect("tetrahedron should produce an analysis mesh");

        validate_analysis_mesh(&mesh, Default::default()).expect("carved mesh should validate");
        assert!(!mesh.volume_elements.is_empty());
        assert!(mesh.volume_elements.len() < 4 * 4 * 4 * 6);
        assert!(mesh.nodes.len() < 5 * 5 * 5);
        assert!(all_nodes_are_referenced(&mesh));
        assert!(mesh.quality.mean_boundary_projection_error_m.is_finite());
        assert!(mesh.quality.max_boundary_projection_error_m.is_finite());
        assert!(mesh.quality.max_boundary_projection_error_m > 0.0);
        assert!(mesh.volume_elements.iter().all(|element| {
            let centroid = tet_centroid(
                [
                    element.node_ids[0],
                    element.node_ids[1],
                    element.node_ids[2],
                    element.node_ids[3],
                ],
                &mesh.nodes,
            );
            point_inside_closed_surface(
                &BoundaryMeshInput::from_geometry(&geometry).expect("boundary input"),
                centroid,
            )
        }));
        assert!(mesh.boundary_faces.len() < 6 * 4 * 4 * 2);
        assert!(mesh
            .boundary_faces
            .iter()
            .all(|face| !face.adjacent_volume_element_ids.is_empty()));
    }

    #[test]
    fn occupied_cells_keep_largest_connected_component() {
        let grid = StructuredGrid {
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            y: vec![0.0, 1.0],
            z: vec![0.0, 1.0],
        };
        let occupied = vec![true, true, false, true];

        let retained = largest_connected_occupied_component(&grid, occupied);

        assert_eq!(retained, vec![true, true, false, false]);
    }

    #[test]
    fn boundary_triangle_centroids_mark_intersected_cells_occupied() {
        let input = BoundaryMeshInput {
            mesh_id: "surface".to_string(),
            source_geometry_id: "geo_surface".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
            vertices: vec![[1.2, 0.1, 0.1], [1.8, 0.1, 0.1], [1.5, 0.8, 0.1]],
            triangles: vec![BoundaryMeshTriangle {
                triangle_id: 0,
                node_ids: [0, 1, 2],
                region_ids: vec!["region".to_string()],
                provenance: Vec::new(),
            }],
            region_ids: vec!["region".to_string()],
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [2.0, 1.0, 1.0],
        };
        let grid = StructuredGrid {
            x: vec![0.0, 1.0, 2.0],
            y: vec![0.0, 1.0],
            z: vec![0.0, 1.0],
        };

        let cells = boundary_triangle_centroid_cells(&input, &grid);

        assert_eq!(cells, vec![false, true]);
    }

    #[test]
    fn boundary_projection_accepts_quality_preserving_move() {
        let input = BoundaryMeshInput {
            mesh_id: "projection_surface".to_string(),
            source_geometry_id: "geo_projection_surface".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
            vertices: vec![[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]],
            triangles: vec![BoundaryMeshTriangle {
                triangle_id: 0,
                node_ids: [0, 1, 2],
                region_ids: vec!["region_surface".to_string()],
                provenance: Vec::new(),
            }],
            bounds_min_m: [-10.0, -10.0, 0.0],
            bounds_max_m: [10.0, 10.0, 2.0],
            region_ids: vec!["region_surface".to_string()],
        };
        let nodes = vec![
            AnalysisMeshNode {
                node_id: 1,
                coordinates_m: [0.0, 0.0, 0.5],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 2,
                coordinates_m: [1.0, 0.0, 0.5],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 3,
                coordinates_m: [0.0, 1.0, 0.5],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 4,
                coordinates_m: [0.0, 0.0, 2.0],
                provenance: Vec::new(),
            },
        ];
        let volume_elements = vec![AnalysisVolumeElement {
            element_id: "tet_1".to_string(),
            kind: VolumeElementKind::Tet4,
            node_ids: vec![1, 2, 3, 4],
            material_region_id: "region_surface".to_string(),
            provenance: Vec::new(),
        }];
        let boundary_faces = vec![AnalysisBoundaryFace {
            face_id: "bf_1".to_string(),
            kind: BoundaryElementKind::Tri3,
            node_ids: vec![1, 2, 3],
            adjacent_volume_element_ids: vec!["tet_1".to_string()],
            region_ids: vec!["region_surface".to_string()],
            provenance: Vec::new(),
        }];
        let original_quality = quality_report(
            element_quality_for_nodes(&volume_elements, &nodes).expect("quality"),
            boundary_projection_errors(&input, &boundary_faces, &nodes),
        );

        let (projected_nodes, projected_quality) = project_boundary_nodes_if_quality_improves(
            &input,
            nodes,
            &volume_elements,
            &boundary_faces,
            original_quality.clone(),
        );

        assert!(
            projected_quality.max_boundary_projection_error_m
                < original_quality.max_boundary_projection_error_m
        );
        assert!(
            projected_quality.min_scaled_jacobian
                >= QualityThresholds::default().min_scaled_jacobian
        );
        assert!(projected_nodes[0].coordinates_m[2] < 0.5);
    }

    fn all_nodes_are_referenced(mesh: &AnalysisMeshArtifact) -> bool {
        mesh.nodes.iter().all(|node| {
            mesh.volume_elements
                .iter()
                .any(|element| element.node_ids.contains(&node.node_id))
                || mesh
                    .boundary_faces
                    .iter()
                    .any(|face| face.node_ids.contains(&node.node_id))
        })
    }

    #[test]
    fn sizing_field_controls_structured_tet_density() {
        let geometry = cube_geometry();
        let base_options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        let mut base_options = base_options;
        base_options.refinement.focus.curvature = false;
        base_options.refinement.focus.small_features = false;
        let coarse = generate_analysis_mesh(&geometry, base_options.clone())
            .expect("coarse mesh should generate");
        let sizing = MeshSizingField {
            samples: vec![SizingSample {
                position_m: [0.5, 0.5, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.stress_gradient".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let refined = generate_analysis_mesh_with_sizing(&geometry, base_options, &sizing)
            .expect("sizing-driven mesh should generate");

        assert!(refined.volume_elements.len() > coarse.volume_elements.len());
        assert_eq!(refined.sizing.samples.len(), 1);
        assert_eq!(
            refined.sizing.samples[0].reason.as_deref(),
            Some("structural.stress_gradient")
        );
    }

    #[test]
    fn sizing_field_creates_local_structured_breakpoints() {
        let geometry = cube_geometry();
        let mut options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = false;
        let sizing = MeshSizingField {
            samples: vec![SizingSample {
                position_m: [0.4, 0.4, 0.4],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let mesh = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
            .expect("local sizing-driven mesh should generate");

        validate_analysis_mesh(&mesh, Default::default()).expect("local mesh should validate");
        assert_eq!(mesh.sizing.applied_samples.len(), 1);
        assert_eq!(
            mesh.sizing.applied_samples[0].reason.as_deref(),
            Some("structural.load_regions")
        );
        assert!(mesh.sizing.applied_samples[0].inserted_breakpoint_count > 0);
        let x = unique_axis_coordinates(&mesh, 0);
        assert!(x.iter().any(|value| (*value - 0.4).abs() <= 1.0e-12));
        let spacings = x
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect::<Vec<_>>();
        let min_spacing = spacings.iter().copied().fold(f64::INFINITY, f64::min);
        let max_spacing = spacings.iter().copied().fold(0.0_f64, f64::max);
        assert!(min_spacing <= 0.25 + 1.0e-12);
        assert!(max_spacing > min_spacing * 1.5);
    }

    #[test]
    fn sizing_field_reports_duplicate_and_invalid_samples() {
        let geometry = cube_geometry();
        let mut options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = false;
        let sizing = MeshSizingField {
            samples: vec![
                SizingSample {
                    position_m: [0.5, 0.5, 0.5],
                    target_size_m: 0.5,
                    reason: Some("structural.stress_gradient".to_string()),
                },
                SizingSample {
                    position_m: [0.5, 0.5, 0.5],
                    target_size_m: 0.5,
                    reason: Some("structural.stress_gradient".to_string()),
                },
                SizingSample {
                    position_m: [f64::NAN, 0.5, 0.5],
                    target_size_m: 0.25,
                    reason: Some("structural.invalid_position".to_string()),
                },
                SizingSample {
                    position_m: [0.25, 0.25, 0.25],
                    target_size_m: f64::NAN,
                    reason: Some("structural.invalid_size".to_string()),
                },
            ],
            ..MeshSizingField::default()
        };

        let mesh = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
            .expect("audited sizing mesh should generate");

        assert_eq!(mesh.sizing.applied_samples.len(), 1);
        assert_eq!(
            mesh.sizing.applied_samples[0].reason.as_deref(),
            Some("structural.stress_gradient")
        );
        assert!(mesh.sizing.rejected_samples.iter().any(|rejection| {
            rejection.status == "skipped_duplicate"
                && rejection.reason.as_deref() == Some("structural.stress_gradient")
        }));
        assert!(mesh.sizing.rejected_samples.iter().any(|rejection| {
            rejection.status == "skipped_invalid"
                && rejection.reason.as_deref() == Some("structural.invalid_position")
        }));
        assert!(mesh.sizing.rejected_samples.iter().any(|rejection| {
            rejection.status == "skipped_invalid"
                && rejection.reason.as_deref() == Some("structural.invalid_size")
        }));
    }

    #[test]
    fn sizing_field_skips_breakpoints_that_would_violate_quality() {
        let geometry = cube_geometry();
        let mut options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = false;
        let sizing = MeshSizingField {
            samples: vec![SizingSample {
                position_m: [0.2, 0.2, 0.2],
                target_size_m: 0.01,
                reason: Some("structural.stress_gradient".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let mesh = generate_analysis_mesh_with_sizing(&geometry, options, &sizing)
            .expect("quality-guarded local sizing mesh should generate");

        validate_analysis_mesh(&mesh, Default::default())
            .expect("quality-guarded local mesh should validate");
        assert!(
            mesh.quality.min_scaled_jacobian >= QualityThresholds::default().min_scaled_jacobian
        );
        assert!(mesh.sizing.rejected_samples.iter().any(|rejection| {
            rejection.status == "skipped_quality"
                && rejection.reason.as_deref() == Some("structural.stress_gradient")
        }));
        let min_spacing = unique_axis_coordinates(&mesh, 0)
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .fold(f64::INFINITY, f64::min);
        assert!(min_spacing > 0.01);
    }

    #[test]
    fn sizing_field_refinement_respects_element_budget() {
        let geometry = cube_geometry();
        let sizing = MeshSizingField {
            samples: vec![SizingSample {
                position_m: [0.5, 0.5, 0.5],
                target_size_m: 0.01,
                reason: Some("structural.stress_gradient".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let mesh = generate_analysis_mesh_with_sizing(
            &geometry,
            VolumeMeshingOptions {
                target_size: MeshTargetSize::LengthM(1.0),
                max_elements: 48,
                ..VolumeMeshingOptions::default()
            },
            &sizing,
        )
        .expect("budgeted sizing-driven mesh should generate");

        assert!(mesh.volume_elements.len() <= 48);
        assert_eq!(mesh.volume_elements.len(), 48);
        assert!(mesh
            .sizing
            .rejected_samples
            .iter()
            .any(|rejection| rejection.status == "skipped_budget"));
    }

    #[test]
    fn curvature_focus_adds_geometry_sizing_samples() {
        let geometry = cube_geometry();
        let mut coarse_options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        coarse_options.refinement.focus.curvature = false;
        coarse_options.refinement.focus.small_features = false;
        let coarse = generate_analysis_mesh(&geometry, coarse_options.clone())
            .expect("coarse mesh should generate");

        coarse_options.refinement.focus.curvature = true;
        let focused = generate_analysis_mesh(&geometry, coarse_options)
            .expect("curvature-focused mesh should generate");

        assert!(focused.volume_elements.len() > coarse.volume_elements.len());
        assert!(focused
            .sizing
            .samples
            .iter()
            .any(|sample| sample.reason.as_deref() == Some("geometry.curvature")));
    }

    #[test]
    fn small_feature_focus_adds_geometry_sizing_samples() {
        let geometry = thin_box_geometry();
        let mut coarse_options = VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(1.0),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        };
        coarse_options.refinement.focus.curvature = false;
        coarse_options.refinement.focus.small_features = false;
        let coarse = generate_analysis_mesh(&geometry, coarse_options.clone())
            .expect("coarse thin-box mesh should generate");

        coarse_options.refinement.focus.small_features = true;
        let focused = generate_analysis_mesh(&geometry, coarse_options)
            .expect("small-feature-focused mesh should generate");

        assert!(focused.volume_elements.len() > coarse.volume_elements.len());
        assert!(focused
            .sizing
            .samples
            .iter()
            .any(|sample| sample.reason.as_deref() == Some("geometry.small_features")));
    }

    #[test]
    fn invalid_open_shell_returns_meshing_error() {
        let mut geometry = cube_geometry();
        geometry.surface_meshes[0].triangles.pop();
        geometry.meshes[0].element_count -= 1;

        let err = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
            .expect_err("open shell should fail");

        assert!(matches!(err, MeshingError::BoundaryInput(_)));
        assert!(err.to_string().contains("incidence"));
    }

    #[test]
    fn unsupported_element_kind_is_rejected() {
        let geometry = cube_geometry();
        let err = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                element: VolumeElementKind::Hex8,
                ..VolumeMeshingOptions::default()
            },
        )
        .expect_err("hex backend is not implemented");

        assert_eq!(
            err,
            MeshingError::UnsupportedElementKind(VolumeElementKind::Hex8)
        );
    }

    #[test]
    fn unsupported_mesh_kind_is_rejected() {
        let geometry = cube_geometry();
        let err = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                kind: MeshKindRequest::Surrogate,
                ..VolumeMeshingOptions::default()
            },
        )
        .expect_err("surrogate compatibility mode is not an analysis mesh backend");

        assert_eq!(
            err,
            MeshingError::UnsupportedMeshKind(MeshKindRequest::Surrogate)
        );
        assert!(err.to_string().contains("unsupported analysis mesh kind"));
    }

    #[test]
    fn auto_backend_uses_structured_fallback_until_production_backend_exists() {
        let geometry = cube_geometry();
        let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
            .expect("auto backend should generate with fallback");

        assert_eq!(mesh.provenance.algorithm, "structured_bbox_tet/v1");
    }

    #[test]
    fn explicit_production_backend_fails_until_backend_exists() {
        let geometry = cube_geometry();
        let err = generate_analysis_mesh(
            &geometry,
            VolumeMeshingOptions {
                backend: MeshBackendKind::Production,
                ..VolumeMeshingOptions::default()
            },
        )
        .expect_err("production backend is not wired yet");

        match err {
            MeshingError::ProductionBackend(message) => {
                assert!(message.contains("production Tet generation is pending"));
                assert!(message.contains("volume component"));
            }
            other => panic!("unexpected production error: {other:?}"),
        }
    }

    #[test]
    fn volume_meshing_options_default_backend_when_deserializing_old_payloads() {
        let options: VolumeMeshingOptions = serde_json::from_value(serde_json::json!({
            "kind": "solid",
            "element": "tet4",
            "element_order": "linear",
            "profile": "analysis_ready",
            "max_elements": 250000,
            "target_size": "auto",
            "refinement": {
                "strategy": "auto",
                "max_iterations": 4,
                "convergence": {
                    "field_change_tolerance": 0.05,
                    "energy_change_tolerance": 0.02
                },
                "focus": {
                    "loads": "fine",
                    "constraints": "fine",
                    "interfaces": "normal",
                    "curvature": true,
                    "small_features": true
                }
            }
        }))
        .expect("old mesh options payload should deserialize");

        assert_eq!(options.backend, MeshBackendKind::Auto);
    }

    fn unique_axis_coordinates(mesh: &AnalysisMeshArtifact, axis: usize) -> Vec<f64> {
        let mut coordinates = mesh
            .nodes
            .iter()
            .map(|node| node.coordinates_m[axis])
            .collect::<Vec<_>>();
        coordinates.sort_by(f64::total_cmp);
        coordinates.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-12);
        coordinates
    }

    fn tet_centroid(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> [f64; 3] {
        let points = tet_points(node_ids, nodes).expect("test tet nodes should resolve");
        [
            (points[0][0] + points[1][0] + points[2][0] + points[3][0]) * 0.25,
            (points[0][1] + points[1][1] + points[2][1] + points[3][1]) * 0.25,
            (points[0][2] + points[1][2] + points[2][2] + points[3][2]) * 0.25,
        ]
    }
}
