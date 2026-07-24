use std::collections::BTreeMap;

use super::*;
use runmat_meshing_core::contracts::{
    artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisMeshProvenance, MeshBackendSummary,
};

pub(super) fn build_analysis_mesh_artifact(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    grid: &StructuredGrid,
    parts: AnalysisMeshArtifactParts,
) -> AnalysisMeshArtifact {
    let AnalysisMeshArtifactParts {
        nodes,
        volume_elements,
        boundary_faces,
        quality,
        mut sizing,
    } = parts;
    sizing.global_target_size_m = target_size_m(input, options, grid);
    if sizing.min_size_m.is_none() {
        sizing.min_size_m = options.min_size_m;
    }
    if sizing.max_size_m.is_none() {
        sizing.max_size_m = options.max_size_m;
    }
    if sizing.growth_rate.is_none() {
        sizing.growth_rate = options.growth_rate;
    }

    let tetrahedron_element_count = volume_elements.len();
    compact_analysis_mesh_nodes(AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: format!("analysis_{}", input.mesh_id),
        nodes,
        volume_elements,
        boundary_faces,
        boundary_edges: Vec::new(),
        quality,
        sizing,
        field_topology: Vec::new(),
        backend: MeshBackendSummary {
            backend: "structured_grid_tetrahedron".to_string(),
            algorithm: "structured_bbox_tetrahedron/v1".to_string(),
            tetrahedron_element_count,
            boundary_face_recovery_ratio: 1.0,
            ..MeshBackendSummary::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "structured_bbox_tetrahedron/v1".to_string(),
            source_geometry_id: input.source_geometry_id.clone(),
            source_geometry_revision: input.source_geometry_revision,
            source_geometry_sha256: input.source_geometry_sha256.clone(),
        },
    })
}

pub(super) struct AnalysisMeshArtifactParts {
    pub(super) nodes: Vec<AnalysisMeshNode>,
    pub(super) volume_elements: Vec<AnalysisVolumeElement>,
    pub(super) boundary_faces: Vec<AnalysisBoundaryFace>,
    pub(super) quality: AnalysisMeshQualityReport,
    pub(super) sizing: MeshSizingField,
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
        mesh.refresh_field_topology();
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
    mesh.refresh_field_topology();
    mesh
}

fn remap_node_ids(node_ids: &mut [u32], node_id_map: &BTreeMap<u32, u32>) {
    for node_id in node_ids {
        if let Some(new_node_id) = node_id_map.get(node_id) {
            *node_id = *new_node_id;
        }
    }
}

fn target_size_m(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    grid: &StructuredGrid,
) -> Option<f64> {
    let target_size = match options.target_size {
        MeshTargetSize::LengthM(value) => Some(value),
        MeshTargetSize::Auto => grid.min_cell_size().or_else(|| {
            let max_span = (0..3)
                .map(|axis| input.bounds_max_m[axis] - input.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            Some(max_span)
        }),
    };
    target_size.map(|value| clamp_mesh_target_size(value, options))
}

fn clamp_mesh_target_size(mut value: f64, options: &VolumeMeshingOptions) -> f64 {
    if let Some(min_size_m) = options.min_size_m {
        value = value.max(min_size_m);
    }
    if let Some(max_size_m) = options.max_size_m {
        value = value.min(max_size_m);
    }
    value
}
