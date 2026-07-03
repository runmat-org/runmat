use super::*;

pub(super) struct GeneratedGridTetrahedrons {
    pub(super) volume_elements: Vec<AnalysisVolumeElement>,
    pub(super) element_quality: Vec<ElementQuality>,
    pub(super) cell_tetrahedron_ids: Vec<Option<Vec<String>>>,
}

pub(super) fn generate_grid_tetrahedra(
    input: &BoundaryMeshInput,
    grid: &StructuredGrid,
    nodes: &[AnalysisMeshNode],
    occupied_cells: &[bool],
) -> GeneratedGridTetrahedrons {
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

    let mut cell_tetrahedron_ids = vec![None; grid.cell_count()];
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
                    node_id_at(grid, i, j, k),
                    node_id_at(grid, i + 1, j, k),
                    node_id_at(grid, i, j + 1, k),
                    node_id_at(grid, i + 1, j + 1, k),
                    node_id_at(grid, i, j, k + 1),
                    node_id_at(grid, i + 1, j, k + 1),
                    node_id_at(grid, i, j + 1, k + 1),
                    node_id_at(grid, i + 1, j + 1, k + 1),
                ];
                for tetrahedron in [
                    [cell_nodes[0], cell_nodes[1], cell_nodes[3], cell_nodes[7]],
                    [cell_nodes[0], cell_nodes[3], cell_nodes[2], cell_nodes[7]],
                    [cell_nodes[0], cell_nodes[2], cell_nodes[6], cell_nodes[7]],
                    [cell_nodes[0], cell_nodes[6], cell_nodes[4], cell_nodes[7]],
                    [cell_nodes[0], cell_nodes[4], cell_nodes[5], cell_nodes[7]],
                    [cell_nodes[0], cell_nodes[5], cell_nodes[1], cell_nodes[7]],
                ] {
                    let element_id = format!("tetrahedron_{}", volume_elements.len() + 1);
                    let oriented = orient_tetrahedron(tetrahedron, nodes);
                    let volume_m3 = tetrahedron_volume(oriented, nodes).abs();
                    let aspect_ratio = tetrahedron_aspect_ratio(oriented, nodes);
                    let exact_scaled_jacobian = tetrahedron_points(oriented, nodes)
                        .map(tetrahedron_scaled_jacobian)
                        .unwrap_or(0.0);
                    element_quality.push(ElementQuality {
                        element_id: element_id.clone(),
                        scaled_jacobian: 1.0 / aspect_ratio.max(1.0),
                        exact_scaled_jacobian,
                        aspect_ratio,
                        volume_m3,
                    });
                    volume_elements.push(AnalysisVolumeElement {
                        element_id: element_id.clone(),
                        kind: VolumeElementKind::Tetrahedron4,
                        node_ids: oriented.to_vec(),
                        material_region_id: material_region_id.clone(),
                        provenance: provenance.clone(),
                    });
                    cell_tetrahedron_ids[cell_index]
                        .get_or_insert_with(Vec::new)
                        .push(element_id);
                }
            }
        }
    }

    GeneratedGridTetrahedrons {
        volume_elements,
        element_quality,
        cell_tetrahedron_ids,
    }
}
