use crate::FeaPrepContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegacySurrogateTopology {
    pub dof_count: usize,
    pub solid_element_count: usize,
}

pub fn legacy_surrogate_topology(
    base_dof_count: usize,
    prep_context: Option<&FeaPrepContext>,
) -> LegacySurrogateTopology {
    let dof_count = if let Some(prep) = prep_context {
        let prep_dof = ((prep.prepared_node_count as f64) * prep.topology_dof_multiplier)
            .round()
            .max(base_dof_count as f64) as usize;
        prep_dof.clamp(base_dof_count, base_dof_count.saturating_mul(6).max(3))
    } else {
        base_dof_count
    };
    let solid_element_count = prep_context
        .map(|prep| prep.prepared_element_count.max(prep.prepared_mesh_count))
        .unwrap_or_else(|| element_count_for_legacy_dofs(dof_count));
    LegacySurrogateTopology {
        dof_count,
        solid_element_count,
    }
}

fn element_count_for_legacy_dofs(dof_count: usize) -> usize {
    dof_count.div_ceil(3).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_surrogate_uses_load_based_topology_without_prep() {
        let topology = legacy_surrogate_topology(6, None);
        assert_eq!(topology.dof_count, 6);
        assert_eq!(topology.solid_element_count, 2);
    }

    #[test]
    fn legacy_surrogate_uses_bounded_prep_topology() {
        let prep = FeaPrepContext {
            prepared_mesh_count: 1,
            prepared_node_count: 100,
            prepared_element_count: 180,
            mapped_region_count: 2,
            min_scaled_jacobian: 0.8,
            mean_aspect_ratio: 1.4,
            inverted_element_count: 0,
            mapped_load_count: 1,
            mapped_bc_count: 1,
            layout_seed: 7,
            topology_dof_multiplier: 2.0,
            topology_bandwidth_estimate: 4,
            mapped_region_participation_ratio: 1.0,
            topology_surface_patch_ratio: 0.0,
            topology_volume_core_ratio: 1.0,
            topology_mixed_family_ratio: 0.0,
            topology_region_span_mean: 1.0,
            topology_region_block_count: 1,
            topology_region_mesh_mean: 1.0,
            topology_region_mesh_variance: 0.0,
            topology_triangle_family_ratio: 0.0,
            topology_quad_family_ratio: 0.0,
            topology_tet_family_ratio: 1.0,
            topology_hex_family_ratio: 0.0,
            coordinate_span_x_m: 1.0,
            coordinate_span_y_m: 1.0,
            coordinate_span_z_m: 1.0,
            coordinate_active_dimension_count: 3,
            coordinate_characteristic_length_m: 0.1,
            element_geometry_node_count: 0,
            element_geometry_edge_count: 0,
            mean_element_edge_length_m: 0.0,
            mean_element_area_m2: 0.0,
            element_geometry_coverage_ratio: 0.0,
            reference_element_coordinates_m: [[0.0; 3]; 3],
            reference_element_area_m2: 0.0,
            element_topology_sample_element_count: 0,
            element_topology_sample_edge_count: 0,
            element_topology_sample_edge_nodes: [[0; 2]; 8],
            element_topology_sample_node_coordinates_m: [[0.0; 3]; 8],
            element_topology_sample_element_edges: [[0; 3]; 4],
            element_topology_sample_element_orientations: [[0; 3]; 4],
            element_topology_sample_element_areas_m2: [0.0; 4],
            element_topology_node_coordinates_m: Vec::new(),
            element_topology_edge_nodes: Vec::new(),
            element_topology_element_edges: Vec::new(),
            element_topology_element_orientations: Vec::new(),
            element_topology_element_areas_m2: Vec::new(),
            calibration_profile_override: None,
        };

        let topology = legacy_surrogate_topology(9, Some(&prep));
        assert_eq!(topology.dof_count, 54);
        assert_eq!(topology.solid_element_count, 180);
    }
}
