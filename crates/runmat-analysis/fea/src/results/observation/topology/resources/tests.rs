use super::*;
use runmat_meshing_core::{
    fixtures::{canonical_tetrahedron_solver_mesh, quadratic_slab_mesh, slab_mesh},
    ElementOrder,
};

#[test]
fn original_topology_requested_allocations_fit_inventory_and_cap_rejects_before_index_construction()
{
    for mesh in [
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
        slab_mesh(2),
        quadratic_slab_mesh(2),
    ] {
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let budget = solid_observation_topology_resident_bytes(&projection).unwrap();
        let (topology, live, peak) = crate::topology::measure_test_allocations(|| {
            ObservationTopology::from_projection(&projection).unwrap()
        });
        assert!(
            peak as u64 <= budget,
            "requested peak {peak} exceeds {budget}"
        );
        assert!(live <= peak);
        assert!(std::ptr::eq(
            topology.solid_projection.as_deref().unwrap(),
            &projection
        ));
        assert!(matches!(topology.solid_projection, Some(Cow::Borrowed(_))));
        assert_eq!(topology.nodes.len(), projection.nodes.len());
        assert_eq!(
            topology.volume_integration_point_count(),
            4 * projection.elements.len()
        );
        for (index, node) in projection.nodes.iter().enumerate() {
            assert_eq!(topology.node_index(node.node_id).unwrap(), Some(index));
        }
        let (admitted, remaining) =
            ObservationTopology::from_projection_admitted(&projection, budget).unwrap();
        assert_eq!(remaining, 0);
        assert_eq!(admitted.digest(), topology.digest());
        let (_, _, rejected_peak) = crate::topology::measure_test_allocations(|| {
            assert!(matches!(
                ObservationTopology::from_projection_admitted(&projection, budget - 1),
                Err(ObservationError::Limit(
                    "solid observation topology residency"
                ))
            ));
        });
        assert_eq!(
            rejected_peak, 0,
            "count gate must reject without allocating a topology"
        );
        let mut long_labels = projection.clone();
        for element in &mut long_labels.elements {
            element
                .region_id
                .source_topology_id
                .push_str(&"region".repeat(1024));
        }
        let long_budget = solid_observation_topology_resident_bytes(&long_labels).unwrap();
        let (_, _, long_peak) = crate::topology::measure_test_allocations(|| {
            ObservationTopology::from_projection(&long_labels).unwrap()
        });
        assert!(long_peak as u64 <= long_budget);
        assert!(long_budget > budget);
        println!("observation nodes={} elements={} requested_peak={peak} retained={live} admitted={budget}", projection.nodes.len(), projection.elements.len());
    }
}

#[test]
fn the_mesh_topology_law_bounds_the_projection_law_before_any_projection_exists() {
    for mesh in [
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
        slab_mesh(2),
        quadratic_slab_mesh(2),
    ] {
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let from_mesh = solid_observation_topology_resident_bytes_for_mesh(&mesh).unwrap();
        assert!(from_mesh >= solid_observation_topology_resident_bytes(&projection).unwrap());
        let (_, _, peak) = crate::topology::measure_test_allocations(|| {
            ObservationTopology::from_projection(&projection).unwrap()
        });
        assert!(peak as u64 <= from_mesh);
    }
}
