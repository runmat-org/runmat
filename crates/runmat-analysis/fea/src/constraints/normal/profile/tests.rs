use super::*;
use runmat_analysis_core::RegionTarget;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
#[test]
fn profile_composes_original_projection_and_one_normal_buffer() {
    for (order, width, nodes) in [(ElementOrder::Tet4, 3, 3), (ElementOrder::Tet10, 6, 6)] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let target = super::super::super::target::resolve_nodal_target(
            &mesh,
            &RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
            "normal",
        )
        .unwrap();
        let p = constraint_normal_workspace_profile(&mesh, &target).unwrap();
        assert_eq!(
            p.projection_bytes,
            AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap()
        );
        assert_eq!(p.normal_slots, nodes);
        assert_eq!(p.facet_coordinate_slots, width);
        assert_eq!(p.facet_vector_slots, 2 * width);
        let slots = std::mem::size_of::<(StableDigest, [f64; 3])>() * nodes
            + std::mem::size_of::<[f64; 3]>() * 3 * width;
        assert_eq!(p.total_requested_bytes, p.projection_bytes + slots as u64);
    }
}
