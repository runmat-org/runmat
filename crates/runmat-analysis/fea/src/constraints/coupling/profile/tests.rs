use super::*;
use runmat_analysis_core::RegionTarget;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
#[test]
fn profile_bounds_original_coupling_buffers_for_both_orders_and_rejects_overflow() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let target = super::super::super::target::resolve_nodal_target(
            &mesh,
            &RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
            "surface",
        )
        .unwrap();
        for kind in [
            ConstraintCouplingKind::Rigid,
            ConstraintCouplingKind::Distributing,
        ] {
            let p = constraint_coupling_workspace_profile(&mesh, kind, "surface").unwrap();
            assert_eq!(p.coordinate_slots, mesh.topology.nodes.len());
            assert_eq!(p.measure_slots, mesh.topology.nodes.len());
            let reference = ReferencePoint {
                reference_point_id: "reference".into(),
                position: runmat_analysis_core::Position::new([0.2, 0.3, -1.]),
            };
            let (rows, evidence) = super::super::compile_reference_coupling(
                &mesh,
                &target,
                &reference,
                Digest::sha256(b"reference"),
                kind,
                "surface",
            )
            .unwrap();
            assert!(p.weight_slots >= evidence.target_weights.capacity());
            assert!(p.row_slots >= rows.capacity());
            let terms: usize = rows.iter().map(|r| r.terms.capacity()).sum();
            assert!(p.retained_term_slots >= terms);
        }
    }
    assert!(rows_profile(usize::MAX, ConstraintCouplingKind::Rigid, 7).is_err());
    assert!(rows_profile(usize::MAX, ConstraintCouplingKind::Distributing, 7).is_err());
}
