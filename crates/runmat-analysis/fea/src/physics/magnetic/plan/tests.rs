use super::*;
use crate::physics::magnetic::{
    solve_grounded_volume, solve_grounded_volume_with_plan, MagneticVolumeData,
};
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};

#[test]
fn magnetic_system_plan_roundtrip_has_exact_symmetric_support_and_no_gauge_block() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let plan = MagneticSystemPlan::derive(
            &projection,
            &crate::physics::magnetic::exterior_face_identities(&projection),
        )
        .unwrap();
        let bytes = plan.canonical_bytes().unwrap();
        let reopened = MagneticSystemPlan::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(plan, reopened);
        assert_eq!(plan.identity().unwrap(), reopened.identity().unwrap());
        reopened.validate_for(&projection).unwrap();
        let free = plan.free_edge_count();
        assert!(free > 0 && plan.gauge_node_count() > 0);
        let saddle = plan.saddle_pattern();
        for row in 0..saddle.rows() {
            for entry in saddle.row_range(row) {
                let column = saddle.column(entry);
                assert!(row < free || column < free);
                assert!(saddle.row_range(column).any(|i| saddle.column(i) == row));
                if row < free && column < free {
                    assert!(plan
                        .physical_pattern()
                        .row_range(row)
                        .any(|i| plan.physical_pattern().column(i) == column));
                }
            }
        }
        assert_eq!(plan.physical_pattern().rows(), free);
        // Plans are support, not matrix values: changing μ/J cannot change them.
        for (mu, current) in [(1.0, 1.0), (3.0, -2.0), (2.0, 0.0)] {
            let volumes: Vec<_> = projection
                .elements
                .iter()
                .map(|e| MagneticVolumeData {
                    element_identity: e.stable_identity,
                    permeability_h_per_m: runmat_types::physical::MagneticPermeability::new(mu),
                    current_density_a_per_m2: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::ElectricCurrentDensity,
                        _,
                    >::new([0.0, 0.0, current]),
                })
                .collect();
            let explicit =
                solve_grounded_volume_with_plan(&projection, &volumes, &[], 1e-10, &reopened)
                    .unwrap();
            let automatic = solve_grounded_volume(
                &projection,
                &crate::physics::magnetic::exterior_face_identities(&projection),
                &[],
                &volumes,
                1e-10,
            )
            .unwrap();
            assert_eq!(explicit, automatic);
            assert_eq!(
                plan,
                MagneticSystemPlan::derive(
                    &projection,
                    &crate::physics::magnetic::exterior_face_identities(&projection)
                )
                .unwrap()
            );
        }
    }
}

#[test]
fn magnetic_system_plan_rejects_foreign_mesh_and_spliced_support_before_physical_assembly() {
    let projection = AnalysisProjection::derive(&slab_mesh(2)).unwrap();
    let plan = MagneticSystemPlan::derive(
        &projection,
        &crate::physics::magnetic::exterior_face_identities(&projection),
    )
    .unwrap();
    for mesh in [slab_mesh(1), quadratic_slab_mesh(2)] {
        assert!(plan
            .validate_for(&AnalysisProjection::derive(&mesh).unwrap())
            .is_err());
    }
    let mut mutations = Vec::new();
    let mut foreign = plan.clone();
    foreign.data.projection_digest = Digest::sha256(b"foreign projection");
    mutations.push(foreign);
    let mut missing = plan.clone();
    let n = missing.data.physical_pattern.rows() as u64;
    missing.data.physical_pattern =
        SparsePattern::try_new(n, n, vec![0; n as usize + 1], vec![]).unwrap();
    mutations.push(missing);
    let mut missing = plan.clone();
    let n = missing.data.saddle_pattern.rows() as u64;
    missing.data.saddle_pattern =
        SparsePattern::try_new(n, n, vec![0; n as usize + 1], vec![]).unwrap();
    mutations.push(missing);
    let mut foreign = plan.clone();
    foreign.data.gauge_node_identities[0] = StableDigest([255; 32]);
    mutations.push(foreign);
    for mutation in mutations {
        mutation.validate().unwrap(); // Shape validity alone cannot authorize support.
        assert!(mutation.validate_for(&projection).is_err());
        // Empty laws would fail input later; exact-support rejection must occur first.
        let error =
            solve_grounded_volume_with_plan(&projection, &[], &[], 1e-10, &mutation).unwrap_err();
        assert!(error
            .to_string()
            .contains("exact canonical magnetic support"));
    }
    let mut duplicate = plan;
    duplicate.data.free_edge_indices[1] = duplicate.data.free_edge_indices[0];
    assert!(duplicate.validate().is_err());
    assert!(duplicate.canonical_bytes().is_err());
}
