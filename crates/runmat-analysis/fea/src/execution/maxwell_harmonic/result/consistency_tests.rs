use super::*;
use runmat_meshing_core::ElementOrder;

#[test]
fn cold_original_both_orders_pass_but_fresh_rehashed_edge_substitution_fails() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let workload = super::super::tests::workload();
        let mesh = super::super::tests::mesh(order);
        let original = workload.execute(&mesh).unwrap();
        let bytes = original.canonical_bytes().unwrap();
        let cold = MaxwellHarmonicSolverResult::from_canonical_bytes_for(
            &bytes,
            Digest::sha256(&bytes),
            &workload,
            &mesh,
        )
        .unwrap();
        assert_eq!(cold, original);
        let mut changed = original.clone();
        let interior = changed
            .data
            .response
            .edge_circulations_v
            .iter()
            .position(|value| *value != crate::physics::maxwell::ComplexPhasor::ZERO)
            .unwrap();
        changed.data.response.edge_circulations_v[interior].real += 1.0;
        // Structural validation and a fresh digest do not confer source authority.
        let bytes = changed.canonical_bytes().unwrap();
        assert!(MaxwellHarmonicSolverResult::from_canonical_bytes_for(
            &bytes,
            Digest::sha256(&bytes),
            &workload,
            &mesh,
        )
        .is_err());
    }
}

#[test]
fn coherently_changed_retained_fields_fail_original_matrix_free_equation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let workload = super::super::tests::workload();
        let mesh = super::super::tests::mesh(order);
        let original = workload.execute(&mesh).unwrap();
        let mut response = original.response().clone();
        // Test-only exact binary scaling keeps E/B mutually consistent with
        // the edge coefficients, while violating the unmodified original J.
        for edge in &mut response.edge_circulations_v {
            *edge = *edge * 2.0;
        }
        for sample in &mut response.samples {
            for field in [
                &mut sample.electric_field_v_per_m,
                &mut sample.flux_density_t,
                &mut sample.magnetic_field_a_per_m,
                &mut sample.electric_displacement_c_per_m2,
                &mut sample.conduction_current_density_a_per_m2,
            ] {
                for value in field {
                    *value = *value * 2.0;
                }
            }
        }
        let error = crate::physics::maxwell::validate_retained_consistency(
            &mesh,
            original.input(),
            &response,
            workload.relative_residual_target,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("original physical residual differs"),
            "{error}"
        );
    }
}

#[test]
fn fresh_rehashed_charge_summary_is_bound_to_original_weak_continuity() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let workload = super::super::tests::workload();
        let mesh = super::super::tests::mesh(order);
        let mut changed = workload.execute(&mesh).unwrap();
        changed.data.response.relative_charge_continuity_residual =
            8.0 * workload.relative_residual_target;
        let bytes = changed.canonical_bytes().unwrap();
        let error = MaxwellHarmonicSolverResult::from_canonical_bytes_for(
            &bytes,
            Digest::sha256(&bytes),
            &workload,
            &mesh,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("original charge continuity differs"),
            "{error}"
        );
    }
}
