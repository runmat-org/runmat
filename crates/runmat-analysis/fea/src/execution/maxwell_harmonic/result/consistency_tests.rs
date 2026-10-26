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
            .canonical_values()
            .iter()
            .position(|value| *value != crate::physics::maxwell::ComplexPhasor::ZERO)
            .unwrap();
        changed.data.response.edge_circulations_v = changed
            .data
            .response
            .edge_circulations_v
            .clone()
            .try_map_canonical_storage(|mut values| {
                values[interior].real += 1.0;
                values
            })
            .unwrap();
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
        response.edge_circulations_v = response
            .edge_circulations_v
            .clone()
            .try_map_canonical_storage(|values| {
                values
                    .into_iter()
                    .map(|value| value * 2.0)
                    .collect::<Vec<_>>()
            })
            .unwrap();
        for sample in &mut response.samples {
            sample.electric_field_v_per_m = sample
                .electric_field_v_per_m
                .try_map_canonical_storage(|values| values.map(|value| value * 2.0))
                .unwrap();
            sample.flux_density_t = sample
                .flux_density_t
                .try_map_canonical_storage(|values| values.map(|value| value * 2.0))
                .unwrap();
            sample.magnetic_field_a_per_m = sample
                .magnetic_field_a_per_m
                .try_map_canonical_storage(|values| values.map(|value| value * 2.0))
                .unwrap();
            sample.electric_displacement_c_per_m2 = sample
                .electric_displacement_c_per_m2
                .try_map_canonical_storage(|values| values.map(|value| value * 2.0))
                .unwrap();
            sample.conduction_current_density_a_per_m2 = sample
                .conduction_current_density_a_per_m2
                .try_map_canonical_storage(|values| values.map(|value| value * 2.0))
                .unwrap();
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
                .contains("differs from re-evaluation of its original equations"),
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
                .contains("differs from re-evaluation of its original equations"),
            "{error}"
        );
    }
}
