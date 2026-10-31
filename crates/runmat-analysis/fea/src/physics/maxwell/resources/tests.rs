use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn exact_local_ceiling_and_raw_metadata_are_counted_before_expansion() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        let reservation = require_canonical_ceiling(&model, &mesh, u64::MAX).unwrap();
        assert_eq!(reservation.edge_count_upper_bound, 6);
        assert_eq!(reservation.real_block_dimension_upper_bound, 12);
        // One complex entry per edge pair of each element.
        assert_eq!(reservation.support_entries_upper_bound, 36);
        assert_eq!(reservation.quadrature_sample_count, 15);
        assert_eq!(
            require_canonical_ceiling(&model, &mesh, reservation.total_bytes).unwrap(),
            reservation
        );
        assert!(require_canonical_ceiling(&model, &mesh, reservation.total_bytes - 1).is_err());
        mesh.topology.volume_elements[0]
            .region_id
            .assembly_path
            .push("instance".repeat(512));
        let large = require_canonical_ceiling(&model, &mesh, u64::MAX).unwrap();
        assert!(large.total_bytes > reservation.total_bytes);
        // Malformed raw connectivity/stale digest is profiled, not projected.
        // Thus malformed input cannot bypass admission by delaying its size.
        mesh.topology.volume_elements[0].node_ids.extend(100..200);
        assert!(
            require_canonical_ceiling(&model, &mesh, u64::MAX)
                .unwrap()
                .total_bytes
                > large.total_bytes
        );
        let mut denied = model.clone();
        denied.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(0);
        let step_id = denied.sequence.steps[0].step_id.as_str();
        let error =
            crate::physics::maxwell::compile_canonical_input(&denied, &mesh, step_id, 1e-8, 1.0)
                .unwrap_err();
        assert!(error
            .to_string()
            .contains("Maxwell local resource preflight"));
    }
}

#[test]
fn checked_counts_and_actual_shared_krylov_counter_fail_closed() {
    for values in [
        (1, u64::MAX, 1, 0, 0, 0, 0),
        (u64::MAX, 1, 1, 0, 0, 0, 0),
        (1, 1, 1, u64::MAX, 0, 0, 0),
        (1, 1, 1, 0, u64::MAX, 0, 0),
        (1, 1, 1, 0, 0, u64::MAX, 0),
        (1, 1, 1, 0, 0, 0, u64::MAX),
    ] {
        assert!(
            from_counts(values.0, values.1, values.2, values.3, values.4, values.5, values.6)
                .is_err()
        );
    }
    // The sparse LU reservation scales with the element count, with no iteration-counter cap.
    let huge = from_counts(1, u64::from(u32::MAX) / 480 + 1, 1, 0, 0, 0, 0).unwrap();
    assert!(huge.assembly_and_solver_bytes > 16 * (1u64 << 30));
    let small = from_counts(10, 1, 4, 100, 100, 100, 0).unwrap();
    let large = from_counts(27, 8, 24, 101, 101, 101, 0).unwrap();
    let current_drive = from_counts(10, 1, 4, 100, 100, 100, 1).unwrap();
    assert_eq!(current_drive.real_block_dimension_upper_bound, 14);
    assert_eq!(current_drive.support_entries_upper_bound, 49);
    assert!(current_drive.assembly_and_solver_bytes > small.assembly_and_solver_bytes);
    assert!(large.total_bytes > small.total_bytes);
    assert_eq!(
        small.total_bytes,
        small.topology_and_properties_bytes
            + small.compiled_and_retained_bytes
            + small.assembly_and_solver_bytes
    );
}

#[test]
fn replicated_property_metadata_and_selected_policy_are_not_ignored() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut big = model.clone();
    big.materials[0]
        .name
        .push_str(&"material metadata ".repeat(4096));
    let ordinary = require_canonical_ceiling(&model, &mesh, u64::MAX).unwrap();
    let grown = require_canonical_ceiling(&big, &mesh, u64::MAX).unwrap();
    assert!(grown.topology_and_properties_bytes > ordinary.topology_and_properties_bytes);
    let id = big.sequence.steps[0].step_id.clone();
    big.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(grown.total_bytes);
    // The authored ceiling is part of the serialized model, so re-measure with it set.
    let exact = require_canonical_ceiling(&big, &mesh, u64::MAX)
        .unwrap()
        .total_bytes;
    big.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(exact);
    assert!(require_selected_ceiling(&big, &mesh, id.as_str()).is_ok());
    big.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(exact - 1);
    assert!(require_selected_ceiling(&big, &mesh, id.as_str()).is_err());
    assert!(require_selected_ceiling(&big, &mesh, "foreign-step").is_err());
    big.sequence.steps.push(big.sequence.steps[0].clone());
    assert!(require_selected_ceiling(&big, &mesh, id.as_str()).is_err());
}
