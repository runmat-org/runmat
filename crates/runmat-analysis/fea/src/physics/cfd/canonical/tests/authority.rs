//! Admission/selection witnesses, not additional continuum accuracy claims.
use super::*;
use runmat_analysis_core::{EntityDimension, ExcitationTarget, RegionTarget};

#[test]
fn body_force_requires_every_original_volume_and_volume_kind() {
    let mut fixture = case(1, ReferenceCase::Duct, false);
    let valid = fixture.source.elements[0].region_id.clone();
    let mut missing = valid.clone();
    missing.source_topology_id = "missing-original-fluid-volume".into();
    let mut target = RegionTarget::volume(runmat_meshing_core::fixtures::fixture_region(
        "exact-volume-selection",
    ));
    target
        .bind_resolved_entities(vec![valid.clone(), missing])
        .unwrap();
    let ExcitationAction::BodyForce(action) = &mut fixture.actions[0] else {
        unreachable!()
    };
    action.target = ExcitationTarget::Region(target);
    assert!(matches!(
        CanonicalFluidProblem::prepare(fixture.input()),
        Err(CanonicalFluidError::Invalid(
            "incomplete original bodyforce volume coverage"
        ))
    ));

    let mut target = RegionTarget::volume(runmat_meshing_core::fixtures::fixture_region(
        "exact-volume-selection",
    ));
    target.bind_resolved_entities(vec![valid]).unwrap();
    target.entity_dimension = EntityDimension::Face;
    let ExcitationAction::BodyForce(action) = &mut fixture.actions[0] else {
        unreachable!()
    };
    action.target = ExcitationTarget::Region(target);
    assert!(matches!(
        CanonicalFluidProblem::prepare(fixture.input()),
        Err(CanonicalFluidError::Invalid(
            "bodyforce exact volume target"
        ))
    ));
}

#[test]
fn disconnected_outlet_references_use_one_face_traversal_and_linear_envelope() {
    let fixture = case(1, ReferenceCase::Couette, false);
    // Connectivity-only records exercise 1024 components. These are intentionally
    // not submitted as a valid geometric mesh or used as a physical reference.
    let mut topology = fixture.topology.clone();
    let count = 1024;
    topology.cells = vec![topology.cells[0].clone(); count];
    topology.faces = (0..count)
        .map(|owner| {
            let mut face = fixture.topology.faces[0].clone();
            face.owner = owner;
            face.neighbor = None;
            face
        })
        .collect();
    let ids = (0..count)
        .map(|i| format!("component-outlet-{i}"))
        .collect::<Vec<_>>();
    let boundaries = (0..count)
        .map(|i| boundary::FaceBoundary::PressureOutlet {
            authority_id: ids[i].as_str(),
            pressure: i as f64,
        })
        .collect::<Vec<_>>();
    let (_, references) = super::super::pressure_reference::PressureReferencePlan::derive(
        &fixture.source,
        &topology,
        &boundaries,
        &[],
        &[],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    assert_eq!(references, (0..count).map(|i| i as f64).collect::<Vec<_>>());
    let mut missing = boundaries;
    missing[count - 1] = boundary::FaceBoundary::Interior;
    assert!(
        super::super::pressure_reference::PressureReferencePlan::derive(
            &fixture.source,
            &topology,
            &missing,
            &[],
            &[],
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .is_err()
    );

    // One authored gauge reaches all disconnected components without supplying
    // physical face laws. Connectivity-only inventory, not a physical mesh.
    let original = &fixture.source.boundary_faces[topology.faces[0]
        .boundary_source
        .unwrap()
        .projection_boundary_index];
    let mut target = runmat_analysis_core::RegionTarget::boundary(
        runmat_meshing_core::fixtures::fixture_face("original-gauge-inventory"),
    );
    target
        .bind_resolved_entities(original.provenance.clone())
        .unwrap();
    let gauge = runmat_analysis_core::FluidConstraint {
        constraint_id: "shared-component-gauge".into(),
        kind: runmat_analysis_core::FluidConstraintKind::PressureReference {
            target,
            value: runmat_analysis_core::PrescribedValue::Constant(
                runmat_analysis_core::Pressure::new(101325.0),
            ),
        },
    };
    let walls = vec![
        boundary::FaceBoundary::Velocity {
            authority_id: "wall",
            value: [0.0; 3],
            moving_wall: false
        };
        count
    ];
    let (_, closed) = super::super::pressure_reference::PressureReferencePlan::derive(
        &fixture.source,
        &topology,
        &walls,
        &[gauge],
        &[],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    assert_eq!(closed, vec![101325.0; count]);
    let geometry = |cells| crate::topology::FiniteVolumeReservation {
        cell_count: cells,
        face_incidence_count: cells * 4,
        boundary_face_count: cells * 4,
        scratch_bytes: cells * 8192,
        traversal_operations: cells * 8192,
    };
    let small =
        CanonicalFluidReservation::for_counts(&geometry(count), count * 4, policy()).unwrap();
    let large =
        CanonicalFluidReservation::for_counts(&geometry(count * 2), count * 8, policy()).unwrap();
    assert!(large.operator_work_upper_bound <= small.operator_work_upper_bound * 2);
    // Verification of a retained result excludes the solver-only workspace: at least the
    // coupled momentum block (9 x 41 three-word entries per cell) and the GMRES/ILU peaks.
    for reservation in [small, large] {
        let cells = reservation.cells;
        assert!(reservation.verification_resident_bytes_upper_bound > 0);
        assert!(
            reservation.verification_resident_bytes_upper_bound + cells * 9 * 41 * 24
                <= reservation.resident_bytes_upper_bound
        );
    }
}

#[test]
fn actual_authority_profile_is_reserved_and_rejects_before_binding() {
    let fixture = case(1, ReferenceCase::Duct, false);
    let base = resources::reserve_exact(
        &fixture.source,
        &fixture.topology,
        fixture.policy,
        fixture.topology.cells.len() as u64,
    )
    .unwrap();
    let admit = |policy| {
        resources::admit_authority_profile(
            &fixture.source,
            &fixture.constraints,
            &fixture.operators,
            &fixture.signals,
            &fixture.actions,
            policy,
            base,
        )
    };
    let exact = admit(fixture.policy).unwrap();
    assert!(exact.operator_work_upper_bound > base.operator_work_upper_bound);
    assert!(exact.resident_bytes_upper_bound > base.resident_bytes_upper_bound);
    let mut cap = fixture.policy;
    cap.maximum_operator_work = Some(base.operator_work_upper_bound);
    assert!(matches!(
        admit(cap),
        Err(CanonicalFluidError::Resource(
            "source authority operator work"
        ))
    ));
    cap = fixture.policy;
    cap.maximum_operator_work = Some(exact.operator_work_upper_bound);
    cap.maximum_resident_bytes = Some(exact.resident_bytes_upper_bound);
    assert_eq!(admit(cap).unwrap(), exact);
    cap.maximum_authority_items = 1;
    assert!(matches!(admit(cap), Err(CanonicalFluidError::Resource(_))));
}
