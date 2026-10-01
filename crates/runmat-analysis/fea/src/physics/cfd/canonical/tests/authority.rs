//! Admission/selection witnesses, not additional continuum accuracy claims.
use super::*;
use runmat_analysis_core::{EntityDimension, ExcitationTarget, RegionTarget};

#[test]
fn body_force_requires_every_original_volume_and_volume_kind() {
    let mut fixture = case(1, ReferenceCase::Duct, false);
    let valid = fixture.source.elements[0].region_id.clone();
    let mut missing = valid.clone();
    missing.source_topology_id = "missing-original-fluid-volume".into();
    let mut target = RegionTarget::volume("exact-volume-selection");
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

    let mut target = RegionTarget::volume("exact-volume-selection");
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
    let boundaries = (0..count)
        .map(|i| boundary::FaceBoundary::PressureOutlet {
            authority_id: format!("component-outlet-{i}"),
            pressure: i as f64,
        })
        .collect::<Vec<_>>();
    let references = boundary::pressure_references(&topology, &boundaries).unwrap();
    assert_eq!(references, (0..count).map(|i| i as f64).collect::<Vec<_>>());
    let mut missing = boundaries;
    missing[count - 1] = boundary::FaceBoundary::Interior;
    assert!(boundary::pressure_references(&topology, &missing).is_err());

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
}

#[test]
fn actual_authority_profile_is_reserved_and_rejects_before_binding() {
    let fixture = case(1, ReferenceCase::Duct, false);
    let base = resources::reserve(&fixture.source, &fixture.topology, fixture.policy).unwrap();
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
    cap.maximum_operator_work = base.operator_work_upper_bound;
    assert!(matches!(
        admit(cap),
        Err(CanonicalFluidError::Resource(
            "source authority operator work"
        ))
    ));
    cap = fixture.policy;
    cap.maximum_operator_work = exact.operator_work_upper_bound;
    cap.maximum_resident_bytes = exact.resident_bytes_upper_bound;
    assert_eq!(admit(cap).unwrap(), exact);
    cap.maximum_authority_items = 1;
    assert!(matches!(admit(cap), Err(CanonicalFluidError::Resource(_))));
}
