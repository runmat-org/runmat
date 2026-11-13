use super::*;
use crate::results::observation::tests::{projection, scalar_definition, spec, thermal_step};
use runmat_analysis_core::{ObservationCostPolicy, Position};
use runmat_meshing_core::ElementOrder;

#[test]
fn streamed_verification_matches_full_resolution_for_both_orders_and_cost_policies() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let definition = scalar_definition();
        let mut explicit = spec("bounded", ObservationTarget::WholeModel);
        let mut automatic = explicit.clone();
        automatic.observation_id.0 = "automatic".into();
        automatic.cost = ObservationCostPolicy::Automatic;
        explicit.observation_id.0 = "explicit".into();
        let step = thermal_step(vec![
            ObservationRequest::Field(explicit),
            ObservationRequest::Field(automatic),
        ]);
        let plan = resolve_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        verify_observation_plan(&step, &projection, std::slice::from_ref(&definition), &plan)
            .unwrap();
        let mut empty = plan.clone().into_contents();
        empty.observations.clear();
        empty.identity = empty.derive_identity().unwrap();
        let empty = ResolvedObservationPlan::try_from(empty).unwrap();
        assert!(verify_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            &empty
        )
        .is_err());
        let mut reordered = plan.clone().into_contents();
        reordered.observations.reverse();
        reordered.identity = reordered.derive_identity().unwrap();
        let reordered = ResolvedObservationPlan::try_from(reordered).unwrap();
        assert!(verify_observation_plan(&step, &projection, &[definition], &reordered).is_err());
    }
}

#[test]
fn rehashed_smaller_target_cannot_make_verification_expand_past_its_input() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let step = thermal_step(vec![ObservationRequest::Field(spec(
        "all",
        ObservationTarget::WholeModel,
    ))]);
    let mut plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap()
    .into_contents();
    let ResolvedObservationTarget::Nodes {
        indices,
        identities,
    } = &mut plan.observations[0].target
    else {
        panic!("nodes")
    };
    indices.truncate(1);
    identities.truncate(1);
    plan.identity = plan.derive_identity().unwrap();
    let plan = ResolvedObservationPlan::try_from(plan).unwrap();
    assert!(matches!(
        verify_observation_plan(&step, &projection, &[definition], &plan),
        Err(ObservationError::Limit("selected topology entities"))
    ));
}

#[test]
fn rehashed_same_size_target_substitution_and_cost_changes_are_rejected() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let step = thermal_step(vec![ObservationRequest::Field(spec(
        "all",
        ObservationTarget::WholeModel,
    ))]);
    let original = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let mut plan = original.clone().into_contents();
    let ResolvedObservationTarget::Nodes { identities, .. } = &mut plan.observations[0].target
    else {
        panic!("nodes")
    };
    identities[0] = StableDigest::from_bytes([77; 32]);
    plan.identity = plan.derive_identity().unwrap();
    let plan = ResolvedObservationPlan::try_from(plan).unwrap();
    assert!(
        verify_observation_plan(&step, &projection, std::slice::from_ref(&definition), &plan)
            .is_err()
    );
    let mut plan = original.into_contents();
    plan.observations[0].resolved_cost.operation_count += 1;
    plan.identity = plan.derive_identity().unwrap();
    let plan = ResolvedObservationPlan::try_from(plan).unwrap();
    assert!(verify_observation_plan(&step, &projection, &[definition], &plan).is_err());
}

#[test]
fn over_limit_paths_fail_before_topology_or_point_location() {
    let projection = projection(ElementOrder::Tet4);
    let definition = scalar_definition();
    let mut path = spec(
        "path",
        ObservationTarget::Path {
            positions: vec![Position::new([1.0e9; 3]); 3],
        },
    );
    path.output_location = FieldLocation::Point;
    path.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
    let step = thermal_step(vec![ObservationRequest::Field(path)]);
    let limits = ObservationLimits {
        maximum_selected_entities: 2,
        ..ObservationLimits::default()
    };
    assert!(matches!(
        resolve_observation_plan(&step, &projection, &[definition], limits),
        Err(ObservationError::Limit("selected topology entities"))
    ));
    let model = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    );
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
    // No solid topology or temperature field exists here: cardinality rejects before either is consulted.
    assert!(matches!(
        resolve_structural_observation_plan(&step, &model, None, &constraints, &[], limits),
        Err(ObservationError::Limit("selected topology entities"))
    ));
}

#[test]
fn exact_path_limit_retains_repeated_positions_and_canonical_ordinals() {
    let projection = projection(ElementOrder::Tet10);
    let definition = scalar_definition();
    let position = projection.elements[0].centroid_m;
    let mut path = spec(
        "path",
        ObservationTarget::Path {
            positions: vec![Position::new(position); 3],
        },
    );
    path.output_location = FieldLocation::Point;
    path.recovery = RecoveryPolicy::ShapeFunctionInterpolation;
    let step = thermal_step(vec![ObservationRequest::Field(path)]);
    let limits = ObservationLimits {
        maximum_selected_entities: 3,
        ..ObservationLimits::default()
    };
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        limits,
    )
    .unwrap();
    verify_observation_plan(&step, &projection, &[definition], &plan).unwrap();
    let ResolvedObservationTarget::Points { samples } = &plan.observations[0].target else {
        panic!("points")
    };
    assert_eq!(
        samples
            .iter()
            .map(|point| point.ordinal)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_ne!(samples[0].identity, samples[1].identity);
}

#[test]
fn original_verification_inventory_limit_remains_typed_at_source_boundary() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let definition = scalar_definition();
        let mut step = thermal_step(vec![ObservationRequest::Field(spec(
            "budget",
            ObservationTarget::WholeModel,
        ))]);
        let plan = resolve_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        let topology =
            super::super::super::topology::resources::solid_observation_topology_resident_bytes(
                &projection,
            )
            .unwrap();
        let catalog = super::super::field_catalog::declaration_inventory_resident_bytes(1).unwrap();
        step.accuracy.resources.maximum_resident_bytes = Some(topology + catalog - 1);
        let error =
            verify_observation_plan(&step, &projection, std::slice::from_ref(&definition), &plan)
                .unwrap_err();
        assert!(matches!(
            super::super::super::source::observation_error(error),
            crate::execution::AnalysisExecutionError::Limit(
                "recovered-field declaration inventory"
            )
        ));
    }
}
