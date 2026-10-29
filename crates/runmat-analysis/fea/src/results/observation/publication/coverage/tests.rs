use super::*;
use runmat_analysis_core::{ObservationTarget, SamplingPolicy};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn original() -> (Vec<ObservationRequest>, ResolvedObservationPlan) {
    let projection = crate::topology::AnalysisProjection::derive(
        &canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
    )
    .unwrap();
    let mut step = ProcedureStep::thermal_transient("coverage");
    let field = crate::results::observation::tests::spec("z-field", ObservationTarget::WholeModel);
    let mut history =
        crate::results::observation::tests::spec("a-history", ObservationTarget::WholeModel);
    history.sampling = SamplingPolicy::TimeSeconds(vec![0., 1.]);
    step.observations = vec![
        ObservationRequest::Field(field),
        ObservationRequest::History(history),
    ];
    let plan = crate::results::observation::resolve_observation_plan(
        &step,
        &projection,
        &[crate::results::observation::tests::scalar_definition()],
        ObservationLimits::default(),
    )
    .unwrap();
    (step.observations, plan)
}

#[test]
fn sample_coverage_never_discharges_a_history_and_rejects_duplicate_or_unknown_completion() {
    let (declarations, plan) = original();
    let (coverage, retained, peak) = crate::topology::measure_test_allocations(|| {
        PublicationCoverage::derive(&declarations, &plan)
    });
    let mut coverage = coverage.unwrap();
    assert_eq!(
        retained as u64,
        PublicationCoverage::resident_bytes(2).unwrap()
    );
    assert_eq!(retained, peak);
    assert!(!coverage.sample("a-history").unwrap());
    assert!(!coverage.sample("a-history").unwrap());
    assert!(!coverage.complete());
    assert!(coverage.sample("z-field").unwrap());
    assert!(!coverage.complete());
    assert!(coverage.sample("z-field").is_err());
    assert!(coverage.sample("foreign").is_err());
    assert!(!coverage.history("foreign").unwrap());
    assert!(!coverage.history("z-field").unwrap());
    assert!(coverage.history("a-history").unwrap());
    assert!(!coverage.history("a-history").unwrap());
    assert!(coverage.complete());
}

#[test]
fn original_declarations_must_match_the_plan_before_coverage_is_accepted() {
    let (mut declarations, plan) = original();
    assert!(PublicationCoverage::derive(&declarations[..1], &plan).is_err());
    declarations.swap(0, 1);
    assert!(PublicationCoverage::derive(&declarations, &plan).is_err());
    let mut contents = plan.into_contents();
    contents.observations.swap(0, 1);
    contents.identity = contents.derive_identity().unwrap();
    let plan = ResolvedObservationPlan::try_from(contents).unwrap();
    assert!(PublicationCoverage::derive(&declarations, &plan).is_ok());
    // A plan with a duplicated observation cannot be sealed, so coverage never sees one.
    let mut contents = plan.into_contents();
    contents.observations[1] = contents.observations[0].clone();
    contents.identity = contents.derive_identity().unwrap();
    assert!(ResolvedObservationPlan::try_from(contents).is_err());
    assert!(PublicationCoverage::resident_bytes(usize::MAX).is_err());
}
