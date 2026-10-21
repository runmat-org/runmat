//! Authored identity and controls cannot be replaced by portable block plumbing.
use super::*;
use runmat_analysis_fea::study::plan_nonlinear_solid_study_product;

pub(super) use super::support::compiled;

#[test]
fn exact_authored_j2_planning_preserves_schedule_and_all_state_declarations() {
    for quadratic in [false, true] {
        let (mut workload, mesh) = fixture(quadratic);
        workload.model.sequence.steps[0]
            .accuracy
            .criteria
            .push(AccuracyCriterion {
                criterion_id: "authored-newton".into(),
                target: AccuracyTarget::GlobalEquilibrium,
                contribution: ErrorContributionKind::Nonlinear,
                tolerance: AccuracyTolerance::Relative(2e-10),
                evidence: RequiredAccuracyEvidence::ResidualBound,
                mandatory: true,
            });
        let study = compiled(&workload, &mesh);
        let original = study.clone();
        let product =
            plan_nonlinear_solid_study_product(study, &mesh, workload.controls.clone()).unwrap();
        assert_eq!(product.compiled, original);
        assert_eq!(product.workload.model, workload.model);
        let mut expected = workload.controls.clone();
        expected.tolerance = 2e-10;
        assert_eq!(product.workload.controls, expected);
        assert_eq!(
            product.workload.model.sequence.steps[0].state_outputs.len(),
            3
        );
        let request = NonlinearSolidBlockRequest::initial(
            &product.workload,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            9,
        )
        .unwrap();
        assert_eq!(
            (request.block_ordinal, request.first_output_ordinal),
            (0, 1)
        );
        assert!(
            NonlinearSolidBlockRequest::initial(&product.workload, request.topology, 0).is_err()
        );
        assert!(
            NonlinearSolidBlockRequest::initial(&product.workload, request.topology, 13).is_err()
        );
    }
}

#[test]
fn authored_j2_planning_rejects_invalid_controls_and_substituted_topology_or_properties() {
    let (workload, mesh) = fixture(false);
    let study = compiled(&workload, &mesh);
    let mut controls = workload.controls.clone();
    controls.tolerance = f64::NAN;
    assert!(plan_nonlinear_solid_study_product(study.clone(), &mesh, controls).is_err());
    assert!(plan_nonlinear_solid_study_product(
        study.clone(),
        &slab_mesh(2),
        workload.controls.clone()
    )
    .is_err());
    let mut substituted = study;
    substituted.model.materials[0].plastic = None;
    assert!(plan_nonlinear_solid_study_product(substituted, &mesh, workload.controls).is_err());
}
