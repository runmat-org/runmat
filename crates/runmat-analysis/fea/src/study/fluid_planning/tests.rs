use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures, SolverMeshArtifact};

fn study() -> (CompiledAnalysisStudy, SolverMeshArtifact) {
    let mesh = fixtures::slab_mesh(1);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.electromagnetic = None;
    model.cfd = None;
    model.constraints.clear();
    model.boundaries.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.initial_conditions.clear();
    model.interfaces.clear();
    model.connectors.clear();
    model.response_spectra.clear();
    model.random_excitations.clear();
    model.materials[0].fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(0.001),
    });
    model.material_assignments[0].target =
        PropertyRegionTarget::cell(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.sequence =
        AnalysisSequence::from_steps("flow", vec![ProcedureStep::fluid_static("steady")]);
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "signed-inlet".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Constant([1.0, -0.25, 0.125].map(Velocity::new)),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "physical-outlet".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(25.0)),
        },
    });
    let properties = crate::properties::compile_property_map(&model, Some(&mesh)).unwrap();
    (
        CompiledAnalysisStudy {
            assessments: Vec::new(),
            module_path: "tests/fluid.fea".into(),
            study_id: "flow".into(),
            study_inputs: vec![],
            outputs: vec![],
            model,
            properties,
            solver_mesh_digest: mesh.canonical_digest,
        },
        mesh,
    )
}

fn policy() -> FluidStudyPlanningPolicy {
    FluidStudyPlanningPolicy {
        numerical: CanonicalFluidPolicy {
            pressure_velocity_coupling: Default::default(),
            fixed_point_acceleration: None,
            maximum_outer_iterations: 200,
            maximum_nonorthogonal_iterations: 4,
            maximum_linear_iterations: 100,
            gmres_restart: 8,
            momentum_relaxation: 0.5,
            pressure_relaxation: 0.25,
            linear_relative_tolerance: 1e-5,
            physical_relative_tolerance: 1e-8,
            maximum_resident_bytes: Some(1_000_000_000),
            maximum_operator_work: Some(1_000_000_000_000),
            maximum_authority_items: 100,
        },
        geometry: FiniteVolumeProjectionPolicy {
            max_cells: Some(100),
            max_face_incidences: Some(400),
            max_boundary_faces: Some(400),
            max_scratch_bytes: Some(10_000_000),
            max_traversal_operations: Some(1_000_000),
        },
    }
}

#[test]
fn preserves_signed_authority_and_consumes_only_relative_algebraic_criterion() {
    let (mut study, _) = study();
    for criterion in &mut study.model.sequence.steps[0].accuracy.criteria {
        if criterion.contribution == ErrorContributionKind::Algebraic {
            criterion.tolerance = AccuracyTolerance::Relative(2e-9);
        }
    }
    let before = study.clone();
    let workload = plan_fluid_study(&study, policy()).unwrap();
    assert_eq!(workload.model, before.model);
    assert_eq!(study, before);
    assert_eq!(workload.policy.linear_relative_tolerance, 2e-9);
    assert_eq!(workload.policy.physical_relative_tolerance, 2e-9);
}

#[test]
fn rejects_missing_criterion_and_nan_controls() {
    let (mut study, _) = study();
    let mut invalid = policy();
    invalid.numerical.linear_relative_tolerance = f64::NAN;
    assert!(plan_fluid_study(&study, invalid).is_err());
    study.model.sequence.steps[0]
        .accuracy
        .criteria
        .retain(|criterion| criterion.contribution != ErrorContributionKind::Algebraic);
    assert!(matches!(
        plan_fluid_study(&study, policy()),
        Err(FluidStudyPlanningError::MissingRelativeAlgebraicTarget)
    ));
}

#[test]
fn product_authenticates_exact_cell_material_without_injecting_legacy_domain() {
    let (compiled, mesh) = study();
    let before = compiled.clone();
    let product = crate::study::plan_fluid_study_product(compiled, &mesh, policy()).unwrap();
    assert_eq!(*product.compiled(), before);
    assert_eq!(product.workload().model, before.model);
    assert!(product.workload().model.cfd.is_none());
    assert!(product.validate_planned_on(&mesh, policy()).is_ok());
    let mut other = policy();
    other.numerical.momentum_relaxation *= 0.5;
    assert!(product.validate_planned_on(&mesh, other).is_err());
    let mut moved = mesh.clone();
    moved.canonical_digest = runmat_meshing_core::StableDigest([7; 32]);
    assert!(product.validate_planned_on(&moved, policy()).is_err());
    let mut substituted = before;
    substituted.model.materials[0].fluid = None;
    assert!(crate::study::plan_fluid_study_product(substituted, &mesh, policy()).is_err());
}

#[test]
fn closure_controls_follow_only_matching_relative_evidence_and_never_loosen() {
    let (mut study, _) = study();
    let original = study.model.sequence.steps[0].accuracy.criteria[0].clone();
    for evidence in [
        RequiredAccuracyEvidence::ResidualBound,
        RequiredAccuracyEvidence::ConservationBalance,
    ] {
        let mut criterion = original.clone();
        criterion.evidence = evidence;
        criterion.tolerance = AccuracyTolerance::Relative(2e-9);
        study.model.sequence.steps[0].accuracy.criteria = vec![criterion.clone()];
        assert_eq!(
            plan_fluid_study(&study, policy())
                .unwrap()
                .policy
                .physical_relative_tolerance,
            2e-9
        );
        let mut stricter = policy();
        stricter.numerical.physical_relative_tolerance = 1e-10;
        assert_eq!(
            plan_fluid_study(&study, stricter)
                .unwrap()
                .policy
                .physical_relative_tolerance,
            1e-10
        );
        criterion.target = AccuracyTarget::NumericalResult;
        study.model.sequence.steps[0].accuracy.criteria = vec![criterion];
        assert_eq!(
            plan_fluid_study(&study, policy())
                .unwrap()
                .policy
                .physical_relative_tolerance,
            policy().numerical.physical_relative_tolerance
        );
    }
    study.model.sequence.steps[0].accuracy.criteria = vec![original.clone()];
    let mut spatial = original;
    spatial.criterion_id = "spatial".into();
    spatial.target = AccuracyTarget::NumericalResult;
    spatial.contribution = ErrorContributionKind::SpatialDiscretization;
    spatial.evidence = RequiredAccuracyEvidence::ConvergenceStudy;
    spatial.tolerance = AccuracyTolerance::Relative(1e-12);
    study.model.sequence.steps[0]
        .accuracy
        .criteria
        .push(spatial);
    assert_eq!(
        plan_fluid_study(&study, policy())
            .unwrap()
            .policy
            .physical_relative_tolerance,
        policy().numerical.physical_relative_tolerance
    );
}
