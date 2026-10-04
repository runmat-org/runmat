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
    model.material_assignments[0].target = PropertyRegionTarget::cell("slab");
    model.sequence =
        AnalysisSequence::from_steps("flow", vec![ProcedureStep::fluid_static("steady")]);
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "signed-inlet".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: RegionTarget::boundary("slab_left"),
                value: PrescribedValue::Constant([1.0, -0.25, 0.125].map(Velocity::new)),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "physical-outlet".into(),
        target: RegionTarget::boundary("slab_right"),
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
            formulation_requirement: FormulationRequirement::Experimental,
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
        formulation_requirement: FormulationRequirement::Experimental,
        numerical: CanonicalFluidPolicy {
            fixed_point_acceleration: None,
            maximum_outer_iterations: 200,
            maximum_nonorthogonal_iterations: 4,
            maximum_linear_iterations: 100,
            gmres_restart: 8,
            momentum_relaxation: 0.5,
            pressure_relaxation: 0.25,
            linear_relative_tolerance: 1e-5,
            physical_relative_tolerance: 1e-8,
            maximum_resident_bytes: 1_000_000_000,
            maximum_operator_work: 1_000_000_000_000,
            maximum_authority_items: 100,
        },
        geometry: FiniteVolumeProjectionPolicy {
            max_cells: 100,
            max_face_incidences: 400,
            max_boundary_faces: 400,
            max_scratch_bytes: 10_000_000,
            max_traversal_operations: 1_000_000,
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
    assert_eq!(
        workload.policy.physical_relative_tolerance,
        policy().numerical.physical_relative_tolerance
    );
}

#[test]
fn rejects_mismatched_requirement_missing_criterion_and_nan_controls() {
    let (mut study, _) = study();
    let mut wrong = policy();
    wrong.formulation_requirement = FormulationRequirement::Production;
    assert!(matches!(
        plan_fluid_study(&study, wrong),
        Err(FluidStudyPlanningError::FormulationRequirementMismatch)
    ));
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
    assert_eq!(product.compiled, before);
    assert_eq!(product.workload.model, before.model);
    assert!(product.workload.model.cfd.is_none());
    let mut substituted = before;
    substituted.model.materials[0].fluid = None;
    assert!(crate::study::plan_fluid_study_product(substituted, &mesh, policy()).is_err());
}
