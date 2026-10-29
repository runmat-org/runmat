//! The large static path (elastic rigidity proof + AMG-PCG, no global factor) agrees with the
//! proving factorization on a cantilever both can solve.
use super::*;
use crate::{contracts::StaticOperatorProof, solve::preconditioner::SpdPreconditionerKind};

fn solve(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
    preconditioner: SpdPreconditionerKind,
) -> crate::FeaRunResult {
    crate::run_linear_static_with_options(
        model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            preconditioner_kind: preconditioner,
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("the cantilever must solve")
}

pub(super) fn qualify() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = structured_cantilever(3, order);
        let direct = solve(&model, &mesh, SpdPreconditionerKind::Auto);
        let elastic = solve(&model, &mesh, SpdPreconditionerKind::Amg);
        let direct_convergence = direct.structural_static_convergence.as_ref().unwrap();
        let elastic_convergence = elastic.structural_static_convergence.as_ref().unwrap();
        assert_eq!(
            direct_convergence.operator_proof,
            StaticOperatorProof::SparseCholesky
        );
        assert_eq!(
            elastic_convergence.operator_proof,
            StaticOperatorProof::ElasticRigidity
        );
        assert_eq!(elastic.solver_method, crate::SolverMethod::AssembledPcg);
        assert_eq!(elastic.preconditioner, crate::SolverPreconditioner::Amg);
        let relative = elastic_convergence.relative_residual.unwrap();
        assert!(
            elastic_convergence.converged && relative <= 1.0e-8,
            "{relative}"
        );
        // The Lanczos estimate is recorded as an estimate only; the factor path has none.
        assert!(direct_convergence.spectral_estimate.is_none());
        let estimate = elastic_convergence
            .spectral_estimate
            .expect("the PCG path records its Lanczos estimate");
        assert!(
            estimate.lambda_min_estimate > 0.0 && estimate.lanczos_steps > 0,
            "{estimate:?}"
        );
        eprintln!("elastic static path {order:?}: spectral estimate {estimate:?}");
        let field = |run: &crate::FeaRunResult| {
            run.field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .unwrap()
                .as_host_f64()
                .unwrap()
                .to_vec()
        };
        let (a, b) = (field(&direct), field(&elastic));
        let scale = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let difference = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "elastic static path {order:?}: nodes {} iterations {} relative residual {relative:.3e} max |u| {scale:.6e} max |du| {difference:.3e}",
            mesh.topology.nodes.len(),
            elastic_convergence.iterations
        );
        assert!(difference <= 1.0e-6 * scale, "{difference} vs {scale}");
    }
}

/// The planner keeps the proving factorization while its plan fits the effective ceiling,
/// min(authored ceiling, machine memory), and selects the elastic rigidity path once the
/// ceiling falls between the two plans' memory; an admitted path is reproduced as recorded.
pub(super) fn ceiling_selects_the_static_path() {
    use crate::planning::resource::{
        build_static_resource_plan_selecting, recorded_static_solve_path, FeaResourcePlan,
        LocalExecutionPlan, ResourcePlanningError, StaticResourcePreflight, StaticSolvePath,
        StaticSolveSelection,
    };
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1 << 40);
    let mesh = structured_cantilever(8, ElementOrder::Tet10);
    let constraints =
        crate::constraints::CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let fields = crate::post::fields::structural_static_observable_field_definitions().unwrap();
    let observations = crate::results::observation::resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraints,
        &fields,
        crate::results::observation::ObservationLimits::default(),
    )
    .unwrap();
    let plan = |model: &runmat_analysis_core::AnalysisModel,
                selection: StaticSolveSelection|
     -> Result<FeaResourcePlan, ResourcePlanningError> {
        build_static_resource_plan_selecting(
            StaticResourcePreflight {
                compilation: None,
                prepared_inputs: None,
                model,
                step: &model.sequence.steps[0],
                backend: ComputeBackend::Cpu,
                preconditioner: SpdPreconditionerKind::Auto,
                observation_execution:
                    crate::results::observation::ObservationExecutionPolicy::serial(),
                mesh: Some(&mesh),
                constraints: &constraints,
                observations: &observations,
                accelerator_requests: &[],
                local_execution: LocalExecutionPlan::serial(),
            },
            selection,
        )
    };
    let path = |plan: Result<FeaResourcePlan, ResourcePlanningError>| {
        recorded_static_solve_path(&plan.expect("the plan must be admitted"))
    };
    let direct = plan(
        &model,
        StaticSolveSelection::Recorded(StaticSolvePath::ProvingFactorization),
    )
    .unwrap()
    .request
    .memory_bytes;
    let elastic = plan(
        &model,
        StaticSolveSelection::Recorded(StaticSolvePath::ElasticRigidityMultigrid),
    )
    .unwrap()
    .request
    .memory_bytes;
    eprintln!("ceiling selection: direct plan {direct} B, elastic plan {elastic} B");
    assert!(
        elastic < direct,
        "elastic {elastic} B must plan below direct {direct} B"
    );
    let between = elastic + (direct - elastic) / 2;
    let unbounded = StaticSolveSelection::WithinCeiling {
        machine_memory_bytes: None,
    };
    assert_eq!(
        path(plan(&model, unbounded)),
        Some(StaticSolvePath::ProvingFactorization)
    );
    assert_eq!(
        path(plan(
            &model,
            StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: Some(direct),
            }
        )),
        Some(StaticSolvePath::ProvingFactorization)
    );
    assert_eq!(
        path(plan(
            &model,
            StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: Some(between),
            }
        )),
        Some(StaticSolvePath::ElasticRigidityMultigrid)
    );
    let mut authored = model.clone();
    authored.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(between);
    assert_eq!(
        path(plan(&authored, unbounded)),
        Some(StaticSolvePath::ElasticRigidityMultigrid)
    );
    // A recorded elastic path is reproduced on a machine that could have factored.
    assert_eq!(
        path(plan(
            &model,
            StaticSolveSelection::Recorded(StaticSolvePath::ElasticRigidityMultigrid)
        )),
        Some(StaticSolvePath::ElasticRigidityMultigrid)
    );
    // An explicit direct request never takes the elastic path, whatever the ceiling.
    let mut direct_request = authored.clone();
    direct_request.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1 << 40);
    assert_eq!(
        path(build_static_resource_plan_selecting(
            StaticResourcePreflight {
                compilation: None,
                prepared_inputs: None,
                model: &direct_request,
                step: &direct_request.sequence.steps[0],
                backend: ComputeBackend::Cpu,
                preconditioner: SpdPreconditionerKind::Direct,
                observation_execution:
                    crate::results::observation::ObservationExecutionPolicy::serial(),
                mesh: Some(&mesh),
                constraints: &constraints,
                observations: &observations,
                accelerator_requests: &[],
                local_execution: LocalExecutionPlan::serial(),
            },
            StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: Some(between),
            },
        )),
        Some(StaticSolvePath::ProvingFactorization)
    );
}

/// The elastic rigidity fallback answers to the same effective ceiling as the factorization:
/// below both plans the planner refuses with a resource error naming both requirements and the
/// ceiling, instead of returning a plan the worker pool cannot admit.
pub(super) fn fallback_is_admitted_against_the_ceiling() {
    use crate::planning::resource::{
        build_static_resource_plan_selecting, FeaResourcePlan, LocalExecutionPlan,
        ResourcePlanningError, StaticResourcePreflight, StaticSolvePath, StaticSolveSelection,
    };
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1 << 40);
    let mesh = structured_cantilever(8, ElementOrder::Tet10);
    let constraints =
        crate::constraints::CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let fields = crate::post::fields::structural_static_observable_field_definitions().unwrap();
    let observations = crate::results::observation::resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraints,
        &fields,
        crate::results::observation::ObservationLimits::default(),
    )
    .unwrap();
    let plan = |model: &runmat_analysis_core::AnalysisModel,
                selection: StaticSolveSelection|
     -> Result<FeaResourcePlan, ResourcePlanningError> {
        build_static_resource_plan_selecting(
            StaticResourcePreflight {
                compilation: None,
                prepared_inputs: None,
                model,
                step: &model.sequence.steps[0],
                backend: ComputeBackend::Cpu,
                preconditioner: SpdPreconditionerKind::Auto,
                observation_execution:
                    crate::results::observation::ObservationExecutionPolicy::serial(),
                mesh: Some(&mesh),
                constraints: &constraints,
                observations: &observations,
                accelerator_requests: &[],
                local_execution: LocalExecutionPlan::serial(),
            },
            selection,
        )
    };
    let direct = plan(
        &model,
        StaticSolveSelection::Recorded(StaticSolvePath::ProvingFactorization),
    )
    .unwrap()
    .request;
    let elastic = plan(
        &model,
        StaticSolveSelection::Recorded(StaticSolvePath::ElasticRigidityMultigrid),
    )
    .unwrap()
    .request;
    let elastic_need = elastic.memory_bytes.max(elastic.scratch_bytes);
    assert!(elastic_need < direct.memory_bytes);
    // Exactly at the elastic plan's need the fallback is admitted.
    let admitted = plan(
        &model,
        StaticSolveSelection::WithinCeiling {
            machine_memory_bytes: Some(elastic_need),
        },
    )
    .expect("the elastic plan fits its own requirement");
    assert!(admitted.request.memory_bytes <= elastic_need);
    assert!(admitted.request.scratch_bytes <= elastic_need);
    // One byte below, the machine ceiling refuses both paths and names both requirements.
    let ceiling = elastic_need - 1;
    let error = plan(
        &model,
        StaticSolveSelection::WithinCeiling {
            machine_memory_bytes: Some(ceiling),
        },
    )
    .expect_err("no path fits below the elastic plan");
    assert_eq!(
        error,
        ResourcePlanningError::StaticSolveCeiling {
            factorization_resident_bytes: Some(direct.memory_bytes),
            multigrid_resident_bytes: elastic.memory_bytes,
            multigrid_scratch_bytes: Some(elastic.scratch_bytes),
            maximum: ceiling,
        }
    );
    let message = error.to_string();
    for figure in [
        direct.memory_bytes,
        elastic.memory_bytes,
        elastic.scratch_bytes,
        ceiling,
    ] {
        assert!(message.contains(&figure.to_string()), "{message}");
    }
    // The authored ceiling binds the same way: the elastic plan is refused inside planning and
    // its resident requirement is still named.
    let mut authored = model.clone();
    authored.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(elastic.memory_bytes - 1);
    match plan(
        &authored,
        StaticSolveSelection::WithinCeiling {
            machine_memory_bytes: None,
        },
    ) {
        Err(ResourcePlanningError::StaticSolveCeiling {
            factorization_resident_bytes: Some(factor),
            multigrid_resident_bytes,
            maximum,
            ..
        }) => {
            assert!(factor > maximum && multigrid_resident_bytes > maximum);
            assert_eq!(maximum, elastic.memory_bytes - 1);
        }
        other => panic!("expected a static solve ceiling error, got {other:?}"),
    }
}
