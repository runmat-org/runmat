//! Production same-pool composition, not an additional solver-formulation qualification.
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    AnalysisOutputPolicy, StructuralPrestressedFrequencyStudyPreparation, ThermalStudyPreparation,
    ThermalStudyScope,
};
use runmat_analysis_fea::{
    fixtures::{fixture_model, FixtureId},
    planning::resource::LocalExecutionPlan,
    properties::compile_property_map,
    study::{
        plan_structural_study, plan_thermal_study, CompiledAnalysisStudy,
        StructuralNumericalPlanningPolicy, StructuralStudyPlanningPolicy,
        ThermalStudyPlanningPolicy, ThermalStudyProduct,
    },
    ComputeBackend,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::{
    execute_thermal_prestress_study_locally_with_source, execute_thermal_prestress_study_remotely,
    CompletedThermalPrestressStudy, NativeAnalysisHostLimits, NativeExecutionConfig,
    RemotePoolDriver, StructuralSequenceExecutionPolicy, ThermalPrestressStructuralCompletion,
    ThermalPrestressStudyPreparation,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
use std::sync::Arc;

#[path = "../../src/analysis_thermal/tests/prestress/fixture.rs"]
mod fixture;
#[path = "thermal_prestress/observations.rs"]
mod observations;

pub(super) async fn run_local() {
    let directory = tempfile::tempdir().unwrap();
    for branch in 0..4 {
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.executable = std::env::current_exe().unwrap();
        config.max_workers = 2;
        config.worker_capabilities = super::capabilities();
        config.store_root = directory.path().join(format!("coupled-{branch}"));
        let (completed, _) = execute_thermal_prestress_study_locally_with_source(
            config,
            preparation(branch, "analysis-process-conformance", super::revision()),
            policy(),
        )
        .await
        .unwrap();
        assert_completion(&completed, branch);
    }
    observations::run_local().await;
}

pub(super) async fn run_remote(pool: Arc<RemotePoolDriver>, revision: ProgramRevision) {
    for branch in 0..4 {
        let completed = execute_thermal_prestress_study_remotely(
            Arc::clone(&pool),
            preparation(branch, "remote-analysis", revision.clone()),
            policy(),
        )
        .await
        .unwrap();
        assert_completion(&completed, branch);
    }
    observations::run_remote(pool, revision).await;
}

fn policy() -> StructuralSequenceExecutionPolicy {
    StructuralSequenceExecutionPolicy {
        resources: super::analysis_resources(),
        retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    }
}

pub(super) fn preparation(
    branch: usize,
    scope: &str,
    revision: ProgramRevision,
) -> ThermalPrestressStudyPreparation {
    let order = if branch.is_multiple_of(2) {
        ElementOrder::Tet4
    } else {
        ElementOrder::Tet10
    };
    let base = base(order, scope, revision);
    let heat = match branch {
        0 => fixture::setup(base, true),
        1 => fixture::setup(base, false),
        2 => fixture::frequency_setup(base),
        _ => fixture::reduced_setup(base),
    };
    let compiled = heat.product.compiled;
    let workloads = plan_structural_study(
        &compiled,
        StructuralStudyPlanningPolicy {
            backend: ComputeBackend::Cpu,
            numerics: StructuralNumericalPlanningPolicy::production(),
        },
    )
    .unwrap();
    ThermalPrestressStudyPreparation {
        structural: StructuralPrestressedFrequencyStudyPreparation {
            authored_study_digest: compiled.definition_digest().unwrap(),
            frozen_project_digest: None,
            workloads,
            mesh: heat.mesh,
            artifact_access: heat.artifact_access,
            program_revision: heat.program_revision,
            chunk_bytes: heat.output_policy.chunk_bytes(),
            inventory_limits: heat.output_policy.inventory_limits(),
            required_capabilities: Default::default(),
            accelerator_requests: Vec::new(),
            local_execution: LocalExecutionPlan::serial(),
        },
        compiled,
    }
}

fn base(order: ElementOrder, scope: &str, revision: ProgramRevision) -> ThermalStudyPreparation {
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "equilibrium-temperature".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(
                    projection.boundary_faces[0].provenance[0]
                        .analysis_selection_id()
                        .unwrap(),
                ),
                value: PrescribedValue::Constant(Temperature::new(293.15)),
            },
        }));
    let mut heat = ProcedureStep::thermal_steady("heat");
    let mut second = heat.state_outputs[0].clone();
    second.state_id = StateId("second-temperature".into());
    heat.state_outputs.push(second);
    model.sequence = AnalysisSequence::from_steps("process-thermal", vec![heat]);
    let compiled = CompiledAnalysisStudy {
        module_path: "tests/thermal-prestress.fea".into(),
        study_id: "thermal-prestress".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: compile_property_map(&model, Some(&mesh)).unwrap(),
        model,
        solver_mesh_digest: mesh.canonical_digest,
    };
    let workload = plan_thermal_study(
        &compiled,
        ThermalStudyPlanningPolicy {
            backend: ComputeBackend::Cpu,
            element_heat_source: None,
        },
    )
    .unwrap();
    let limits = NativeAnalysisHostLimits::default();
    ThermalStudyPreparation {
        maximum_resident_bytes: None,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        field_source: None,
        scope: ThermalStudyScope::Standalone,
        product: ThermalStudyProduct { compiled, workload },
        mesh,
        frozen_project_digest: None,
        artifact_access: ArtifactAccess {
            authorization_scope: scope.into(),
            encryption_context: Digest::sha256("thermal-prestress-process"),
        },
        program_revision: revision,
        output_policy: AnalysisOutputPolicy::new(limits.chunk_bytes, limits.inventory).unwrap(),
        required_capabilities: Default::default(),
    }
}

fn assert_completion(completed: &CompletedThermalPrestressStudy, branch: usize) {
    assert_eq!(completed.thermal.states.len(), 2);
    let preload = match (&completed.structural, branch) {
        (ThermalPrestressStructuralCompletion::Buckling(value), 0) => {
            assert!(value.buckling.result.modes()[0].load_factor > 0.0);
            &value.preload
        }
        (ThermalPrestressStructuralCompletion::Modal(value), 1) => {
            assert!(value.modal.result.basis().modes()[0].frequency_hz() > 0.0);
            &value.preload
        }
        (ThermalPrestressStructuralCompletion::Frequency(value), 2) => {
            assert_eq!(value.frequency.entries.len(), 1);
            &value.preload
        }
        (ThermalPrestressStructuralCompletion::ModalFrequency(value), 3) => {
            assert_eq!(value.frequency.entries.len(), 1);
            &value.preload
        }
        _ => panic!("wrong coupled consumer"),
    };
    assert_eq!(
        preload
            .result
            .linearization
            .as_ref()
            .unwrap()
            .thermal_source()
            .unwrap()
            .transition()
            .source_state,
        completed.thermal.states[1].state.identity
    );
}
