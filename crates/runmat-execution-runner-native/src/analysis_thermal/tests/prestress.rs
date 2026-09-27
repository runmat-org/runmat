//! Real, nonzero thermal preload through replayed publication authority and shared consumers.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::*;
use runmat_analysis_fea::{
    execution::AnalysisTaskIdentity,
    planning::resource::LocalExecutionPlan,
    study::{
        plan_structural_study, plan_thermal_step, StructuralNumericalPlanningPolicy,
        StructuralStudyPlanningPolicy,
    },
};

fn setup(order: ElementOrder, buckling: bool) -> ThermalStudyPreparation {
    let mut setup = preparation(order, false);
    let model = &mut setup.product.compiled.model;
    model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .expansion_coefficient_per_k = 1.0e-5;
    for constraint in &mut model.constraints {
        if let AnalysisConstraint::Thermal(ThermalConstraint {
            kind: ThermalConstraintKind::Prescribed { value, .. },
            ..
        }) = constraint
        {
            *value = PrescribedValue::Constant(Temperature::new(303.15));
        }
    }
    // Restrained x expansion creates uniform compression; y/z remain physically free.
    for node in &setup.mesh.topology.nodes {
        let axes: &[usize] = if node.coordinates_m == [0.0, 0.0, 0.0] {
            &[0, 1, 2]
        } else if node.coordinates_m == [0.0, 1.0, 0.0] {
            &[0, 2]
        } else {
            &[0]
        };
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("axial-{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations: axes
                        .iter()
                        .map(|&axis| TranslationDofValue {
                            dof: [
                                StructuralTranslationDof::Ux,
                                StructuralTranslationDof::Uy,
                                StructuralTranslationDof::Uz,
                            ][axis],
                            value: PrescribedValue::Constant(Displacement::new(0.0)),
                        })
                        .collect(),
                    rotations: Vec::new(),
                }),
            }));
    }
    let heat = model.sequence.steps[0].clone();
    let mut preload = ProcedureStep::coupled(
        "preload",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Static,
    );
    preload.state_outputs = ProcedureStep::structural_preload("template").state_outputs;
    preload.state_inputs.push(StateInput {
        input_id: "named-temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: heat.step_id.clone(),
            state_id: heat.state_outputs[1].state_id.clone(),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    let mut consumer = if buckling {
        ProcedureStep::structural_buckling("consumer", "preload", "geometric_stiffness")
    } else {
        ProcedureStep::structural_prestressed_modal("consumer", "preload", "geometric_stiffness")
    };
    consumer.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
    model.sequence = AnalysisSequence::from_steps(
        "thermal-prestress-consumer",
        vec![heat.clone(), preload.clone(), consumer.clone()],
    );
    for dependency in &mut model.sequence.dependencies {
        dependency.reason = SequenceDependencyReason::StateTransfer;
    }
    model.sequence.transitions = vec![
        transition(
            "heat-preload",
            &heat,
            &preload,
            heat.state_outputs[1].state_id.clone(),
        ),
        transition(
            "preload-consumer",
            &preload,
            &consumer,
            StateId("geometric_stiffness".into()),
        ),
    ];
    setup.product.compiled.properties = compile_property_map(model, Some(&setup.mesh)).unwrap();
    setup.product.workload =
        plan_thermal_step(&setup.product.compiled, "heat", planning_policy()).unwrap();
    setup.scope = ThermalStudyScope::SelectedStep;
    setup
}

fn transition(
    id: &str,
    producer: &ProcedureStep,
    consumer: &ProcedureStep,
    state_id: StateId,
) -> StateTransition {
    StateTransition {
        transition_id: TransitionId(id.into()),
        producer_step: producer.step_id.clone(),
        consumer_step: consumer.step_id.clone(),
        state_id,
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }
}

fn solve_heat(
    store: &mut FilesystemObjectStore,
    setup: ThermalStudyPreparation,
) -> CompletedThermalStudy {
    let mut study = ThermalStudyCoordinator::new(setup).unwrap();
    let task = study.prepare_solve().unwrap();
    let (response, accounting) = execute(store, &task.host, &task.program, &task.resource_plan, 0);
    study.complete_solve(store, response, accounting).unwrap();
    let task = study.prepare_accuracy().unwrap();
    let (response, accounting) = execute(store, &task.host, &task.program, &task.resource_plan, 1);
    study
        .complete_accuracy(store, response, accounting)
        .unwrap();
    for ordinal in 0..2 {
        let task = study.prepare_next_state().unwrap();
        let (response, accounting) = execute(
            store,
            &task.host,
            &task.program,
            &task.resource_plan,
            ordinal + 2,
        );
        study
            .complete_next_state(store, response, accounting)
            .unwrap();
    }
    study.freeze_baseline().unwrap();
    let task = study.prepare_publication().unwrap();
    let (response, accounting) = execute(store, &task.host, &task.program, &task.resource_plan, 4);
    study
        .complete_publication(store, response, accounting)
        .unwrap();
    study.into_completed().unwrap()
}

fn seal(
    store: &FilesystemObjectStore,
    completed: &CompletedThermalStudy,
) -> ThermalPrestressSource {
    ThermalPrestressSource::import(
        store,
        completed.preparation(),
        completed.result_closures().to_vec(),
    )
    .unwrap()
}

macro_rules! structural_preparation {
    ($kind:ident, $completed:expr) => {{
        let heat = $completed.preparation();
        let limits = NativeAnalysisHostLimits::default();
        $kind {
            authored_study_digest: heat.product.compiled.definition_digest().unwrap(),
            frozen_project_digest: heat.frozen_project_digest,
            workloads: plan_structural_study(
                &heat.product.compiled,
                StructuralStudyPlanningPolicy {
                    backend: ComputeBackend::Cpu,
                    formulation_requirement: FormulationRequirement::Experimental,
                    numerics: StructuralNumericalPlanningPolicy::production(),
                },
            )
            .unwrap(),
            mesh: heat.mesh,
            artifact_access: heat.artifact_access,
            program_revision: heat.program_revision,
            chunk_bytes: limits.chunk_bytes,
            inventory_limits: limits.inventory,
            required_capabilities: Default::default(),
            accelerator_requests: Vec::new(),
            local_execution: LocalExecutionPlan::serial(),
        }
    }};
}

macro_rules! preload {
    ($study:expr, $store:expr, $dependency:expr) => {{
        let task = $study.prepare_preload().unwrap();
        let (response, _) = execute($store, &task.host, &task.program, &task.resource_plan, 5);
        $study.complete_preload($store, response, &task).unwrap();
        let result = $study.preload_result().unwrap();
        let heat = $study.thermal_source().unwrap().completed();
        let retained = result
            .result
            .linearization
            .as_ref()
            .unwrap()
            .thermal_source()
            .unwrap();
        assert_eq!(
            retained.transition().source_state,
            heat.states[1].state.identity
        );
        assert_eq!(retained.structural_result(), result.identity);
        retained.validate_model(&heat.compiled.model).unwrap();
        let stress = result
            .result
            .run
            .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_STRESS)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let expected = -heat.compiled.model.materials[0]
            .require_mechanical()
            .unwrap()
            .youngs_modulus_pa
            * 1.0e-4;
        for tensor in stress.chunks_exact(6) {
            assert!((tensor[0] - expected).abs() < expected.abs() * 1.0e-7);
            assert!(tensor[1..]
                .iter()
                .all(|value| value.abs() < expected.abs() * 1.0e-7));
        }
        assert_eq!(
            $study.preload_task_identity(),
            Some(
                AnalysisTaskIdentity::new(task.host.workload.identity, vec![$dependency]).unwrap()
            )
        );
        let task = $study.prepare_state().unwrap();
        let (response, _) = execute($store, &task.host, &task.program, &task.resource_plan, 6);
        $study.complete_state($store, response, &task).unwrap();
        assert_eq!(
            $study.semantic_state().unwrap().state.source_result,
            $study.preload_result().unwrap().identity
        );
    }};
}

mod frequency;

#[test]
fn replayed_named_thermal_publication_drives_real_modal_and_buckling_workers() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for buckling in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let mut store = FilesystemObjectStore::open(
                directory.path(),
                NativeAnalysisHostLimits::default()
                    .inventory
                    .max_object_bytes,
            )
            .unwrap();
            let completed = solve_heat(&mut store, setup(order, buckling));
            let source = seal(&store, &completed);
            assert_eq!(source.completed().result, completed.result);
            let dependency = source
                .state_publication_dependency(&StateId("second-temperature".into()))
                .unwrap();
            assert_ne!(
                dependency,
                source
                    .state_publication_dependency(&completed.states[0].state.state_id)
                    .unwrap()
            );
            assert!(source
                .state_publication_dependency(&StateId("absent".into()))
                .is_err());
            let mut forged = completed.result_closures().to_vec();
            forged
                .last_mut()
                .unwrap()
                .accounting
                .resources
                .artifact_bytes += 1;
            assert!(
                ThermalPrestressSource::import(&store, completed.preparation(), forged).is_err()
            );
            let mut foreign = completed.preparation();
            let mut swapped = completed.result_closures().to_vec();
            swapped[2].response = swapped[3].response.clone();
            assert!(
                ThermalPrestressSource::import(&store, completed.preparation(), swapped).is_err()
            );
            let mut forged_receipt = completed.result_closures().to_vec();
            forged_receipt.last_mut().unwrap().response =
                completed.result_closures()[0].response.clone();
            assert!(ThermalPrestressSource::import(
                &store,
                completed.preparation(),
                forged_receipt
            )
            .is_err());
            foreign.product.compiled.model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .expansion_coefficient_per_k *= 2.0;
            foreign.product.compiled.properties =
                compile_property_map(&foreign.product.compiled.model, Some(&foreign.mesh)).unwrap();
            foreign.product.workload =
                plan_thermal_step(&foreign.product.compiled, "heat", planning_policy()).unwrap();
            assert!(ThermalPrestressSource::import(
                &store,
                foreign,
                completed.result_closures().to_vec()
            )
            .is_err());
            if buckling {
                assert!(
                    StructuralBucklingStudyCoordinator::new(structural_preparation!(
                        StructuralBucklingStudyPreparation,
                        completed
                    ))
                    .is_err()
                );
                let mut study = StructuralBucklingStudyCoordinator::new_with_thermal_source(
                    structural_preparation!(StructuralBucklingStudyPreparation, completed),
                    source,
                )
                .unwrap();
                preload!(study, &mut store, dependency);
                let task = study.prepare_buckling().unwrap();
                let (response, _) = execute(
                    &mut store,
                    &task.host,
                    &task.program,
                    &task.resource_plan,
                    7,
                );
                study.complete_buckling(&store, response, &task).unwrap();
                assert!(study.buckling_result().unwrap().result.modes()[0].load_factor > 0.0);
            } else {
                let mut wrong_context =
                    structural_preparation!(StructuralPrestressedModalStudyPreparation, completed);
                wrong_context.authored_study_digest = Digest::sha256("another-authored-study");
                assert!(
                    StructuralPrestressedModalStudyCoordinator::new_with_thermal_source(
                        wrong_context,
                        seal(&store, &completed)
                    )
                    .is_err()
                );
                let mut wrong_authority =
                    structural_preparation!(StructuralPrestressedModalStudyPreparation, completed);
                wrong_authority
                    .artifact_access
                    .authorization_scope
                    .push_str("-foreign");
                assert!(
                    StructuralPrestressedModalStudyCoordinator::new_with_thermal_source(
                        wrong_authority,
                        seal(&store, &completed)
                    )
                    .is_err()
                );
                assert!(
                    StructuralPrestressedModalStudyCoordinator::new(structural_preparation!(
                        StructuralPrestressedModalStudyPreparation,
                        completed
                    ))
                    .is_err()
                );
                let mut study =
                    StructuralPrestressedModalStudyCoordinator::new_with_thermal_source(
                        structural_preparation!(
                            StructuralPrestressedModalStudyPreparation,
                            completed
                        ),
                        source,
                    )
                    .unwrap();
                preload!(study, &mut store, dependency);
                let task = study.prepare_modal().unwrap();
                let (response, _) = execute(
                    &mut store,
                    &task.host,
                    &task.program,
                    &task.resource_plan,
                    7,
                );
                study.complete_modal(&store, response, &task).unwrap();
                assert!(
                    study.modal_result().unwrap().result.basis().modes()[0].frequency_hz() > 0.0
                );
                let task = study.prepare_basis_state().unwrap();
                let (response, _) = execute(
                    &mut store,
                    &task.host,
                    &task.program,
                    &task.resource_plan,
                    8,
                );
                study.complete_basis_state(&store, response, &task).unwrap();
                assert_eq!(
                    study.basis_state().unwrap().state.source_result,
                    study.modal_result().unwrap().identity
                );
            }
        }
    }
}
