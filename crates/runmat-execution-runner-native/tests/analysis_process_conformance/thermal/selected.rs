//! Selected heat remains part of the authored mixed sequence, including its real consumer.
use super::super::publication_chain::PublicationChainDriver;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    import_structural_linear_completion, prepare_structural_sequence_step, CompletedThermalStudy,
    StructuralLinearProgramPreparation,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, StructuralLinearStateInput, StructuralLinearWorkload,
    },
    planning::resource::LocalExecutionPlan,
    results::observation::ObservationExecutionPolicy,
    solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
    ComputeBackend,
};
use runmat_meshing_core::SolverMeshArtifact;

pub(super) fn configure_model(
    model: &mut AnalysisModel,
    mesh: &SolverMeshArtifact,
    heat: ProcedureStep,
) {
    model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(1.0e-5);
    let prefix = ProcedureStep::structural_static("mechanical_context");
    let mut consumer = ProcedureStep::coupled(
        "thermal_expansion",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Static,
    );
    consumer.state_inputs.push(StateInput {
        input_id: "solved-temperature".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: heat.step_id.clone(),
            state_id: heat.state_outputs[0].state_id.clone(),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    let transition = StateTransition {
        transition_id: TransitionId("cold-temperature-reuse".into()),
        producer_step: heat.step_id.clone(),
        consumer_step: consumer.step_id.clone(),
        state_id: heat.state_outputs[0].state_id.clone(),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    };
    let mut unselected_heat = heat.clone();
    unselected_heat.step_id = ProcedureStepId("unselected_heat".into());
    for output in &mut unselected_heat.state_outputs {
        output.state_id = StateId(format!("unselected-{}", output.state_id.0));
    }
    model.sequence = AnalysisSequence::from_steps(
        "mixed-thermal-context",
        vec![prefix, heat, consumer, unselected_heat],
    );
    // List position is not an execution dependency. Both thermal producers are
    // independently executable; only the real mechanical consumer depends on heat.
    model
        .sequence
        .dependencies
        .retain(|dependency| dependency.successor.0 == "thermal_expansion");
    model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    model.sequence.transitions.push(transition);
    // The unchanged isothermal equilibrium is a zero-expansion patch. Full restraint
    // makes the downstream mechanical solve determinate without introducing new loads.
    for node in &mesh.topology.nodes {
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("restrained-node-{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations: [
                        StructuralTranslationDof::Ux,
                        StructuralTranslationDof::Uy,
                        StructuralTranslationDof::Uz,
                    ]
                    .into_iter()
                    .map(|dof| TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    })
                    .collect(),
                    rotations: Vec::new(),
                }),
            }));
    }
}

pub(super) async fn execute_downstream<D: PublicationChainDriver>(
    driver: &mut D,
    recovered: &CompletedThermalStudy,
) {
    // This is a new downstream solve AFTER cold thermal replay has finished. The
    // source objects here were recovered exclusively from the closed thermal root.
    let preparation = recovered.preparation();
    let limits = preparation.output_policy.inventory_limits();
    let source = &recovered.states[0];
    let source_bytes = source.state.canonical_bytes().unwrap();
    let workload = StructuralLinearWorkload {
        schema_version: 2,
        model: recovered.compiled.model.clone(),
        step_id: ProcedureStepId("thermal_expansion".into()),
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution: ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let prepared = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
        authored_study_digest: recovered.compiled.definition_digest().unwrap(),
        frozen_project_digest: preparation.frozen_project_digest,
        program_revision: preparation.program_revision,
        artifact_access: preparation.artifact_access,
        workload,
        mesh: Some(preparation.mesh),
        state_input: Some(source.objects.clone()),
        chunk_bytes: preparation.output_policy.chunk_bytes(),
        inventory_limits: limits,
        required_capabilities: Default::default(),
        accelerator_requests: Vec::new(),
        local_execution: LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    let invocation = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
        .unwrap();
    let admitted = StructuralLinearWorkload::from_canonical_bytes(&invocation.payload()).unwrap();
    assert_eq!(admitted.model, recovered.compiled.model);
    let Some(StructuralLinearStateInput::ThermalPrestress(transition)) = &admitted.state_input
    else {
        panic!("selected thermal consumer lost its typed source");
    };
    assert_eq!(transition.producer, source.state.producer);
    assert_eq!(transition.source_state, source.state.identity);
    assert_eq!(transition.consumer, prepared.host.workload.step);
    let response = driver
        .execute(
            &prepared.host,
            &prepared.program,
            Some(&prepared.resource_plan),
        )
        .await;
    let imported =
        import_structural_linear_completion(driver.store(), response.response, &prepared, limits)
            .unwrap();
    let retained = imported
        .result
        .linearization
        .as_ref()
        .unwrap()
        .thermal_source()
        .unwrap();
    retained.validate_model(&recovered.compiled.model).unwrap();
    assert_eq!(retained.transition(), transition);
    assert_eq!(retained.structural_result(), imported.identity);
    assert_eq!(imported.numerical_receipt.source_result, imported.identity);
    assert!(imported
        .result
        .run
        .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .as_host_f64()
        .unwrap()
        .iter()
        .all(|value| value.abs() < 1.0e-14));
    assert_eq!(source.state.canonical_bytes().unwrap(), source_bytes);
}
