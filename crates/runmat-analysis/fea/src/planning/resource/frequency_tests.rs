use runmat_analysis_core::{
    FormulationRequirement, IndependentDomain, ProcedureStep, ProcedureStepId,
    ReferenceStatePolicy, SequenceDependency, SequenceDependencyReason, StateId, StateTransition,
    StateTransitionMethod, TopologyTransitionPolicy, TransitionId,
};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::Digest;

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        FieldRecoveryRequest, FieldRecoverySource, SequenceIdentity, SolverResultIdentity,
        StepIdentity, StructuralFrequencyEntryRequest, StructuralFrequencySolverPlan,
        StructuralFrequencyWorkload, StructuralFrequencyWorkloadInput,
    },
    fixtures::{fixture_model, FixtureId},
    frequency::FrequencyResponsePlan,
    pipeline::structural_frequency::{
        execute_structural_frequency_entry, prepare_structural_frequency,
        recover_structural_frequency_entry_fields,
    },
    results::field::{canonical_field_partitions, StructuralRecoveredFieldTopology},
    results::observation::ObservationExecutionPolicy,
    solve::dynamics::DirectHarmonicPolicy,
    ComputeBackend, ModalSolveOptions, ModalTruncationPolicy,
};

use super::{
    build_structural_frequency_entry_resource_plan,
    build_structural_frequency_recovery_resource_plan,
    structural_frequency_entry_payload_bytes_upper_bound,
    structural_frequency_recovery_payload_bytes_upper_bound, ByteCount, FeaResourcePhase,
    PreparedInputResourceFacts, PreparedOutputResourceFacts, PreparedOutputSetResourceFacts,
    PreparedPhaseResourceFacts, SolveResourcePolicy,
};

fn input_facts(encoded_bytes: u64) -> PreparedInputResourceFacts {
    PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(encoded_bytes),
        transport_closure_bytes: ByteCount(encoded_bytes + 128 * 1024),
        decoded_resident_bytes_admission_budget: ByteCount(encoded_bytes * 128 + 128 * 1024),
    }
}

fn entry_output(request: &StructuralFrequencyEntryRequest) -> PreparedOutputResourceFacts {
    let payload = structural_frequency_entry_payload_bytes_upper_bound(request).unwrap();
    PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(payload),
        transport_closure_bytes_upper_bound: ByteCount(payload + 128 * 1024),
        construction_resident_bytes_admission_budget: ByteCount(payload * 128 + 128 * 1024),
    }
}

fn recovery_output(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralFrequencyWorkload,
) -> PreparedOutputResourceFacts {
    let payload =
        structural_frequency_recovery_payload_bytes_upper_bound(mesh, constraints, workload)
            .unwrap();
    PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(payload),
        transport_closure_bytes_upper_bound: ByteCount(payload + 128 * 1024),
        construction_resident_bytes_admission_budget: ByteCount(payload * 128 + 128 * 1024),
    }
}

fn output_set() -> PreparedOutputSetResourceFacts {
    PreparedOutputSetResourceFacts {
        encoded_bytes_upper_bound: ByteCount(8 * 1024 * 1024),
    }
}

fn prepared(
    inputs: PreparedInputResourceFacts,
    output: PreparedOutputResourceFacts,
) -> PreparedPhaseResourceFacts {
    PreparedPhaseResourceFacts {
        inputs,
        output,
        output_set: output_set(),
    }
}

fn direct_case(
    restart: usize,
) -> (
    StructuralFrequencyWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalStructuralConstraintPlan,
    StructuralFrequencyEntryRequest,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_harmonic("frequency");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0]);
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: DirectHarmonicPolicy {
                restart,
                ..DirectHarmonicPolicy::default()
            },
        },
    })
    .unwrap();
    let prepared = prepare_structural_frequency(&workload, &mesh).unwrap();
    let entry = workload.plan().entries().unwrap()[0];
    let request = StructuralFrequencyEntryRequest::new(
        &workload,
        prepared.study,
        prepared.step,
        prepared.execution_plan,
        prepared.scope,
        entry,
    )
    .unwrap();
    (workload, mesh, constraints, request)
}

fn modal_case() -> (
    StructuralFrequencyWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalStructuralConstraintPlan,
    StructuralFrequencyEntryRequest,
    CanonicalSemanticState,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut frequency_step =
        ProcedureStep::structural_modal_harmonic("frequency", "modes", "modal_basis");
    frequency_step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0]);
    model.sequence.steps = vec![modal_step, frequency_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("frequency".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-frequency".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("frequency".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
    let modal = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(3),
            solver_mesh: Some(mesh.clone()),
            formulation_requirement: FormulationRequirement::Experimental,
            ..ModalSolveOptions::default()
        },
    )
    .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal.basis.identity().digest()).unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal.basis),
        },
    })
    .unwrap();
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::ModalReduced {
            basis_state: state.identity,
        },
    })
    .unwrap();
    let prepared = prepare_structural_frequency(&workload, &mesh).unwrap();
    let request = StructuralFrequencyEntryRequest::new(
        &workload,
        prepared.study,
        prepared.step,
        prepared.execution_plan,
        prepared.scope,
        workload.plan().entries().unwrap()[0],
    )
    .unwrap();
    (workload, mesh, constraints, request, state)
}

#[test]
fn direct_frequency_entry_plan_binds_request_and_concrete_gmres_growth() {
    let (small_workload, mesh, constraints, small_request) = direct_case(2);
    let small = build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &small_workload,
        &small_request,
        None,
        &[],
        prepared(input_facts(1024), entry_output(&small_request)),
    )
    .unwrap();
    assert_eq!(
        small.input.phase,
        FeaResourcePhase::StructuralFrequencyEntry(small_request.identity())
    );
    let SolveResourcePolicy::RestartedGmres {
        block_dimension,
        restart,
        workspace_bytes_upper_bound,
        ..
    } = small.input.structural_operator().unwrap().solve
    else {
        panic!("direct harmonic work must publish its GMRES workspace")
    };
    assert_eq!(restart.0, 2);
    assert!(block_dimension.0 > 0);
    assert!(workspace_bytes_upper_bound.0 > 0);
    assert!(small.request.accelerators.is_empty());
    let result =
        execute_structural_frequency_entry(&small_workload, &mesh, &small_request, None).unwrap();
    let encoded_result_bytes = u64::try_from(result.canonical_bytes().unwrap().len()).unwrap();
    let output_ceiling =
        structural_frequency_entry_payload_bytes_upper_bound(&small_request).unwrap();
    assert!(encoded_result_bytes <= output_ceiling);
    assert!(
        output_ceiling > u64::try_from(small_request.scope().physical_dof_count()).unwrap() * 112
    );
    let mut understated_output = entry_output(&small_request);
    understated_output.encoded_payload_bytes_upper_bound.0 -= 1;
    assert!(build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &small_workload,
        &small_request,
        None,
        &[],
        prepared(input_facts(1024), understated_output),
    )
    .is_err());

    let (large_workload, large_mesh, large_constraints, large_request) = direct_case(4);
    let large = build_structural_frequency_entry_resource_plan(
        &large_mesh,
        &large_constraints,
        &large_workload,
        &large_request,
        None,
        &[],
        prepared(input_facts(1024), entry_output(&large_request)),
    )
    .unwrap();
    assert!(
        large.breakdown.workspace_bytes.0 > small.breakdown.workspace_bytes.0,
        "larger Krylov bases must increase the admitted peak"
    );
}

#[test]
fn decoded_frequency_inputs_remain_resident_during_the_solve_peak() {
    let (workload, mesh, constraints, request) = direct_case(2);
    let payload = structural_frequency_entry_payload_bytes_upper_bound(&request).unwrap();
    let output = PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(payload),
        transport_closure_bytes_upper_bound: ByteCount(payload),
        construction_resident_bytes_admission_budget: ByteCount(payload),
    };
    let inputs = input_facts(1_024);
    let baseline = build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        None,
        &[],
        prepared(inputs.clone(), output.clone()),
    )
    .unwrap();
    let mut larger_inputs = inputs;
    larger_inputs.decoded_resident_bytes_admission_budget.0 += 1;
    let larger = build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        None,
        &[],
        prepared(larger_inputs, output),
    )
    .unwrap();
    assert_eq!(
        larger.request.memory_bytes,
        baseline.request.memory_bytes + 1
    );
}

#[test]
fn frequency_entry_preflight_rejects_accelerator_and_request_substitution() {
    let (workload, mesh, constraints, request) = direct_case(2);
    assert!(build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        None,
        &[AcceleratorRequest {
            class: "gpu".into(),
            count: 1,
            memory_bytes_each: 1024,
        }],
        prepared(input_facts(1024), entry_output(&request)),
    )
    .is_err());

    let (other_workload, _, _, other_request) = direct_case(3);
    assert!(build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &other_workload,
        &request,
        None,
        &[],
        prepared(input_facts(1024), entry_output(&request)),
    )
    .is_err());
    assert_ne!(request.identity(), other_request.identity());
}

#[test]
fn modal_frequency_entry_plan_accounts_for_the_exact_basis_and_dense_solver() {
    let (workload, mesh, constraints, request, state) = modal_case();
    let plan = build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        Some(&state),
        &[],
        prepared(input_facts(1024), entry_output(&request)),
    )
    .unwrap();
    let mode_count = u64::try_from(state.payload.modal_basis().unwrap().modes().len()).unwrap();
    let SolveResourcePolicy::ModalDenseHarmonic {
        full_dimension,
        mode_count: planned_modes,
        workspace_bytes_upper_bound,
    } = plan.input.structural_operator().unwrap().solve
    else {
        panic!("modal harmonic work must publish its dense workspace")
    };
    assert_eq!(
        full_dimension.0,
        u64::try_from(request.scope().physical_dof_count()).unwrap()
    );
    assert_eq!(planned_modes.0, mode_count);
    assert!(workspace_bytes_upper_bound.0 > 0);
    let large_basis_inputs = build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        Some(&state),
        &[],
        prepared(input_facts(4 * 1024 * 1024), entry_output(&request)),
    )
    .unwrap();
    assert!(large_basis_inputs.request.memory_bytes > plan.request.memory_bytes);
    assert!(large_basis_inputs.request.max_relay_bytes > plan.request.max_relay_bytes);
    assert_ne!(large_basis_inputs.identity, plan.identity);
    assert!(build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        None,
        &[],
        prepared(input_facts(1024), entry_output(&request)),
    )
    .is_err());
}

#[test]
fn frequency_recovery_plan_uses_the_same_exact_complex_field_layout_as_recovery() {
    let (workload, mesh, constraints, entry_request) = direct_case(2);
    let entry = execute_structural_frequency_entry(&workload, &mesh, &entry_request, None).unwrap();
    let topology = StructuralRecoveredFieldTopology::derive(workload.model(), Some(&mesh)).unwrap();
    let request = FieldRecoveryRequest::new(
        entry_request.step(),
        FieldRecoverySource::StructuralFrequencyEntry(entry.identity()),
        topology.identity,
    )
    .unwrap();
    let plan = build_structural_frequency_recovery_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        &entry,
        &[],
        prepared(
            input_facts(1024),
            recovery_output(&mesh, &constraints, &workload),
        ),
    )
    .unwrap();
    assert_eq!(
        plan.input.phase,
        FeaResourcePhase::StructuralFrequencyFieldRecovery(request.identity())
    );
    assert!(matches!(
        plan.input.structural_operator().unwrap().solve,
        SolveResourcePolicy::NoSolve { .. }
    ));
    let artifact =
        recover_structural_frequency_entry_fields(&workload, &mesh, &request, &entry).unwrap();
    let actual_values = artifact
        .fields()
        .iter()
        .map(|field| field.storage().as_host_f64().unwrap().len() as u64)
        .sum::<u64>();
    let actual_payload_bytes = u64::try_from(
        artifact
            .manifest()
            .unwrap()
            .canonical_bytes()
            .unwrap()
            .len(),
    )
    .unwrap()
        + artifact
            .fields()
            .iter()
            .flat_map(|field| canonical_field_partitions(field).unwrap())
            .map(|partition| u64::try_from(partition.canonical_bytes().unwrap().len()).unwrap())
            .sum::<u64>();
    assert!(
        actual_payload_bytes
            <= structural_frequency_recovery_payload_bytes_upper_bound(
                &mesh,
                &constraints,
                &workload,
            )
            .unwrap()
    );
    assert_eq!(plan.breakdown.phase_retained_bytes.0, actual_values * 8);
    assert!(plan.breakdown.phase_resident_bytes.0 > plan.breakdown.phase_retained_bytes.0);
    assert!(plan.request.accelerators.is_empty());

    let (other_workload, other_mesh, _, other_request) = direct_case(3);
    let other_entry =
        execute_structural_frequency_entry(&other_workload, &other_mesh, &other_request, None)
            .unwrap();
    let substituted = FieldRecoveryRequest::new(
        entry_request.step(),
        FieldRecoverySource::StructuralFrequencyEntry(other_entry.identity()),
        topology.identity,
    )
    .unwrap();
    assert!(build_structural_frequency_recovery_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &substituted,
        &entry,
        &[],
        prepared(
            input_facts(1024),
            recovery_output(&mesh, &constraints, &workload),
        ),
    )
    .is_err());
}
