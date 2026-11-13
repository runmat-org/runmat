//! Cold modal-state/workload/result/field chain; not process-restart qualification.
use super::*;
use crate::{
    execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
        SequenceIdentity, SolverResultIdentity, StepIdentity, StructuralFrequencyEntryRequest,
        StructuralFrequencyEntryResult, StructuralFrequencySolverPlan, StructuralFrequencyWorkload,
        StructuralFrequencyWorkloadInput,
    },
    frequency::FrequencyResponsePlan,
    pipeline::structural_frequency::{
        execute_structural_frequency_entry, prepare_structural_frequency,
        recover_structural_frequency_entry_fields,
    },
    results::{field::StructuralRecoveredFieldTopology, observation::ObservationExecutionPolicy},
};
use runmat_analysis_core::{
    ObservationQuantity, ProcedureStepId, ReferenceStatePolicy, SequenceDependency,
    SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod,
    TopologyTransitionPolicy, TransitionId,
};
use runmat_execution::Digest;
use runmat_meshing_core::CanonicalMeshingContract;

#[test]
fn damped_modal_cold_state_entry_and_recovered_field_preserve_analytic_phase() {
    for quadratic in [false, true] {
        let mesh = rod_mesh(if quadratic { 4 } else { 8 }, quadratic);
        let k = 1.0;
        let f = k * WAVE_SPEED / std::f64::consts::TAU;
        let mut model = rod_model(&mesh, &[k]);
        damped::add_damping(&mut model);
        let basis = modal_basis(&model, &mesh, 1);
        let mut response =
            ProcedureStep::structural_modal_harmonic("rod_response", "rod_basis", "modal_basis");
        response.independent_domain = IndependentDomain::FrequencyHertz(vec![f]);
        model.sequence.steps = vec![ProcedureStep::structural_modal("rod_basis"), response];
        model.sequence.dependencies = vec![SequenceDependency {
            predecessor: ProcedureStepId("rod_basis".into()),
            successor: ProcedureStepId("rod_response".into()),
            reason: SequenceDependencyReason::StateTransfer,
        }];
        model.sequence.transitions = vec![StateTransition {
            transition_id: TransitionId("basis-to-response".into()),
            producer_step: ProcedureStepId("rod_basis".into()),
            consumer_step: ProcedureStepId("rod_response".into()),
            state_id: StateId("modal_basis".into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        }];
        let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
        let producer =
            StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
        let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
            producer,
            declaration: model.sequence.steps[0].state_outputs[0].clone(),
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            source_result: SolverResultIdentity::from_step_result(
                producer,
                basis.identity().digest(),
            )
            .unwrap(),
            payload: CanonicalSemanticStatePayload::ModalBasis {
                basis: Box::new(basis),
            },
        })
        .unwrap();
        let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
            model,
            step_id: ProcedureStepId("rod_response".into()),
            backend: ComputeBackend::Cpu,
            observation_execution: ObservationExecutionPolicy::scalable(),
            plan: FrequencyResponsePlan::new(f, &[], &[]).unwrap(),
            solver: StructuralFrequencySolverPlan::ModalReduced {
                basis_state: state.identity,
            },
            prestress_transition: None,
        })
        .unwrap();
        let prepared = prepare_structural_frequency(&workload, &mesh).unwrap();
        let request = StructuralFrequencyEntryRequest::new(
            &workload,
            prepared.study,
            prepared.step,
            prepared.execution_plan,
            prepared.scope.clone(),
            workload.plan().entries().unwrap()[0],
        )
        .unwrap();
        let hot =
            execute_structural_frequency_entry(&workload, &mesh, &request, Some(&state)).unwrap();
        let cold_state =
            CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap())
                .unwrap();
        let cold_workload =
            StructuralFrequencyWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap();
        let cold_request = StructuralFrequencyEntryRequest::from_canonical_bytes(
            &request.canonical_bytes().unwrap(),
        )
        .unwrap();
        let cold_mesh =
            SolverMeshArtifact::canonical_decode(&mesh.canonical_encode().unwrap()).unwrap();
        cold_mesh.validate_canonical().unwrap();
        let cold_prepared = prepare_structural_frequency(&cold_workload, &cold_mesh).unwrap();
        assert_eq!(cold_prepared.study, prepared.study);
        assert_eq!(cold_prepared.step, prepared.step);
        assert_eq!(cold_prepared.execution_plan, prepared.execution_plan);
        assert_eq!(cold_prepared.scope, prepared.scope);
        let cold = execute_structural_frequency_entry(
            &cold_workload,
            &cold_mesh,
            &cold_request,
            Some(&cold_state),
        )
        .unwrap();
        assert_eq!(hot.sample(), cold.sample());
        let imported =
            StructuralFrequencyEntryResult::from_canonical_bytes(&cold.canonical_bytes().unwrap())
                .unwrap();
        let displacement = imported.sample().displacement();
        let bound = if quadratic { 0.0005 } else { 0.005 };
        assert!(
            field_error_with(&mesh, displacement.real(), displacement.imaginary(), |x| {
                first_mode_reference(x, k)
            }) < bound
        );
        let topology =
            StructuralRecoveredFieldTopology::derive(cold_workload.model(), Some(&mesh)).unwrap();
        let recovery = FieldRecoveryRequest::new(
            prepared.step,
            FieldRecoverySource::StructuralFrequencyEntry(imported.identity()),
            FieldTopologyIdentity::Structural {
                topology: topology.identity,
            },
            FieldRecoveryPlan::resolve_structural(
                cold_workload.procedure_step().unwrap(),
                topology,
                0,
            )
            .unwrap(),
        )
        .unwrap();
        let fields =
            recover_structural_frequency_entry_fields(&cold_workload, &mesh, &recovery, &imported)
                .unwrap();
        let values = fields
            .fields()
            .iter()
            .find(|field| field.definition().quantity() == ObservationQuantity::Displacement)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let real = values.chunks_exact(2).map(|v| v[0]).collect::<Vec<_>>();
        let imaginary = values.chunks_exact(2).map(|v| v[1]).collect::<Vec<_>>();
        assert_eq!(real, displacement.real());
        assert_eq!(imaginary, displacement.imaginary());
        assert!(field_error_with(&mesh, &real, &imaginary, |x| first_mode_reference(x, k)) < bound);
        assert!(
            execute_structural_frequency_entry(&cold_workload, &mesh, &cold_request, None).is_err()
        );
        // Valid canonical authority naming another topology, not a corrupt state digest.
        let foreign = CanonicalSemanticState::new(CanonicalSemanticStateInput {
            producer: cold_state.producer,
            declaration: cold_workload.model().sequence.steps[0].state_outputs[0].clone(),
            topology_digest: Digest::sha256("foreign-rod"),
            source_result: cold_state.source_result,
            payload: cold_state.payload.clone(),
        })
        .unwrap();
        foreign.validate().unwrap();
        assert!(execute_structural_frequency_entry(
            &cold_workload,
            &mesh,
            &cold_request,
            Some(&foreign)
        )
        .is_err());
        assert_eq!(cold_state, state);
    }
}
