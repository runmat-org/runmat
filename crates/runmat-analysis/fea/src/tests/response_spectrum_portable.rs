use runmat_analysis_core::{
    AnalysisSequence, DirectionalCombination, ModalCombination, ModalTruncationCorrection,
    ModalTruncationPolicy as SpectrumTruncationPolicy, ProcedureStep, ReferenceStatePolicy,
    ResponseSpectrum, ResponseSpectrumId, RigidResponsePolicy, SequenceDependencyReason,
    SpectrumAbscissa, SpectrumConstruction, SpectrumDamping, SpectrumDirection, SpectrumEnvelope,
    SpectrumEvidence, SpectrumExtrapolation, SpectrumInterpolation, SpectrumResponse,
    StateTransition, StateTransitionMethod, StateUse, TopologyTransitionPolicy, TransitionId,
};
use runmat_execution::Digest;

use crate::{
    execution::{
        AccuracySourceKind, AccuracySourceReceipt, AnalysisWorkloadIdentity,
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        CanonicalStateTransition, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralModalResponseSpectrumAssemblyRequest,
        StructuralModalResponseSpectrumDirectionRequest,
        StructuralModalResponseSpectrumNumericalReceipt,
        StructuralModalResponseSpectrumSolverResult, StructuralModalResponseSpectrumWorkload,
        StructuralModalResponseSpectrumWorkloadInput, StructuralModalWorkload,
        StructuralModalWorkloadInput,
    },
    fixtures::{fixture_model, FixtureId},
    pipeline::{
        modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
        structural_response_spectrum::{
            execute_structural_modal_response_spectrum_direction,
            prepare_structural_modal_response_spectrum,
            recover_structural_modal_response_spectrum_fields,
        },
    },
    planning::resource::{
        build_resource_plan, structural_modal_response_spectrum_assembly_payload_bytes_upper_bound,
        structural_modal_response_spectrum_direction_payload_bytes_upper_bound,
        structural_modal_response_spectrum_receipt_payload_bytes_upper_bound, ByteCount,
        FeaResourceDemand, PreparedInputResourceFacts, PreparedOutputResourceFacts,
        PreparedOutputSetResourceFacts,
    },
    results::observation::ObservationExecutionPolicy,
    ComputeBackend,
};

#[test]
fn portable_direction_tasks_reassemble_in_authored_order_and_bind_receipt() {
    let spectra = vec![
        spectrum("x", [1.0, 0.0, 0.0]),
        spectrum("y", [0.0, 1.0, 0.0]),
    ];
    let (workload, prepared) = prepared_response(spectra);
    let scope = prepared.scope().clone();
    let mut entries = workload
        .spectra()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(ordinal, spectrum)| {
            let request = StructuralModalResponseSpectrumDirectionRequest::new(
                &workload,
                scope.clone(),
                ordinal,
            )
            .unwrap();
            assert_eq!(request.spectrum(), *spectrum);
            execute_structural_modal_response_spectrum_direction(&workload, &prepared, &request)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let projection = prepared.resources();
    let direction_payload = structural_modal_response_spectrum_direction_payload_bytes_upper_bound(
        entries[0].request(),
    )
    .unwrap();
    let direction_plan = projection
        .plan_for_direction(
            entries[0].request(),
            prepared_inputs(),
            prepared_output(direction_payload),
            prepared_output_set(),
            crate::planning::resource::LocalExecutionPlan::serial(),
        )
        .unwrap();
    let FeaResourceDemand::StructuralModalResponseSpectrumDirection(direction_demand) =
        &direction_plan.input.demand
    else {
        panic!("direction plan has the wrong typed demand");
    };
    assert_eq!(direction_demand.mode_count_upper_bound.0, 3);
    entries.reverse();
    let assembly = StructuralModalResponseSpectrumAssemblyRequest::new(scope, entries).unwrap();
    assert_eq!(assembly.entries()[0].request().ordinal(), 0);
    let result = StructuralModalResponseSpectrumSolverResult::new(&workload, assembly).unwrap();
    let recovered =
        recover_structural_modal_response_spectrum_fields(&workload, &prepared, result.numerical())
            .unwrap();
    assert_eq!(recovered.source_result(), result.numerical().identity());
    assert!(recovered
        .field(crate::solve::dynamics::StructuralLinearFieldQuantity::Stress)
        .is_some());
    let result_payload =
        structural_modal_response_spectrum_assembly_payload_bytes_upper_bound(result.assembly())
            .unwrap();
    let assembly_plan = projection
        .plan_for_assembly(
            result.assembly(),
            prepared_inputs(),
            crate::planning::resource::PreparedStructuralResponseSpectrumAssemblyOutputs {
                result: prepared_output(result_payload),
                receipt: prepared_output(
                    structural_modal_response_spectrum_receipt_payload_bytes_upper_bound(),
                ),
                field_recovery: prepared_output(
                    projection
                        .field_recovery_payload_bytes_upper_bound()
                        .unwrap(),
                ),
                state_profiles: publication_output(
                    &workload,
                    result.assembly().scope().topology_digest(),
                ),
            },
            prepared_output_set(),
            crate::planning::resource::LocalExecutionPlan::serial(),
        )
        .unwrap();
    let FeaResourceDemand::StructuralModalResponseSpectrumAssembly(assembly_demand) =
        &assembly_plan.input.demand
    else {
        panic!("assembly plan has the wrong typed demand");
    };
    assert_eq!(assembly_demand.missing_mass_static_solve_count.0, 0);
    assert_eq!(assembly_demand.missing_mass_reduced_dof_count.0, 0);
    assert_eq!(
        assembly_demand
            .missing_mass_system_workspace_bytes_upper_bound
            .0,
        0
    );
    assert_eq!(
        assembly_demand
            .missing_mass_total_iteration_count_upper_bound
            .0,
        0
    );
    let mut inconsistent = assembly_plan.input.clone();
    let FeaResourceDemand::StructuralModalResponseSpectrumAssembly(demand) =
        &mut inconsistent.demand
    else {
        unreachable!()
    };
    demand.missing_mass_system_workspace_bytes_upper_bound = ByteCount(1);
    assert!(build_resource_plan(inconsistent).is_err());
    assert_eq!(
        StructuralModalResponseSpectrumSolverResult::from_canonical_bytes(
            &result.canonical_bytes().unwrap(),
        )
        .unwrap(),
        result
    );
    let generic_workload = AnalysisWorkloadIdentity::for_test("modal-spectrum").unwrap();
    let receipt =
        StructuralModalResponseSpectrumNumericalReceipt::new(generic_workload, &result).unwrap();
    let step = workload.procedure_step().unwrap();
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&workload.model().sequence).unwrap(),
        step,
    )
    .unwrap();
    let publication = crate::execution::SemanticStatePublicationRequest::new(
        crate::execution::SemanticStatePublicationSource::StructuralResponseEnvelope,
        producer,
        step.state_outputs[0].clone(),
        result.assembly().scope().topology_digest(),
        receipt.source_result(),
    )
    .unwrap();
    let state = CanonicalSemanticState::from_structural_response_envelope_publication(
        &publication,
        &result,
    )
    .unwrap();
    let profile = publication
        .encoding_profile(
            crate::execution::SemanticStatePublicationProfileSource::StructuralResponseEnvelope(
                &result,
            ),
        )
        .unwrap();
    assert_eq!(
        profile,
        runmat_canonical_codec::profile_serialized(
            &state,
            runmat_canonical_codec::CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
        )
        .unwrap()
    );
    assert!(state.canonical_bytes().unwrap().len() as u64 <= profile.encoded_bytes_upper_bound);
    receipt
        .validate_against(receipt.source_result(), &result)
        .unwrap();
    let receipt_bytes = receipt.canonical_bytes().unwrap();
    assert!(matches!(
        AccuracySourceReceipt::from_canonical_bytes(
            AccuracySourceKind::StructuralModalResponseSpectrum,
            &receipt_bytes,
        )
        .unwrap(),
        AccuracySourceReceipt::StructuralModalResponseSpectrum(_)
    ));
    assert!(AccuracySourceReceipt::from_canonical_bytes(
        AccuracySourceKind::StructuralRandomResponse,
        &receipt_bytes,
    )
    .is_err());
}

#[test]
fn pseudo_acceleration_missing_mass_static_solves_are_budgeted_exactly() {
    let mut corrected = spectrum("x", [1.0, 0.0, 0.0]);
    corrected.response = SpectrumResponse::PseudoAcceleration;
    corrected.truncation.correction = ModalTruncationCorrection::MissingMass {
        zero_period_acceleration: 9.81,
    };
    let mut uncorrected = spectrum("y", [0.0, 1.0, 0.0]);
    uncorrected.response = SpectrumResponse::PseudoAcceleration;
    let plan = assembly_plan(vec![corrected, uncorrected]);
    let FeaResourceDemand::StructuralModalResponseSpectrumAssembly(demand) = &plan.input.demand
    else {
        panic!("assembly plan has the wrong typed demand");
    };
    assert_eq!(demand.missing_mass_static_solve_count.0, 1);
    assert!(demand.missing_mass_reduced_dof_count.0 > 0);
    assert!(demand.missing_mass_system_workspace_bytes_upper_bound.0 > 0);
    assert_eq!(
        demand.missing_mass_total_iteration_count_upper_bound.0,
        demand
            .missing_mass_reduced_dof_count
            .0
            .saturating_mul(20)
            .max(1_000)
    );

    let mut inconsistent = plan.input.clone();
    let FeaResourceDemand::StructuralModalResponseSpectrumAssembly(demand) =
        &mut inconsistent.demand
    else {
        unreachable!()
    };
    demand.missing_mass_total_iteration_count_upper_bound.0 -= 1;
    assert!(build_resource_plan(inconsistent).is_err());

    let mut partial = plan.input.clone();
    let FeaResourceDemand::StructuralModalResponseSpectrumAssembly(demand) = &mut partial.demand
    else {
        unreachable!()
    };
    demand.missing_mass_system_workspace_bytes_upper_bound = ByteCount(0);
    assert!(build_resource_plan(partial).is_err());
}

#[test]
fn absolute_acceleration_missing_mass_does_not_budget_a_deformation_solve() {
    let spectra = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        .into_iter()
        .enumerate()
        .map(|(ordinal, direction)| {
            let mut spectrum = spectrum(&format!("absolute-{ordinal}"), direction);
            spectrum.response = SpectrumResponse::AbsoluteAcceleration;
            spectrum.truncation.correction = ModalTruncationCorrection::MissingMass {
                zero_period_acceleration: 9.81,
            };
            spectrum
        })
        .collect();
    let plan = assembly_plan(spectra);
    let FeaResourceDemand::StructuralModalResponseSpectrumAssembly(demand) = &plan.input.demand
    else {
        panic!("assembly plan has the wrong typed demand");
    };
    assert_eq!(demand.missing_mass_static_solve_count.0, 0);
    assert_eq!(demand.missing_mass_reduced_dof_count.0, 0);
    assert_eq!(demand.missing_mass_system_workspace_bytes_upper_bound.0, 0);
    assert_eq!(demand.missing_mass_total_iteration_count_upper_bound.0, 0);
}

fn assembly_plan(spectra: Vec<ResponseSpectrum>) -> crate::planning::resource::FeaResourcePlan {
    let (workload, prepared) = prepared_response(spectra);
    let entries = workload
        .spectra()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(ordinal, _)| {
            let request = StructuralModalResponseSpectrumDirectionRequest::new(
                &workload,
                prepared.scope().clone(),
                ordinal,
            )
            .unwrap();
            execute_structural_modal_response_spectrum_direction(&workload, &prepared, &request)
                .unwrap()
        })
        .collect();
    let assembly =
        StructuralModalResponseSpectrumAssemblyRequest::new(prepared.scope().clone(), entries)
            .unwrap();
    let result = StructuralModalResponseSpectrumSolverResult::new(&workload, assembly).unwrap();
    let result_payload =
        structural_modal_response_spectrum_assembly_payload_bytes_upper_bound(result.assembly())
            .unwrap();
    prepared
        .resources()
        .plan_for_assembly(
            result.assembly(),
            prepared_inputs(),
            crate::planning::resource::PreparedStructuralResponseSpectrumAssemblyOutputs {
                result: prepared_output(result_payload),
                receipt: prepared_output(
                    structural_modal_response_spectrum_receipt_payload_bytes_upper_bound(),
                ),
                field_recovery: prepared_output(
                    prepared
                        .resources()
                        .field_recovery_payload_bytes_upper_bound()
                        .unwrap(),
                ),
                state_profiles: publication_output(
                    &workload,
                    result.assembly().scope().topology_digest(),
                ),
            },
            prepared_output_set(),
            crate::planning::resource::LocalExecutionPlan::serial(),
        )
        .unwrap()
}

fn publication_output(
    workload: &StructuralModalResponseSpectrumWorkload,
    topology: Digest,
) -> PreparedOutputResourceFacts {
    prepared_output(
        crate::execution::structural_state_publication_profiles_encoding_profile(
            workload.model(),
            workload.procedure_step().unwrap(),
            &[crate::execution::SemanticStatePublicationSource::StructuralResponseEnvelope],
            topology,
        )
        .unwrap()
        .encoded_bytes_upper_bound,
    )
}

fn prepared_response(
    spectra: Vec<ResponseSpectrum>,
) -> (
    StructuralModalResponseSpectrumWorkload,
    crate::pipeline::structural_response_spectrum::PreparedStructuralModalResponseSpectrum,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut modal = ProcedureStep::structural_modal("modes");
    modal.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 3 };
    let response = ProcedureStep::structural_modal_response_spectrum(
        "spectrum",
        spectra
            .iter()
            .map(|spectrum| spectrum.spectrum_id.clone())
            .collect(),
        "modes",
        "modal_basis",
    );
    let mut sequence =
        AnalysisSequence::from_steps("modal-spectrum", vec![modal.clone(), response.clone()]);
    sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId("modes-to-spectrum".into()),
        producer_step: modal.step_id.clone(),
        consumer_step: response.step_id.clone(),
        state_id: runmat_analysis_core::StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    model.sequence = sequence;
    model.response_spectra = spectra;
    let mesh = super::single_tetrahedron_solver_mesh_for_model(&model);
    let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let constraints = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    let modal_workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: modal.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let modal_result = crate::execute_structural_modal_workload(&modal_workload, &mesh).unwrap();
    let sequence_identity = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence_identity, &modal).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal_result.basis_identity().digest())
            .unwrap();
    let basis_state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: modal.state_outputs[0].clone(),
        topology_digest,
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal_result.basis().clone()),
        },
    })
    .unwrap();
    let transition = CanonicalStateTransition::bind_upstream(
        &model,
        &response.step_id,
        &basis_state,
        StateUse::ModalBasis,
        topology_digest,
    )
    .unwrap();
    let workload = StructuralModalResponseSpectrumWorkload::new(
        StructuralModalResponseSpectrumWorkloadInput {
            model,
            step_id: response.step_id,
            backend: ComputeBackend::Cpu,
            observation_execution: ObservationExecutionPolicy::scalable(),
            modal_basis_transition: transition,
        },
    )
    .unwrap();
    let prepared = prepare_structural_modal_response_spectrum(
        &workload,
        &mesh,
        &basis_state,
        &constraints,
        &[],
    )
    .unwrap();
    (workload, prepared)
}

fn prepared_inputs() -> PreparedInputResourceFacts {
    PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(1024),
        transport_closure_bytes: ByteCount(2048),
        decoded_resident_bytes_admission_budget: ByteCount(4096),
    }
}

fn prepared_output(payload: u64) -> PreparedOutputResourceFacts {
    PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(payload),
        transport_closure_bytes_upper_bound: ByteCount(payload + 4096),
        construction_resident_bytes_admission_budget: ByteCount(payload),
    }
}

fn prepared_output_set() -> PreparedOutputSetResourceFacts {
    PreparedOutputSetResourceFacts {
        encoded_bytes_upper_bound: ByteCount(4096),
    }
}

fn spectrum(id: &str, direction: [f64; 3]) -> ResponseSpectrum {
    ResponseSpectrum {
        spectrum_id: ResponseSpectrumId(id.into()),
        abscissa: SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Frequency,
            _,
        >::new(vec![1.0e-3, 1.0e8])),
        ordinates: runmat_analysis_core::NumericalDataSource::authored(
            vec![1.0e-3, 1.0e-3],
            format!("sha256:{id}"),
        ),
        damping: SpectrumDamping::Ratio(0.05),
        response: SpectrumResponse::RelativeDisplacement,
        envelope: SpectrumEnvelope::Maximax,
        interpolation: SpectrumInterpolation::LinearFrequency,
        extrapolation: SpectrumExtrapolation::HoldEndpoint,
        direction: SpectrumDirection {
            frame: runmat_analysis_core::CoordinateFrame::Global,
            vector: direction,
        },
        modal_combination: ModalCombination::AbsoluteSum,
        directional_combination: DirectionalCombination::Srss,
        truncation: SpectrumTruncationPolicy {
            minimum_cumulative_mass_ratio: 0.9,
            correction: ModalTruncationCorrection::None,
        },
        rigid_response: RigidResponsePolicy::Disabled,
        construction: SpectrumConstruction::AuthoredDesign,
        evidence: SpectrumEvidence {
            sampling_adequacy_ratio: 1.0,
            max_interpolation_error: None,
        },
    }
}
