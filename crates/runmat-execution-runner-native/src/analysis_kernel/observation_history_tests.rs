use std::collections::BTreeSet;

use runmat_analysis_execution::{
    decode_prepared_observation_history_checkpoint, prepare_analysis_objects,
    prepared_input_resource_facts, projected_output_resource_facts,
    projected_output_set_resource_facts, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisKernel, AnalysisKernelInvocation, AnalysisWorkloadInput, AnalysisWorkloadRequest,
    PreparedAnalysisObjects,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ConstraintPlanIdentity, DiscretizationIdentity,
    ExecutionPlanIdentity, NumericalPlanIdentity, ObservationHistoryCheckpointRequest,
    OperatorIdentity, ProblemIdentity, SequenceIdentity, StepIdentity, StudyIdentity,
};
use runmat_analysis_fea::planning::resource::{
    build_observation_history_checkpoint_resource_plan, build_resource_plan,
    observation_history_checkpoint_payload_bytes_upper_bound, FeaResourceDemand, FeaResourcePlan,
};
use runmat_analysis_fea::results::field::{
    RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition, RecoveredFieldEntityDomain,
    RecoveredFieldNumericRepresentation, RecoveredFieldSample,
};
use runmat_analysis_fea::results::observation::{
    evaluate_observation_sample, resolve_observation_plan, ObservationLimits, ObservationResult,
};
use runmat_analysis_fea::topology::AnalysisProjection;
use runmat_execution::value::ValueRefKind;
use runmat_execution::Digest;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::ElementOrder;

use super::{AnalysisKernelOutput, NativeAnalysisKernel};

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn execution_identities() -> (StudyIdentity, StepIdentity, ExecutionPlanIdentity) {
    let study = StudyIdentity::from_authored_digest(digest("history-study"), None).unwrap();
    let sequence = SequenceIdentity::new(digest("history-sequence")).unwrap();
    let step = StepIdentity::new(sequence, digest("history-step")).unwrap();
    let constraints =
        ConstraintPlanIdentity::new(step, digest("history-mesh"), digest("history-constraints"))
            .unwrap();
    let problem =
        ProblemIdentity::new(step, digest("history-model"), None, constraints, Vec::new()).unwrap();
    let discretization = DiscretizationIdentity::new(
        problem,
        digest("history-mesh"),
        digest("history-formulation"),
    )
    .unwrap();
    let operator = OperatorIdentity::new(discretization, digest("history-operator")).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest("history-solver")).unwrap();
    let execution = ExecutionPlanIdentity::new(numerical, digest("history-execution")).unwrap();
    (study, step, execution)
}

fn checkpoint_host(inputs: &[PreparedAnalysisObjects]) -> AnalysisHostWorkload {
    let (study, step, execution) = execution_identities();
    let workload_inputs = inputs
        .iter()
        .map(|input| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: if input.manifest.kind
                == AnalysisNumericalObjectKind::ObservationHistoryCheckpointRequest
            {
                ValueRefKind::DriverObject
            } else {
                ValueRefKind::ResultObject
            },
        })
        .collect();
    AnalysisHostWorkload::new(
        AnalysisWorkloadRequest::new(
            study,
            step,
            execution,
            AnalysisWorkKind::CheckpointObservationHistory,
            workload_inputs,
            vec![AnalysisNumericalObjectKind::ObservationHistoryCheckpoint],
            BTreeSet::from([AnalysisCapabilityRequirement::Host {
                abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
            }]),
        )
        .unwrap(),
        ArtifactAccess {
            authorization_scope: "study:history".into(),
            encryption_context: digest("history-encryption"),
        },
        runmat_analysis_execution::AnalysisOutputPolicy::new(
            1024,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn prepare_kernel_output(output: &AnalysisKernelOutput) -> PreparedAnalysisObjects {
    output
        .clone()
        .into_prepared(256, ObjectInventoryLimits::default())
        .unwrap()
}

fn history_observation() -> runmat_analysis_core::ObservationRequest {
    runmat_analysis_core::ObservationRequest::History(runmat_analysis_core::ObservationSpec {
        observation_id: runmat_analysis_core::ObservationId("native_checkpoint_history".into()),
        quantity: runmat_analysis_core::ObservationQuantity::Displacement,
        value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
        target: runmat_analysis_core::ObservationTarget::WholeModel,
        source_location: runmat_analysis_core::FieldLocation::Node,
        output_location: runmat_analysis_core::FieldLocation::Node,
        recovery: runmat_analysis_core::RecoveryPolicy::Native,
        component: runmat_analysis_core::ObservationComponent::All,
        frame: runmat_analysis_core::CoordinateFrame::Global,
        sign_convention: runmat_analysis_core::SignConvention::Native,
        complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
        sampling: runmat_analysis_core::SamplingPolicy::TimeSeconds(vec![0.0, 1.0]),
        reduction: None,
        storage: runmat_analysis_core::ObservationStoragePolicy {
            precision: runmat_analysis_core::SignalPrecision::Real64,
            compression: runmat_analysis_core::CompressionPolicy::Lossless,
            retention: runmat_analysis_core::RetentionPolicy::LatestSamples { count: 1 },
            partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: runmat_analysis_core::ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(
            runmat_analysis_core::ObservationCostEstimate {
                operation_count: 1_000,
                peak_resident_bytes: 1_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            },
        ),
    })
}

#[test]
fn native_kernel_advances_exact_history_checkpoint_lineage() {
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let definition = RecoveredFieldDefinition::new(
        "displacement.native_checkpoint",
        runmat_analysis_core::ObservationQuantity::Displacement,
        runmat_analysis_core::FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        runmat_analysis_core::ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let observation_id = runmat_analysis_core::ObservationId("native_checkpoint_history".into());
    let mut step =
        runmat_analysis_core::ProcedureStep::structural_transient("native-checkpoint-history");
    step.independent_domain = runmat_analysis_core::IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: 10,
    };
    step.observations = vec![history_observation()];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let result_at = |time: f64| {
        let sample = RecoveredFieldSample::TimeSeconds(time);
        let field = RecoveredField::new(
            definition.clone(),
            sample,
            runmat_analysis_core::AnalysisField::host_f64(
                definition.field_id(),
                definition.layout(),
                vec![projection.nodes.len(), 3],
                vec![time + 1.0; projection.nodes.len() * 3],
            ),
        )
        .unwrap();
        let fields = RecoveredFieldCatalog::new(&projection, vec![field]).unwrap();
        evaluate_observation_sample(&plan, &observation_id, &projection, &fields, sample).unwrap()
    };
    let limits = ObjectInventoryLimits::default();
    let prepare_request = |request: &ObservationHistoryCheckpointRequest| {
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationHistoryCheckpointRequest,
            request.identity.digest(),
            &request.canonical_bytes().unwrap(),
            256,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap()
    };
    let prepare_sample = |sample: &ObservationResult| {
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationResult,
            sample.logical_identity.digest(),
            &sample.canonical_bytes().unwrap(),
            256,
            ObjectNamespace::ResultValue,
            limits,
        )
        .unwrap()
    };
    let with_resource_plan =
        |request: &ObservationHistoryCheckpointRequest,
         previous: Option<
            &runmat_analysis_fea::results::observation::ObservationHistoryCheckpoint,
        >,
         sample: &ObservationResult,
         mut inputs: Vec<PreparedAnalysisObjects>| {
            let input_facts = prepared_input_resource_facts(&inputs).unwrap();
            let payload_ceiling = observation_history_checkpoint_payload_bytes_upper_bound(
                request,
                previous,
                sample,
                &input_facts,
            )
            .unwrap();
            let plan = build_observation_history_checkpoint_resource_plan(
                request,
                previous,
                sample,
                input_facts,
                projected_output_resource_facts(payload_ceiling).unwrap(),
                projected_output_set_resource_facts(),
            )
            .unwrap();
            inputs.push(
                prepare_analysis_objects(
                    AnalysisNumericalObjectKind::ResourcePlan,
                    plan.identity.digest(),
                    &plan.canonical_bytes().unwrap(),
                    256,
                    ObjectNamespace::InputValue,
                    limits,
                )
                .unwrap(),
            );
            inputs
        };

    let first_sample = result_at(0.0);
    let first_request = ObservationHistoryCheckpointRequest::new(
        plan.clone(),
        execution_identities().1,
        observation_id.clone(),
        None,
        0,
        &first_sample,
    )
    .unwrap();
    let first_inputs = with_resource_plan(
        &first_request,
        None,
        &first_sample,
        vec![
            prepare_request(&first_request),
            prepare_sample(&first_sample),
        ],
    );
    let first_host = checkpoint_host(&first_inputs);
    let first_output = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(
                256,
                ObjectInventoryLimits::default(),
            )
            .unwrap(),
            kind: AnalysisWorkKind::CheckpointObservationHistory,
            host: &first_host,
            inputs: &first_inputs,
        })
        .unwrap();
    let first_objects = prepare_kernel_output(&first_output[0]);
    let first_checkpoint = decode_prepared_observation_history_checkpoint(&first_objects).unwrap();
    assert_eq!(first_checkpoint.manifest().predecessor, None);
    assert_eq!(first_checkpoint.manifest().observed_sample_count, 1);

    let second_sample = result_at(1.0);
    let second_request = ObservationHistoryCheckpointRequest::new(
        plan,
        execution_identities().1,
        observation_id,
        Some(first_checkpoint.manifest().identity),
        1,
        &second_sample,
    )
    .unwrap();
    let second_inputs = with_resource_plan(
        &second_request,
        Some(&first_checkpoint),
        &second_sample,
        vec![
            prepare_request(&second_request),
            prepare_sample(&second_sample),
            first_objects,
        ],
    );
    let second_host = checkpoint_host(&second_inputs);
    let second_output = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(
                256,
                ObjectInventoryLimits::default(),
            )
            .unwrap(),
            kind: AnalysisWorkKind::CheckpointObservationHistory,
            host: &second_host,
            inputs: &second_inputs,
        })
        .unwrap();
    let second_checkpoint =
        decode_prepared_observation_history_checkpoint(&prepare_kernel_output(&second_output[0]))
            .unwrap();
    assert_eq!(
        second_checkpoint.manifest().predecessor,
        Some(first_checkpoint.manifest().identity)
    );
    assert_eq!(second_checkpoint.manifest().observed_sample_count, 2);
    assert_eq!(second_checkpoint.samples().len(), 1);

    enum UnderstatedBudget {
        Output,
        EvidenceEncoding,
        EvidenceWorkspace,
    }
    for budget in [
        UnderstatedBudget::Output,
        UnderstatedBudget::EvidenceEncoding,
        UnderstatedBudget::EvidenceWorkspace,
    ] {
        let mut understated_inputs = second_inputs.clone();
        let resource_index = understated_inputs
            .iter()
            .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .unwrap();
        let resource_plan =
            FeaResourcePlan::from_canonical_bytes(&understated_inputs[resource_index].payload())
                .unwrap();
        let mut understated_input = resource_plan.input;
        let FeaResourceDemand::StructuralObservationHistoryCheckpoint(demand) =
            &mut understated_input.demand
        else {
            unreachable!()
        };
        assert!(demand.execution_evidence_encoded_bytes_upper_bound.0 > 1);
        assert!(
            demand.execution_evidence_workspace_bytes_upper_bound.0
                > demand.execution_evidence_encoded_bytes_upper_bound.0
        );
        match budget {
            UnderstatedBudget::Output => {
                demand.output.construction_resident_bytes_admission_budget.0 -= 1
            }
            UnderstatedBudget::EvidenceEncoding => {
                demand.execution_evidence_encoded_bytes_upper_bound.0 -= 1
            }
            UnderstatedBudget::EvidenceWorkspace => {
                demand.execution_evidence_workspace_bytes_upper_bound.0 -= 1
            }
        }
        let understated_plan = build_resource_plan(understated_input).unwrap();
        understated_inputs[resource_index] = prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            understated_plan.identity.digest(),
            &understated_plan.canonical_bytes().unwrap(),
            256,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let understated_host = checkpoint_host(&understated_inputs);
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(
                    256,
                    ObjectInventoryLimits::default()
                )
                .unwrap(),
                kind: AnalysisWorkKind::CheckpointObservationHistory,
                host: &understated_host,
                inputs: &understated_inputs,
            })
            .is_err());
    }
    let mut substituted_inputs = second_inputs;
    substituted_inputs[1] = prepare_sample(&first_sample);
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(
                256,
                ObjectInventoryLimits::default()
            )
            .unwrap(),
            kind: AnalysisWorkKind::CheckpointObservationHistory,
            host: &second_host,
            inputs: &substituted_inputs,
        })
        .is_err());
}
