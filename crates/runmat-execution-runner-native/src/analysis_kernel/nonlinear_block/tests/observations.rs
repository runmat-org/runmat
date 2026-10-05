//! Original J2 fields through shared observation metadata; no elastic stress reconstruction.
use super::*;
use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        AccuracySourceKind, AnalysisNumericalObjectKind, FieldRecoveryPlan, FieldRecoveryRequest,
        FieldRecoverySource, FieldTopologyIdentity, ObservationEvaluationRequest,
        ObservationHistorySourceContext, StructuralObservationSource,
    },
    post::fields::take_structural_observable_fields,
    results::{
        field::{RecoveredFieldArtifact, StructuralRecoveredFieldTopology},
        observation::{
            resolve_structural_observation_plan, ObservationArtifact, ObservationArtifactInput,
            ObservationFieldCatalog, ObservationFieldSelection, ObservationLimits,
            ObservationPublicationCatalog, ObservationPublicationEntry,
            ObservationSamplePublication, ObservationSelectionCoordinate,
        },
    },
};

pub(super) fn field(
    id: &str,
    quantity: ObservationQuantity,
    location: FieldLocation,
    shape: ObservationValueShape,
) -> ObservationRequest {
    ObservationRequest::Field(ObservationSpec {
        observation_id: ObservationId(id.into()),
        quantity,
        value_shape: shape,
        target: ObservationTarget::WholeModel,
        source_location: location,
        output_location: location,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 100_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        }),
    })
}

#[test]
fn cold_final_j2_observations_preserve_original_plastic_stress_and_source() {
    for quadratic in [false, true] {
        let (mut workload, mesh) = fixture(quadratic);
        workload.model.sequence.steps[0].observations = vec![
            field(
                "displacement",
                ObservationQuantity::Displacement,
                FieldLocation::Node,
                ObservationValueShape::Vector3,
            ),
            field(
                "stress",
                ObservationQuantity::Stress,
                FieldLocation::Element,
                ObservationValueShape::SymmetricTensor3,
            ),
        ];
        let request = NonlinearSolidBlockRequest::initial(
            &workload,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            workload.controls.max_accepted_increments,
        )
        .unwrap();
        let block = workload
            .execute_block(mesh.clone(), &request, None)
            .unwrap();
        assert!(block.complete);
        let mut original =
            runmat_analysis_fea::pipeline::nonlinear::recover_original_nonlinear_block(
                &workload, &mesh, &block, 0,
            )
            .unwrap();
        assert_eq!(original.run.observations.len(), 2);
        let fields = take_structural_observable_fields(&mut original.run.fields).unwrap();
        let definitions = fields
            .iter()
            .map(|f| f.definition().clone())
            .collect::<Vec<_>>();
        let topology =
            StructuralRecoveredFieldTopology::derive(&workload.model, Some(&mesh)).unwrap();
        let constraints =
            CanonicalStructuralConstraintPlan::compile_for_model(&workload.model, Some(&mesh))
                .unwrap();
        let step = &workload.model.sequence.steps[0];
        let plan = resolve_structural_observation_plan(
            step,
            &workload.model,
            Some(&mesh),
            &constraints,
            &definitions,
            ObservationLimits::default(),
        )
        .unwrap();
        let producer = workload.producer_identity().unwrap();
        let source_result = block.source_result_identity().unwrap();
        let numerical_source = FieldRecoverySource::SolverResult(source_result);
        let recovery = FieldRecoveryRequest::new(
            producer,
            numerical_source,
            FieldTopologyIdentity::Structural {
                topology: topology.identity,
            },
            FieldRecoveryPlan::resolve_structural(step, topology, constraints.groups.len())
                .unwrap(),
        )
        .unwrap();
        let recovered = RecoveredFieldArtifact::new(
            recovery.identity(),
            numerical_source,
            topology.identity.discretization_digest(),
            topology.entity_counts,
            fields,
        )
        .unwrap();
        let catalog =
            ObservationFieldCatalog::structural(recovery.clone(), recovered.manifest().unwrap())
                .unwrap();
        let selected = catalog
            .canonical_selected_fields(&plan, ObservationSelectionCoordinate::StructuralRecovery)
            .unwrap();
        let selection = ObservationFieldSelection::structural(
            &recovery,
            &recovered.manifest().unwrap(),
            selected,
        )
        .unwrap();
        let evaluation = ObservationEvaluationRequest::new(selection, plan.clone()).unwrap();
        let artifact = ObservationArtifact::new(ObservationArtifactInput {
            evaluation_request: evaluation.identity(),
            producer,
            source_selection: evaluation.selection.identity(),
            source: evaluation.selection.source(),
            topology_digest: plan.topology_digest,
            plan_identity: plan.identity,
            results: original.run.observations,
        })
        .unwrap();
        let sample = ObservationSamplePublication::new(evaluation, &artifact).unwrap();
        let context = ObservationHistorySourceContext::Structural {
            source: StructuralObservationSource::NonlinearSolid(&workload),
            mesh: Some(&mesh),
        };
        let publication = ObservationPublicationCatalog::new(
            producer,
            plan,
            catalog,
            step,
            vec![ObservationPublicationEntry::Sample(Box::new(sample))],
            context,
        )
        .unwrap();
        let bytes = publication.canonical_bytes().unwrap();
        drop(publication);
        let cold = ObservationPublicationCatalog::from_canonical_bytes(&bytes).unwrap();
        cold.validate_for(step, context).unwrap();
        cold.validate_source_result(producer, AccuracySourceKind::NonlinearSolid, source_result)
            .unwrap();
        assert!(cold
            .validate_source_result(
                producer,
                AccuracySourceKind::NonlinearSolid,
                runmat_analysis_fea::execution::SolverResultIdentity::new(
                    request.workload,
                    Digest::sha256(b"foreign original block")
                )
                .unwrap()
            )
            .is_err());
        let stress = artifact
            .results()
            .iter()
            .find(|r| r.observation_id.0 == "stress")
            .unwrap();
        // Independent hardening law: sigma_y + H*alpha = 5 + 30*.02 = 5.6.
        // Elastic reconstruction from terminal strain would instead give 25.6.
        for tensor in stress.field.as_host_f64().unwrap().chunks_exact(6) {
            assert!((tensor[0] - 5.6).abs() < 2e-7, "{tensor:?}");
            assert!(tensor[1..].iter().all(|v| v.abs() < 2e-7));
        }
        let mut foreign_step = step.clone();
        foreign_step.observations.pop();
        assert!(cold.validate_authored_step(&foreign_step).is_err());
        let mut unsupported = workload.clone();
        let spec = unsupported.model.sequence.steps[0].observations[0]
            .spec()
            .unwrap()
            .clone();
        unsupported.model.sequence.steps[0].observations[0] = ObservationRequest::History(spec);
        assert!(StructuralObservationSource::NonlinearSolid(&unsupported)
            .validate()
            .is_err());
    }
}

/// Rehash the entire claimed admission so rejection checks actual source demand.
pub(super) fn reject_recovery_understatements(
    prepared: &runmat_analysis_execution::PreparedNonlinearFieldRecovery,
) {
    use runmat_analysis_execution::{
        prepare_analysis_objects, AnalysisHostWorkload, AnalysisWorkloadInput,
        AnalysisWorkloadRequest,
    };
    use runmat_analysis_fea::{
        execution::{ExecutionPlanIdentity, NumericalPlanIdentity},
        planning::resource::{build_resource_plan, FeaResourceDemand},
    };
    use runmat_execution::value::ValueRefKind;
    use runmat_execution_artifact::ObjectNamespace;
    let original = prepared
        .inputs
        .iter()
        .find(|input| {
            input.manifest.kind == AnalysisNumericalObjectKind::NonlinearSolidBlockArtifact
        })
        .unwrap();
    let block = NonlinearSolidBlockArtifact::from_canonical_bytes(
        &original.payload(),
        original.manifest.logical_identity,
    )
    .unwrap();
    for substitution in 0..3 {
        let mut input = prepared.resource_plan.input.clone();
        let FeaResourceDemand::NonlinearFieldRecovery(demand) = &mut input.demand else {
            panic!("original recovery demand absent")
        };
        match substitution {
            0 => demand.inputs.decoded_resident_bytes_admission_budget.0 -= 1,
            1 => demand.recovery_workspace_bytes.0 -= 1,
            2 => demand.output.construction_resident_bytes_admission_budget.0 -= 1,
            _ => unreachable!(),
        }
        let altered = build_resource_plan(input).unwrap();
        let mut inputs = prepared.inputs.clone();
        let position = inputs
            .iter()
            .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .unwrap();
        inputs[position] = prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            altered.identity.digest(),
            &altered.canonical_bytes().unwrap(),
            prepared.output_policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            prepared.output_policy.inventory_limits(),
        )
        .unwrap();
        let source = &prepared.host.workload;
        let mut closure = source.inputs.clone();
        let reference = closure
            .iter_mut()
            .find(|input| input.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .unwrap();
        *reference = AnalysisWorkloadInput {
            kind: inputs[position].manifest.kind,
            logical_identity: inputs[position].manifest.logical_identity,
            payload_digest: inputs[position].manifest.payload_digest,
            manifest_digest: inputs[position].root.digest,
            manifest_encoded_length: inputs[position].root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        };
        let execution_plan = ExecutionPlanIdentity::new(
            NumericalPlanIdentity::from_nonlinear_solid_block(&block.request).unwrap(),
            altered.identity.digest(),
        )
        .unwrap();
        let host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                source.study,
                source.step,
                execution_plan,
                source.kind,
                closure,
                source.outputs.clone(),
                source.required_capabilities.clone(),
            )
            .unwrap(),
            prepared.host.artifact_access.clone(),
            prepared.host.output_policy,
        )
        .unwrap();
        assert!(
            crate::NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    host: &host,
                    kind: source.kind,
                    inputs: &inputs,
                    output_policy: prepared.output_policy
                })
                .is_err(),
            "self-consistent recovery understatement {substitution} accepted"
        );
    }
}
