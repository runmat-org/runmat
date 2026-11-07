mod resources;
use super::*;
use crate::results::field::{RecoveredField, RecoveredFieldCatalog};
use crate::results::observation::{
    evaluate_observation_sample, resolve_observation_plan, ObservationHistorySample,
    ObservationLimits,
};
use crate::topology::AnalysisProjection;
use crate::{ComputeBackend, ThermalProcedureControl};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

// A genuine but wrong-family authority for thermal catalog rejection checks.
fn structural_source_fixture() -> crate::execution::StructuralLinearWorkload {
    let model = crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let workload = crate::execution::StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    workload.validate().unwrap();
    workload
}

fn fixture(
    retention: RetentionPolicy,
    series: SeriesReduction,
) -> (
    ThermalWorkload,
    SolverMeshArtifact,
    ResolvedObservationPlan,
    ObservationFieldCatalog,
    ObservationHistoryCheckpoint,
) {
    fixture_with_ties(retention, series, ExtremaTiePolicy::FirstCanonical)
}

fn fixture_with_ties(
    retention: RetentionPolicy,
    series: SeriesReduction,
    ties: ExtremaTiePolicy,
) -> (
    ThermalWorkload,
    SolverMeshArtifact,
    ResolvedObservationPlan,
    ObservationFieldCatalog,
    ObservationHistoryCheckpoint,
) {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let mut step = ProcedureStep::thermal_transient("heat");
    step.accuracy.resources.maximum_samples = 3;
    step.observations = vec![ObservationRequest::History(ObservationSpec {
        observation_id: ObservationId("temperature_history".into()),
        quantity: ObservationQuantity::Temperature,
        value_shape: ObservationValueShape::Scalar,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::EveryAcceptedIncrement,
        reduction: (series != SeriesReduction::None).then_some(ReductionPolicy {
            spatial: SpatialReduction::None,
            series,
            extrema_ties: ties,
        }),
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::Automatic,
    })];
    model.sequence = AnalysisSequence::from_steps("history", vec![step.clone()]);
    let workload = ThermalWorkload {
        initial_fields: Vec::new(),
        model,
        step_id: step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    };
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let result = workload
        .prepare(&mesh)
        .unwrap()
        .execute_primary_fields()
        .unwrap();
    let manifest = super::super::ThermalSolverResultManifest::from_result(&result).unwrap();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let definitions = manifest.fields[0]
        .iter()
        .map(|entry| entry.definition.clone())
        .collect::<Vec<_>>();
    let plan = resolve_observation_plan(
        &step,
        &projection,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let id = ObservationId("temperature_history".into());
    let mut history = BoundedObservationHistory::new(&plan, &id).unwrap();
    let mut predecessor = None;
    for (ordinal, entries) in manifest.fields.iter().enumerate() {
        let sample_definitions = entries
            .iter()
            .map(|entry| entry.definition.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            resolve_observation_plan(
                &step,
                &projection,
                &sample_definitions,
                ObservationLimits::default()
            )
            .unwrap(),
            plan,
            "display field IDs must not change the semantic history plan"
        );
        let entry = &entries[0];
        let fields = RecoveredFieldCatalog::new(
            &projection,
            vec![RecoveredField::new(
                entry.definition.clone(),
                entry.sample,
                result.temperature_snapshots[ordinal].clone(),
            )
            .unwrap()],
        )
        .unwrap();
        let sample =
            evaluate_observation_sample(&plan, &id, &projection, &fields, entry.sample).unwrap();
        history.push(entry.sample, sample).unwrap();
        let checkpoint = history.checkpoint(predecessor).unwrap();
        predecessor = Some(checkpoint.manifest().identity);
    }
    let checkpoint = history.checkpoint(predecessor).unwrap();
    (
        workload,
        mesh,
        plan,
        ObservationFieldCatalog::thermal(manifest).unwrap(),
        checkpoint,
    )
}

#[test]
fn publication_catalog_preserves_history_authority_and_exact_coverage() {
    use crate::results::observation::{
        ObservationArtifact, ObservationArtifactInput, ObservationFieldSelection,
        ObservationHistoryPublication, ObservationPublicationCatalog, ObservationPublicationEntry,
        ObservationSamplePublication,
    };
    let (workload, mesh, plan, source, checkpoint) =
        fixture(RetentionPolicy::Canonical, SeriesReduction::Maximum);
    let ObservationFieldCatalogSource::Thermal { manifest } = source.source() else {
        unreachable!()
    };
    let producer = manifest.producer;
    let context = ObservationHistorySourceContext::Thermal {
        workload: &workload,
        mesh: &mesh,
    };
    let request = ObservationHistoryFinalizationRequest::new(
        plan.clone(),
        producer,
        ObservationId("temperature_history".into()),
        &checkpoint,
        &source,
        context,
    )
    .unwrap();
    let finalized = request
        .finalize(checkpoint.clone(), &source, context)
        .unwrap();
    let history = ObservationHistoryPublication::new(
        request,
        &checkpoint,
        &source,
        context,
        &finalized.history,
        finalized.series_reduction.as_ref(),
    )
    .unwrap();
    history
        .validate_history(
            &checkpoint,
            &source,
            context,
            &finalized.history,
            finalized.series_reduction.as_ref(),
        )
        .unwrap();
    let entry = ObservationPublicationEntry::History(Box::new(history.clone()));
    let step = &workload.model.sequence.steps[0];
    let catalog = ObservationPublicationCatalog::new(
        producer,
        plan.clone(),
        source.clone(),
        step,
        vec![entry.clone()],
        context,
    )
    .unwrap();
    let bytes = catalog.canonical_bytes().unwrap();
    assert_eq!(
        ObservationPublicationCatalog::from_canonical_bytes(&bytes).unwrap(),
        catalog
    );
    let accuracy = crate::execution::AccuracyEvaluationRequest::new(
        producer,
        crate::execution::AccuracySourceKind::Thermal,
        manifest.identity(),
        Some(&catalog),
        step.accuracy.clone(),
    )
    .unwrap();
    assert_eq!(accuracy.source_observations, Some(catalog.identity()));
    assert_eq!(
        crate::execution::AccuracyEvaluationRequest::from_canonical_bytes(
            &accuracy.canonical_bytes().unwrap()
        )
        .unwrap(),
        accuracy
    );
    assert!(crate::execution::AccuracyEvaluationRequest::new(
        producer,
        crate::execution::AccuracySourceKind::StructuralLinear,
        manifest.identity(),
        Some(&catalog),
        step.accuracy.clone(),
    )
    .is_err());
    let foreign_result = crate::execution::SolverResultIdentity::from_step_result(
        producer,
        Digest::sha256("foreign result"),
    )
    .unwrap();
    assert!(catalog
        .validate_source_result(
            producer,
            crate::execution::AccuracySourceKind::Thermal,
            foreign_result
        )
        .is_err());
    let mut obsolete = serde_json::to_value(&accuracy).unwrap();
    obsolete["observation_request"] = serde_json::Value::Null;
    assert!(
        serde_json::from_value::<crate::execution::AccuracyEvaluationRequest>(obsolete).is_err()
    );
    // Publishing complete histories is provenance, not fabricated error-estimator evidence.
    let accuracy_artifact =
        crate::results::accuracy::evaluate_accuracy_artifact(&accuracy, vec![]).unwrap();
    assert_eq!(
        accuracy_artifact.source_observations,
        Some(catalog.identity())
    );
    assert_eq!(
        accuracy_artifact.gate().decision,
        crate::results::accuracy::PublicationDecision::Block
    );
    assert!(ObservationPublicationCatalog::new(
        producer,
        plan.clone(),
        source.clone(),
        step,
        vec![],
        context
    )
    .is_err());
    assert!(ObservationPublicationCatalog::new(
        producer,
        plan.clone(),
        source.clone(),
        step,
        vec![entry.clone(), entry],
        context
    )
    .is_err());
    let mut missing_series = history.clone();
    missing_series.series = None;
    assert!(ObservationPublicationCatalog::new(
        producer,
        plan.clone(),
        source.clone(),
        step,
        vec![ObservationPublicationEntry::History(Box::new(
            missing_series
        ))],
        context
    )
    .is_err());
    let mut bad_series = history.series.unwrap();
    bad_series.contributions[0].value_count += 1;
    assert!(bad_series
        .validate_series(
            finalized.series_reduction.as_ref().unwrap(),
            &finalized.history,
            &plan
        )
        .is_err());
    let mut wire = serde_json::to_value(&catalog).unwrap();
    wire["producer"] =
        serde_json::to_value(StepIdentity::derive("foreign", &"step").unwrap()).unwrap();
    assert!(serde_json::from_value::<ObservationPublicationCatalog>(wire).is_err());
    assert!(catalog
        .validate_for(
            step,
            ObservationHistorySourceContext::Structural {
                source: crate::execution::StructuralObservationSource::Linear(
                    &structural_source_fixture()
                ),
                mesh: Some(&mesh)
            }
        )
        .is_err());
    // A genuine last-sample artifact is still not a publication of the authored History.
    let last = manifest.fields.len() - 1;
    let selection = ObservationFieldSelection::thermal(
        &workload,
        manifest,
        &mesh,
        last,
        vec![manifest.fields[last][0].field],
    )
    .unwrap();
    let evaluation =
        crate::execution::ObservationEvaluationRequest::new(selection, plan.clone()).unwrap();
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: evaluation.identity(),
        producer,
        source_selection: evaluation.selection.identity(),
        source: evaluation.selection.source(),
        topology_digest: plan.topology_digest,
        plan_identity: plan.identity,
        results: vec![checkpoint.samples().last().unwrap().result.clone()],
    })
    .unwrap();
    let sample = ObservationSamplePublication::new(evaluation, &artifact).unwrap();
    assert!(ObservationPublicationCatalog::new(
        producer,
        plan,
        source.clone(),
        step,
        vec![ObservationPublicationEntry::Sample(Box::new(sample))],
        context
    )
    .is_err());
}

#[test]
fn finalization_resource_plan_bounds_outputs_and_rejects_missing_series() {
    use crate::planning::resource::*;
    let (workload, mesh, plan, catalog, checkpoint) =
        fixture(RetentionPolicy::Canonical, SeriesReduction::Maximum);
    let ObservationFieldCatalogSource::Thermal { manifest } = catalog.source() else {
        unreachable!()
    };
    let context = ObservationHistorySourceContext::Thermal {
        workload: &workload,
        mesh: &mesh,
    };
    let request = ObservationHistoryFinalizationRequest::new(
        plan,
        manifest.producer,
        ObservationId("temperature_history".into()),
        &checkpoint,
        &catalog,
        context,
    )
    .unwrap();
    let inputs = PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(1024 * 1024),
        transport_closure_bytes: ByteCount(2 * 1024 * 1024),
        decoded_resident_bytes_admission_budget: ByteCount(8 * 1024 * 1024),
    };
    let history_bound =
        observation_history_finalization_payload_bytes_upper_bound(&request, &checkpoint, &inputs)
            .unwrap();
    let series_bound = observation_history_series_reduction_payload_bytes_upper_bound(
        &request,
        &checkpoint,
        &inputs,
    )
    .unwrap()
    .unwrap();
    let history_output = projected_canonical_output_resource_facts(history_bound).unwrap();
    let series_output = projected_canonical_output_resource_facts(series_bound).unwrap();
    let output_set = PreparedOutputSetResourceFacts {
        encoded_bytes_upper_bound: ByteCount(1024 * 1024),
    };
    let resources = build_observation_history_finalization_resource_plan(
        &request,
        &checkpoint,
        &catalog,
        context,
        PreparedPhaseResourceFacts {
            inputs: inputs.clone(),
            output: history_output.clone(),
            output_set: output_set.clone(),
        },
        Some(series_output.clone()),
    )
    .unwrap();
    assert_eq!(
        resources
            .input
            .demand
            .output_facts(crate::execution::AnalysisNumericalObjectKind::History),
        Some(&history_output)
    );
    assert!(build_observation_history_finalization_resource_plan(
        &request,
        &checkpoint,
        &catalog,
        context,
        PreparedPhaseResourceFacts {
            inputs: inputs.clone(),
            output: history_output.clone(),
            output_set: output_set.clone(),
        },
        None,
    )
    .is_err());
    let mut understated = history_output;
    understated.encoded_payload_bytes_upper_bound.0 -= 1;
    assert!(build_observation_history_finalization_resource_plan(
        &request,
        &checkpoint,
        &catalog,
        context,
        PreparedPhaseResourceFacts {
            inputs,
            output: understated,
            output_set,
        },
        Some(series_output),
    )
    .is_err());
    let finalized = request.finalize(checkpoint, &catalog, context).unwrap();
    assert!(finalized.history.canonical_bytes().unwrap().len() as u64 <= history_bound);
    assert!(
        finalized
            .series_reduction
            .unwrap()
            .canonical_bytes()
            .unwrap()
            .len() as u64
            <= series_bound
    );
}

#[test]
fn finalization_rejects_self_consistent_foreign_resolved_indices() {
    let (workload, mesh, plan, catalog, checkpoint) =
        fixture(RetentionPolicy::Canonical, SeriesReduction::None);
    let mut plan = plan.into_contents();
    let crate::results::observation::ResolvedObservationTarget::Nodes { indices, .. } =
        &mut plan.observations[0].target
    else {
        unreachable!()
    };
    for index in indices {
        *index += 1;
    }
    // Re-sign the plan as an untrusted producer could: internal identity consistency
    // does not establish that these indices came from the authoritative projection.
    let bytes = encode_contract(
        b"runmat-fea-observation/v1\0",
        "resolved-observation-plan",
        &(
            plan.schema_version,
            plan.topology_digest,
            &plan.step_id,
            &plan.independent_domain,
            &plan.observations,
        ),
        LIMITS,
    )
    .unwrap();
    plan.identity =
        serde_json::from_value(serde_json::to_value(Digest::sha256(bytes)).unwrap()).unwrap();
    let plan = crate::results::observation::ResolvedObservationPlan::try_from(plan).unwrap();
    let ObservationFieldCatalogSource::Thermal { manifest } = catalog.source() else {
        unreachable!()
    };
    let error = ObservationHistoryFinalizationRequest::new(
        plan,
        manifest.producer,
        ObservationId("temperature_history".into()),
        &checkpoint,
        &catalog,
        ObservationHistorySourceContext::Thermal {
            workload: &workload,
            mesh: &mesh,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("authoritative mesh and fields"));
}

#[test]
fn finalization_binds_catalog_checkpoint_and_authored_series() {
    let (workload, mesh, plan, catalog, checkpoint) =
        fixture(RetentionPolicy::Canonical, SeriesReduction::Maximum);
    let ObservationFieldCatalogSource::Thermal { manifest } = catalog.source() else {
        unreachable!()
    };
    let context = ObservationHistorySourceContext::Thermal {
        workload: &workload,
        mesh: &mesh,
    };
    let request = ObservationHistoryFinalizationRequest::new(
        plan,
        manifest.producer,
        ObservationId("temperature_history".into()),
        &checkpoint,
        &catalog,
        context,
    )
    .unwrap();
    let decoded = ObservationHistoryFinalizationRequest::from_canonical_bytes(
        &request.canonical_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(request, decoded);
    let final_result = decoded
        .finalize(checkpoint.clone(), &catalog, context)
        .unwrap();
    assert_eq!(final_result.history.samples().len(), 3);
    assert_eq!(
        final_result.series_reduction.unwrap().operation,
        SeriesReduction::Maximum
    );
    let mut tampered = request.clone();
    tampered.source_catalog = Digest::sha256("foreign catalog");
    let bytes = encode_contract(PREFIX, "history-finalization-request", &tampered, LIMITS).unwrap();
    assert!(ObservationHistoryFinalizationRequest::from_canonical_bytes(&bytes).is_err());
    tampered.identity = tampered.derive_identity().unwrap();
    assert!(tampered
        .validate_inputs(&checkpoint, &catalog, context)
        .is_err());
    assert!(request
        .validate_inputs(
            &checkpoint,
            &catalog,
            ObservationHistorySourceContext::Structural {
                source: crate::execution::StructuralObservationSource::Linear(
                    &structural_source_fixture()
                ),
                mesh: Some(&mesh)
            }
        )
        .is_err());
    let foreign_checkpoint = BoundedObservationHistory::resume(
        &request.plan,
        &request.observation_id,
        checkpoint.clone(),
    )
    .unwrap()
    .checkpoint(None)
    .unwrap();
    assert_ne!(
        foreign_checkpoint.manifest().identity,
        request.terminal_checkpoint
    );
    assert!(request
        .validate_inputs(&foreign_checkpoint, &catalog, context)
        .is_err());
    let mut other_workload = workload.clone();
    other_workload.residual_target *= 0.1;
    assert!(request
        .validate_inputs(
            &checkpoint,
            &catalog,
            ObservationHistorySourceContext::Thermal {
                workload: &other_workload,
                mesh: &mesh
            }
        )
        .is_err());
}

#[test]
fn finalization_rejects_eviction_and_missing_accepted_samples() {
    let (workload, mesh, plan, catalog, checkpoint) = fixture(
        RetentionPolicy::LatestSamples { count: 1 },
        SeriesReduction::None,
    );
    let ObservationFieldCatalogSource::Thermal { manifest } = catalog.source() else {
        unreachable!()
    };
    assert!(ObservationHistoryFinalizationRequest::new(
        plan,
        manifest.producer,
        ObservationId("temperature_history".into()),
        &checkpoint,
        &catalog,
        ObservationHistorySourceContext::Thermal {
            workload: &workload,
            mesh: &mesh
        }
    )
    .unwrap_err()
    .to_string()
    .contains("source-bound checkpoint lineage"));
    let (workload, mesh, plan, catalog, checkpoint) =
        fixture(RetentionPolicy::Canonical, SeriesReduction::None);
    let ObservationFieldCatalogSource::Thermal { manifest } = catalog.source() else {
        unreachable!()
    };
    let id = ObservationId("temperature_history".into());
    let mut shortened = BoundedObservationHistory::new(&plan, &id).unwrap();
    let ObservationHistorySample {
        coordinate, result, ..
    } = checkpoint.samples()[0].clone();
    shortened.push(coordinate, result).unwrap();
    assert!(ObservationHistoryFinalizationRequest::new(
        plan,
        manifest.producer,
        id,
        &shortened.checkpoint(None).unwrap(),
        &catalog,
        ObservationHistorySourceContext::Thermal {
            workload: &workload,
            mesh: &mesh
        }
    )
    .is_err());
}

#[test]
fn thermal_catalog_validation_fits_its_source_validation_law() {
    use crate::results::observation::{ObservationPublicationCatalog, ObservationPublicationEntry};
    let (workload, mesh, plan, source, checkpoint) =
        fixture(RetentionPolicy::Canonical, SeriesReduction::Maximum);
    let ObservationFieldCatalogSource::Thermal { manifest } = source.source() else {
        unreachable!()
    };
    let producer = manifest.producer;
    let context = ObservationHistorySourceContext::Thermal {
        workload: &workload,
        mesh: &mesh,
    };
    let request = ObservationHistoryFinalizationRequest::new(
        plan.clone(),
        producer,
        ObservationId("temperature_history".into()),
        &checkpoint,
        &source,
        context,
    )
    .unwrap();
    let finalized = request
        .finalize(checkpoint.clone(), &source, context)
        .unwrap();
    let history = crate::results::observation::ObservationHistoryPublication::new(
        request,
        &checkpoint,
        &source,
        context,
        &finalized.history,
        finalized.series_reduction.as_ref(),
    )
    .unwrap();
    let step = &workload.model.sequence.steps[0];
    let catalog = ObservationPublicationCatalog::new(
        producer,
        plan,
        source.clone(),
        step,
        vec![ObservationPublicationEntry::History(Box::new(history))],
        context,
    )
    .unwrap();
    let law = catalog
        .thermal_source_validation_resident_bytes_admitted(&workload, &mesh, u64::MAX)
        .unwrap();
    // Copies of the catalog's plan and selected values are accounted by the caller as one
    // decoded catalog; the canonical bytes bound that copy here.
    let catalog_copy = catalog.canonical_bytes().unwrap().len() as u64;
    let ((), _, peak) = crate::topology::measure_test_allocations(|| {
        catalog.validate_for(step, context).unwrap();
    });
    println!("thermal catalog validation peak={peak} law={law} catalog={catalog_copy}");
    assert!(
        peak as u64 <= law + catalog_copy,
        "validation peak {peak} exceeds {law} + {catalog_copy}"
    );
    assert!(catalog
        .thermal_source_validation_resident_bytes_admitted(&workload, &mesh, law - 1)
        .is_err());
}
