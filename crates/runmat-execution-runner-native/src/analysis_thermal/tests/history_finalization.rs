//! Actual admitted solve -> observations -> checkpoints -> terminal History publication.
mod source_budget;
use super::{execute, planning_policy, preparation};
use runmat_analysis_core::*;
use runmat_analysis_execution::*;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, AnalysisWorkKind,
        ObservationHistoryFinalizationRequest, ObservationHistorySourceContext, SequenceIdentity,
    },
    results::observation::{
        ObservationFieldCatalog, ObservationHistory, ObservationHistoryPublication,
        ObservationLimits, ObservationPublicationCatalog, ObservationPublicationEntry,
        SeriesReductionPayload,
    },
    study::plan_thermal_study,
};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::FilesystemObjectStore;
use runmat_meshing_core::ElementOrder;

pub(crate) fn histories(partial_access: PartialAccessPolicy) -> Vec<ObservationRequest> {
    [
        (
            "temperature",
            ObservationQuantity::Temperature,
            FieldLocation::Node,
        ),
        (
            "gradient",
            ObservationQuantity::TemperatureGradient,
            FieldLocation::Element,
        ),
        (
            "flux",
            ObservationQuantity::HeatFlux,
            FieldLocation::Element,
        ),
        (
            "storage",
            ObservationQuantity::ThermalEnergyStorageRateDensity,
            FieldLocation::Element,
        ),
    ]
    .into_iter()
    .map(|(id, quantity, location)| {
        ObservationRequest::History(ObservationSpec {
            observation_id: ObservationId(id.into()),
            quantity,
            value_shape: quantity.canonical_value_shape(),
            target: ObservationTarget::WholeModel,
            source_location: location,
            output_location: location,
            recovery: RecoveryPolicy::Native,
            component: ObservationComponent::All,
            frame: CoordinateFrame::Global,
            sign_convention: SignConvention::Native,
            complex_representation: ComplexRepresentation::Real,
            sampling: SamplingPolicy::EveryAcceptedIncrement,
            reduction: (quantity == ObservationQuantity::Temperature).then_some(ReductionPolicy {
                spatial: SpatialReduction::None,
                series: SeriesReduction::Mean {
                    weighting: SeriesWeighting::TrapezoidalIndependentDomain,
                },
                extrema_ties: ExtremaTiePolicy::FirstCanonical,
            }),
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostPolicy::Automatic,
        })
    })
    .collect()
}

#[test]
fn thermal_histories_finalize_from_admitted_checkpoints_without_resolving() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for policy in [
            PartialAccessPolicy::ManifestAndChunks,
            PartialAccessPolicy::WholeArtifact,
        ] {
            check_history_chain(order, policy);
        }
    }
}

fn check_history_chain(order: ElementOrder, partial_access: PartialAccessPolicy) {
    let mut setup = preparation(order, true);
    setup.product.compiled.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 4;
    setup.product.compiled.model.sequence.steps[0].observations = histories(partial_access);
    setup.product.workload =
        plan_thermal_study(&setup.product.compiled, planning_policy()).unwrap();
    let workload = setup.product.workload.clone();
    let mesh = setup.mesh.clone();
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap();
    let step = workload.model.sequence.steps[0].clone();
    let revision = setup.program_revision.clone();
    let access = setup.artifact_access.clone();
    let limits = setup.output_policy.inventory_limits();
    let chunk_bytes = setup.output_policy.chunk_bytes();
    let output_policy = setup.output_policy;
    let solve = prepare_thermal_task(ThermalTaskPreparation {
        maximum_resident_bytes: u64::MAX,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        field_source: None,
        authored_study_digest: Digest::sha256("thermal-native-history-finalization"),
        frozen_project_digest: None,
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        workload: setup.product.workload,
        mesh: setup.mesh,
        output_policy: setup.output_policy,
        required_capabilities: setup.required_capabilities,
    })
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store =
        FilesystemObjectStore::open(directory.path().join("objects"), limits.max_object_bytes)
            .unwrap();
    // Exactly one numerical solve; all following tasks consume its published field partitions.
    let (response, solve_accounting) = execute(
        &mut store,
        &solve.host,
        &solve.program,
        &solve.resource_plan,
        0,
    );
    let outputs = import_analysis_output_set(&store, response.clone(), &solve.host).unwrap();
    let completion =
        import_thermal_completion(&store, response, solve_accounting, &solve, output_policy)
            .unwrap();
    let result = completion.result;
    let mut attempts = Vec::new();
    let output = outputs
        .manifest
        .outputs
        .iter()
        .find(|output| output.kind == Kind::SolverResult)
        .unwrap()
        .clone();
    let input = |kind| {
        solve
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .unwrap()
            .clone()
    };
    let mut checkpoints: std::collections::BTreeMap<
        String,
        ImportedObservationHistoryCheckpointTask,
    > = Default::default();
    let mut terminal_tasks = std::collections::BTreeMap::new();
    let mut source_catalog = None;
    let mut source_plan = None;
    let mut ordinal = 1;
    for (sample_ordinal, fields) in result.manifest.fields.iter().enumerate() {
        let fanout = prepare_observation_fanout(
            &store,
            ObservationPreparation {
                source_host: solve.host.clone(),
                source: ObservationPublishedSource::Thermal {
                    manifest: result.manifest.clone(),
                    output: output.clone(),
                    access: access.clone(),
                    sample_ordinal,
                },
                source_workload: ObservationWorkloadSource::Thermal(input(Kind::ThermalWorkload)),
                mesh: Some(input(Kind::Mesh)),
                accelerator_requests: Vec::new(),
                sequence,
                step: step.clone(),
                limits: ObservationLimits::default(),
                program_revision: revision.clone(),
                chunk_bytes,
                inventory_limits: limits,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            },
            fields[0].sample,
            None,
        )
        .unwrap();
        source_plan = Some(fanout.source_request.plan.clone());
        assert_eq!(fanout.partitions.len(), 4);
        for partition in &fanout.partitions {
            source_catalog = Some(
                partition
                    .inputs
                    .iter()
                    .find(|item| item.manifest.kind == Kind::ObservationFieldCatalog)
                    .unwrap()
                    .clone(),
            );
            let (response, accounting) = execute(
                &mut store,
                &partition.host,
                &partition.program,
                &partition.resource_plan,
                ordinal,
            );
            attempts.extend(
                accounting
                    .resource_graph(&partition.resource_plan)
                    .unwrap()
                    .attempts,
            );
            ordinal += 1;
            let contribution =
                import_observation_partition_contribution(&store, response, partition, limits)
                    .unwrap();
            let reduced = prepare_observation_reduction(ObservationReductionPreparation {
                source_host: solve.host.clone(),
                source_request: fanout.source_request.clone(),
                partition_plan: partition.partition_plan.clone(),
                contributions: vec![contribution],
                program_revision: revision.clone(),
                chunk_bytes,
                inventory_limits: limits,
            })
            .unwrap();
            let (response, accounting) = execute(
                &mut store,
                &reduced.host,
                &reduced.program,
                &reduced.resource_plan,
                ordinal,
            );
            attempts.extend(
                accounting
                    .resource_graph(&reduced.resource_plan)
                    .unwrap()
                    .attempts,
            );
            ordinal += 1;
            let sample = import_reduced_observation(&store, response, &reduced, limits).unwrap();
            let id = sample.result.observation_id.clone();
            let expected = if sample.result.quantity == ObservationQuantity::Temperature {
                293.15
            } else {
                0.0
            };
            let tolerance =
                if sample.result.quantity == ObservationQuantity::ThermalEnergyStorageRateDensity {
                    1e-5
                } else {
                    1e-8
                };
            for value in sample.result.field.as_host_f64().unwrap() {
                assert!(
                    (value - expected).abs() < tolerance,
                    "{order:?} {:?}: {value}",
                    sample.result.quantity
                );
            }
            let checkpoint_task = prepare_observation_history_checkpoint_task(
                ObservationHistoryCheckpointPreparation {
                    source_host: reduced.host.clone(),
                    plan: fanout.source_request.plan.clone(),
                    observation_id: id.clone(),
                    sample,
                    previous: checkpoints
                        .get(&id.0)
                        .map(|previous| previous.objects.clone()),
                    program_revision: revision.clone(),
                    chunk_bytes,
                    inventory_limits: limits,
                },
            )
            .unwrap();
            let (response, accounting) = execute(
                &mut store,
                &checkpoint_task.host,
                &checkpoint_task.program,
                &checkpoint_task.resource_plan,
                ordinal,
            );
            attempts.extend(
                accounting
                    .resource_graph(&checkpoint_task.resource_plan)
                    .unwrap()
                    .attempts,
            );
            ordinal += 1;
            let checkpoint = import_observation_history_checkpoint_task(
                &store,
                response,
                &checkpoint_task,
                limits,
            )
            .unwrap();
            checkpoints.insert(id.0.clone(), checkpoint);
            terminal_tasks.insert(id.0, checkpoint_task);
        }
    }
    let catalog_objects = source_catalog.unwrap();
    let catalog =
        ObservationFieldCatalog::from_canonical_bytes(&catalog_objects.payload()).unwrap();
    let plan = source_plan.unwrap();
    let context = ObservationHistorySourceContext::Thermal {
        workload: &workload,
        mesh: &mesh,
    };
    let mut publications = Vec::new();
    for (id, terminal) in terminal_tasks {
        let checkpoint = &checkpoints[&id];
        let request = ObservationHistoryFinalizationRequest::new(
            plan.clone(),
            terminal.host.workload.step,
            ObservationId(id.clone()),
            &checkpoint.checkpoint,
            &catalog,
            ObservationHistorySourceContext::Thermal {
                workload: &workload,
                mesh: &mesh,
            },
        )
        .unwrap();
        let preparation = || ObservationHistoryFinalizationPreparation {
            source_host: terminal.host.clone(),
            request: request.clone(),
            checkpoint: checkpoint.objects.clone(),
            catalog: catalog_objects.clone(),
            source: ObservationHistoryFinalizationSource::Thermal {
                workload: input(Kind::ThermalWorkload),
                mesh: input(Kind::Mesh),
            },
            program_revision: revision.clone(),
            chunk_bytes,
            inventory_limits: limits,
        };
        let mut foreign = preparation();
        foreign.catalog = input(Kind::Mesh);
        assert!(prepare_observation_history_finalization_task(foreign).is_err());
        let mut missing = preparation();
        missing.source_host.workload.outputs.clear();
        assert!(prepare_observation_history_finalization_task(missing).is_err());
        let other = checkpoints
            .iter()
            .find(|(other, _)| **other != id)
            .unwrap()
            .1;
        let mut foreign = preparation();
        foreign.checkpoint = other.objects.clone();
        assert!(prepare_observation_history_finalization_task(foreign).is_err());
        let finalization = prepare_observation_history_finalization_task(preparation()).unwrap();
        assert_eq!(
            finalization.host.workload.kind,
            AnalysisWorkKind::FinalizeObservationHistory
        );
        // A valid checkpoint resource plan is still foreign to finalization admission.
        assert!(crate::analysis_task::prepare_analysis_task(
            runmat_execution::ExecutionScopeId::derive(&[b"history-finalization-negative"]),
            runmat_execution::PoolId::derive(&[b"history-finalization-negative"]),
            &finalization.host,
            &finalization.program,
            Vec::new(),
            Default::default(),
            crate::analysis_task::AnalysisTaskPolicy {
                resources: terminal.resource_plan.request.clone(),
                resource_plan: Some(&terminal.resource_plan),
                prepared_closure: None,
                retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: None,
                priority: 0,
                permitted_trust_tiers: crate::program_host::local_trust(),
            },
        )
        .is_err());
        let (response, accounting) = execute(
            &mut store,
            &finalization.host,
            &finalization.program,
            &finalization.resource_plan,
            ordinal,
        );
        attempts.extend(
            accounting
                .resource_graph(&finalization.resource_plan)
                .unwrap()
                .attempts,
        );
        ordinal += 1;
        let finalized =
            import_observation_history_finalization_task(&store, response, &finalization, limits)
                .unwrap();
        assert_eq!(finalized.history.manifest.observed_sample_count, 4);
        assert_eq!(finalized.history.manifest.observation_id.0, id);
        // Read back the actual published outputs, not a history reconstructed by rerunning
        // thermal physics. Both storage policies must converge on the same semantic contract.
        let samples = finalized
            .history
            .manifest
            .samples
            .iter()
            .map(|entry| {
                import_observation_history_sample(
                    &store,
                    &finalized.history,
                    entry.coordinate,
                    limits,
                )
                .unwrap()
            })
            .collect();
        let history = ObservationHistory::from_manifest_and_samples(
            finalized.history.manifest.clone(),
            samples,
        )
        .unwrap();
        let publication = ObservationHistoryPublication::new(
            request,
            &checkpoint.checkpoint,
            &catalog,
            context,
            &history,
            finalized.series_reduction.as_ref(),
        )
        .unwrap();
        publications.push(ObservationPublicationEntry::History(Box::new(publication)));
        if id == "temperature" {
            let series = finalized.series_reduction.unwrap();
            assert_eq!(
                series.operation,
                SeriesReduction::Mean {
                    weighting: SeriesWeighting::TrapezoidalIndependentDomain
                }
            );
            let SeriesReductionPayload::Values { field, .. } = series.payload else {
                panic!("mean must produce values")
            };
            for value in field.as_host_f64().unwrap() {
                assert!((value - 293.15).abs() < 1e-8);
            }
            assert!(finalized.series_objects.is_some());
        } else {
            assert!(finalized.series_reduction.is_none());
            assert!(finalized.series_objects.is_none());
        }
    }
    let producer = solve.host.workload.step;
    let complete = ObservationPublicationCatalog::new(
        producer,
        plan.clone(),
        catalog.clone(),
        &step,
        publications.clone(),
        context,
    )
    .unwrap();
    assert_eq!(complete.entries().len(), 4);
    let decoded =
        ObservationPublicationCatalog::from_canonical_bytes(&complete.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(decoded, complete);
    decoded.validate_for(&step, context).unwrap();

    let accuracy_preparation = || AccuracyPreparation {
        solve_host: solve.host.clone(),
        source_kind: runmat_analysis_fea::execution::AccuracySourceKind::Thermal,
        source_numerical_receipt: result.numerical_receipt_objects.clone(),
        source_observations: Some(complete.clone()),
        source_context_inputs: solve
            .inputs
            .iter()
            .filter(|input| matches!(input.manifest.kind, Kind::ThermalWorkload | Kind::Mesh))
            .cloned()
            .collect(),
        sequence,
        step: step.clone(),
        program_revision: revision.clone(),
        chunk_bytes,
        inventory_limits: limits,
    };
    let mut missing = accuracy_preparation();
    missing.source_observations = None;
    assert!(prepare_accuracy(missing).is_err());
    let mut different_declarations = accuracy_preparation();
    different_declarations.step.observations.pop();
    assert!(prepare_accuracy(different_declarations).is_err());
    let accuracy = prepare_accuracy(accuracy_preparation()).unwrap();
    assert_worker_rejects_omitted_authored_observations(&accuracy, output_policy);
    source_budget::reject_understated_constructor(&accuracy, output_policy);
    let (response, accounting) = execute(
        &mut store,
        &accuracy.host,
        &accuracy.program,
        &accuracy.resource_plan,
        ordinal,
    );
    ordinal += 1;
    attempts.extend(
        accounting
            .resource_graph(&accuracy.resource_plan)
            .unwrap()
            .attempts,
    );
    let accuracy_result = import_accuracy(&store, response, &accuracy, limits).unwrap();
    assert_eq!(
        accuracy_result.artifact.source_observations,
        Some(complete.identity())
    );
    let baseline = extend_resource_usage(
        &completion.solve_usage,
        &PlannedResourceUsageGraph::new(attempts.clone(), ordinal as u64).unwrap(),
        &access,
        output_policy,
    )
    .unwrap();
    assert_eq!(baseline.evidence.attempts.len(), ordinal);
    let publication_preparation = |usage: &PreparedResourceUsage| ResultPublicationPreparation {
        accuracy_host: accuracy.host.clone(),
        source_accuracy: accuracy_result.objects.clone(),
        source_resource_usage: usage.objects.clone(),
        source_observations: Some(complete.clone()),
        program_revision: revision.clone(),
        chunk_bytes,
        inventory_limits: limits,
    };
    let mut accepted_omissions = Vec::new();
    // Partition/reduction omissions were reproduced separately and remain open until their
    // per-task source provenance is retained; checkpoint lineage is independently derivable.
    for omitted_phase in [
        PublicationPhase::Accuracy,
        PublicationPhase::Finalization,
        PublicationPhase::Checkpoint,
    ] {
        let omitted = attempts
            .iter()
            .filter(|attempt| {
                publication_phase_category(&attempt.plan.input.phase) == Some(omitted_phase)
            })
            .collect::<Vec<_>>();
        assert!(
            !omitted.is_empty(),
            "fixture must execute {omitted_phase:?}"
        );
        assert!(
            omitted
                .iter()
                .all(|attempt| attempt.plan.request.accelerators.is_empty()),
            "{omitted_phase:?} omission must exercise CPU accounting, not accelerator reconciliation"
        );
        // Rebuild internally valid identities, plans and totals after deleting the phase.
        // Rejecting stale hashes would not establish expected-workflow completeness.
        let incomplete = extend_resource_usage(
            &completion.solve_usage,
            &PlannedResourceUsageGraph::new(
                attempts
                    .iter()
                    .filter(|attempt| {
                        publication_phase_category(&attempt.plan.input.phase) != Some(omitted_phase)
                    })
                    .cloned()
                    .collect(),
                ordinal as u64,
            )
            .unwrap(),
            &access,
            output_policy,
        )
        .unwrap();
        incomplete.evidence.validate_identity().unwrap();
        assert_eq!(
            incomplete.evidence.attempts.len() + omitted.len(),
            baseline.evidence.attempts.len()
        );
        if matches!(
            omitted_phase,
            PublicationPhase::Accuracy
                | PublicationPhase::Checkpoint
                | PublicationPhase::Finalization
        ) {
            // Exercise the core constructor independently of bridge-side preparation, using
            // authentic accuracy/catalog artifacts and newly hashed incomplete usage.
            assert!(
                runmat_analysis_fea::execution::ResultPublicationRequest::new(
                    &accuracy_result.artifact,
                    Some(&complete),
                    &incomplete.evidence,
                )
                .is_err()
            );
        }
        if prepare_result_publication(publication_preparation(&incomplete)).is_ok() {
            accepted_omissions.push(omitted_phase);
        }
    }
    assert!(
        accepted_omissions.is_empty(),
        "publication accepted missing CPU accounting: {accepted_omissions:?}"
    );
    // Requiring merely one checkpoint per history is insufficient: every prefix has work.
    let checkpoint_attempts: Vec<_> = attempts
        .iter()
        .filter(|attempt| {
            publication_phase_category(&attempt.plan.input.phase)
                == Some(PublicationPhase::Checkpoint)
        })
        .collect();
    for index in [
        0,
        checkpoint_attempts.len() / 2,
        checkpoint_attempts.len() - 1,
    ] {
        let omitted_id = checkpoint_attempts[index].accounting.attempt_id;
        let incomplete = extend_resource_usage(
            &completion.solve_usage,
            &PlannedResourceUsageGraph::new(
                attempts
                    .iter()
                    .filter(|attempt| attempt.accounting.attempt_id != omitted_id)
                    .cloned()
                    .collect(),
                ordinal as u64,
            )
            .unwrap(),
            &access,
            output_policy,
        )
        .unwrap();
        assert!(
            runmat_analysis_fea::execution::ResultPublicationRequest::new(
                &accuracy_result.artifact,
                Some(&complete),
                &incomplete.evidence,
            )
            .is_err(),
            "missing individual checkpoint {index}"
        );
    }
    // A wrong-source demand must not hide next to the correctly accounted task.
    let mut conflict = (*checkpoint_attempts[0]).clone();
    let mut input = conflict.plan.input.clone();
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::StructuralObservationHistoryCheckpoint(demand) = &mut input.demand else { unreachable!() };
    demand.topology_digest = Digest::sha256("foreign checkpoint topology");
    conflict.plan = runmat_analysis_fea::planning::resource::build_resource_plan(input).unwrap();
    conflict.accounting.attempt_id =
        runmat_execution::identity::AttemptId::derive(&[b"conflicting checkpoint"]);
    let mut conflicting_attempts = attempts.clone();
    conflicting_attempts.push(conflict);
    let conflicting_usage = extend_resource_usage(
        &completion.solve_usage,
        &PlannedResourceUsageGraph::new(conflicting_attempts, ordinal as u64).unwrap(),
        &access,
        output_policy,
    )
    .unwrap();
    assert!(
        runmat_analysis_fea::execution::ResultPublicationRequest::new(
            &accuracy_result.artifact,
            Some(&complete),
            &conflicting_usage.evidence,
        )
        .is_err()
    );
    let publication = prepare_result_publication(publication_preparation(&baseline)).unwrap();
    let (response, accounting) = execute(
        &mut store,
        &publication.host,
        &publication.program,
        &publication.resource_plan,
        ordinal,
    );
    let published = import_published_result(&store, response, &publication, limits).unwrap();
    assert_eq!(
        published.result.source_observations,
        Some(complete.identity())
    );
    let terminal = complete_result_publication_resource_usage(
        &publication,
        &published,
        &baseline,
        &accounting,
        output_policy,
    )
    .unwrap();
    assert_eq!(
        terminal.evidence.attempts.len(),
        baseline.evidence.attempts.len() + 1
    );
    assert_eq!(
        published.result.source_resource_usage,
        baseline.evidence.identity()
    );

    // A completed solve cannot claim publication while silently dropping an authored history.
    publications.pop();
    assert!(ObservationPublicationCatalog::new(
        producer,
        plan,
        catalog,
        &step,
        publications,
        context,
    )
    .is_err());
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublicationPhase {
    Accuracy,
    Finalization,
    Checkpoint,
}

fn publication_phase_category(
    phase: &runmat_analysis_fea::planning::resource::FeaResourcePhase,
) -> Option<PublicationPhase> {
    use runmat_analysis_fea::planning::resource::FeaResourcePhase;
    match phase {
        FeaResourcePhase::AccuracyEvaluation(_) => Some(PublicationPhase::Accuracy),
        FeaResourcePhase::ObservationHistoryFinalization(_) => Some(PublicationPhase::Finalization),
        FeaResourcePhase::StructuralObservationHistoryCheckpoint(_) => {
            Some(PublicationPhase::Checkpoint)
        }
        _ => None,
    }
}

fn assert_worker_rejects_omitted_authored_observations(
    accuracy: &PreparedAccuracy,
    output_policy: AnalysisOutputPolicy,
) {
    use runmat_analysis_fea::execution::AccuracyEvaluationRequest;
    let source = &accuracy.request;
    // Forge a self-consistent, freshly hashed request with no catalog. Step identity alone
    // cannot distinguish this from the original authored request.
    let omitted = AccuracyEvaluationRequest::new(
        source.producer,
        source.source_kind,
        source.source_result,
        None,
        source.plan.clone(),
    )
    .unwrap();
    let mut inputs: Vec<_> = accuracy
        .inputs
        .iter()
        .filter(|input| {
            !matches!(
                input.manifest.kind,
                Kind::ObservationCatalog | Kind::AccuracyEvaluationRequest
            )
        })
        .cloned()
        .collect();
    inputs.push(
        prepare_analysis_objects(
            Kind::AccuracyEvaluationRequest,
            omitted.identity().digest(),
            &omitted.canonical_bytes().unwrap(),
            output_policy.chunk_bytes(),
            runmat_execution_artifact::ObjectNamespace::InputValue,
            output_policy.inventory_limits(),
        )
        .unwrap(),
    );
    // Recompute admission as well: this must still fail on independent authored authority,
    // not merely because the changed input inventory invalidates the original resource plan.
    let replan = |inputs: &mut Vec<runmat_analysis_execution::PreparedAnalysisObjects>| {
        let receipt = inputs
            .iter()
            .find(|input| input.manifest.kind == Kind::NumericalReceipt)
            .unwrap();
        let receipt = runmat_analysis_fea::execution::AccuracySourceReceipt::from_canonical_bytes(
            omitted.source_kind,
            &receipt.payload(),
        )
        .unwrap();
        let plan = runmat_analysis_execution::build_prepared_accuracy_resource_plan(
            &omitted, &receipt, inputs,
        )
        .unwrap();
        let index = inputs
            .iter()
            .position(|input| input.manifest.kind == Kind::ResourcePlan)
            .unwrap();
        inputs[index] = prepare_analysis_objects(
            Kind::ResourcePlan,
            plan.identity.digest(),
            &plan.canonical_bytes().unwrap(),
            output_policy.chunk_bytes(),
            runmat_execution_artifact::ObjectNamespace::InputValue,
            output_policy.inventory_limits(),
        )
        .unwrap();
    };
    replan(&mut inputs);
    let error = crate::NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy,
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &accuracy.host,
            inputs: &inputs,
        })
        .unwrap_err();
    assert!(
        error.to_string().contains("authored requirements"),
        "{error}"
    );
    // Removing the independent source context is not a way to bypass that check.
    inputs.retain(|input| !matches!(input.manifest.kind, Kind::ThermalWorkload | Kind::Mesh));
    replan(&mut inputs);
    let error = crate::NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy,
            kind: AnalysisWorkKind::EvaluateAccuracy,
            host: &accuracy.host,
            inputs: &inputs,
        })
        .unwrap_err();
    assert!(
        error.to_string().contains("original workload and mesh"),
        "{error}"
    );
}
