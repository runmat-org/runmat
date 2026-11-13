//! Shared scheduled-study witness: admission, cold original fields, receipts and publication.
use super::fluid_support as support;
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    import_fluid_study_completion, AnalysisOutputPolicy, FluidStudyPreparation,
};
use runmat_analysis_fea::{
    physics::cfd::canonical::qualification::fixture::{geometry_policy, Fixture},
    properties::compile_property_map,
    results::field::{reconstruct_recovered_field, RecoveredFieldPartition},
    study::{plan_fluid_study_product, CompiledAnalysisStudy, FluidStudyPlanningPolicy},
};
use runmat_execution_artifact::cache::FilesystemObjectStore;
use runmat_execution_runner_native::{execute_fluid_study_locally, FluidStudyExecutionPolicy};

pub(super) async fn cold_fixture(
    fixture: &Fixture,
    identity: &str,
) -> runmat_analysis_execution::CompletedFluidStudy {
    let (base, _) = support::workload(false, 0.0);
    let mut model = base.model;
    model.cfd = Some(fixture.material.clone());
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(
        fixture.material.density.canonical_value(),
    ));
    model.materials[0].fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: fixture.material.dynamic_viscosity,
    });
    model.constraints = fixture
        .constraints
        .iter()
        .cloned()
        .map(AnalysisConstraint::Fluid)
        .collect();
    model.boundaries = fixture.operators.clone();
    model.signals = fixture.signals.clone();
    model.excitations = fixture.actions.clone();
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId(format!("{identity}-sources")),
        action_ids: model
            .excitations
            .iter()
            .map(|a| a.excitation().excitation_id.clone())
            .collect(),
        activation: ExcitationCaseActivation::Always,
    }];
    // Independent matrix evidence does not become an arbitrary authored spatial certificate.
    model.sequence.steps[0]
        .accuracy
        .criteria
        .push(AccuracyCriterion {
            criterion_id: "mandatory-spatial".into(),
            target: AccuracyTarget::NumericalResult,
            contribution: ErrorContributionKind::SpatialDiscretization,
            evidence: RequiredAccuracyEvidence::ConvergenceStudy,
            tolerance: AccuracyTolerance::Relative(0.35),
            mandatory: true,
        });
    let ceiling = model.sequence.steps[0]
        .accuracy
        .resources
        .resident_byte_budget();
    let policy = FluidStudyPlanningPolicy {
        numerical: fixture.policy,
        geometry: geometry_policy(),
    };
    let compiled = CompiledAnalysisStudy {
        module_path: format!("qualification/{identity}.fea"),
        study_id: identity.into(),
        study_inputs: Vec::new(),
        outputs: Vec::new(),
        assessments: Vec::new(),
        properties: compile_property_map(&model, Some(&fixture.mesh)).unwrap(),
        model,
        solver_mesh_digest: fixture.mesh.canonical_digest,
    };
    let product = plan_fluid_study_product(compiled, &fixture.mesh, policy).unwrap();
    let mut planned_policy = fixture.policy;
    planned_policy.physical_relative_tolerance = 1e-8;
    assert_eq!(product.workload().policy, planned_policy);
    let directory = tempfile::tempdir().unwrap();
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().join(identity);
    config.max_workers = 1;
    config.worker_cpu_millicores = 1000;
    config.worker_memory_bytes = ceiling.min(fixture.policy.resident_byte_budget());
    config.worker_scratch_bytes = config.worker_memory_bytes;
    config.worker_capabilities = capabilities();
    let session = NativeProgramSession::new(config).unwrap();
    let _source = session.object_source();
    let store_root = session.object_store().root().to_path_buf();
    let resources = ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: ceiling.min(fixture.policy.resident_byte_budget()),
        scratch_bytes: ceiling.min(fixture.policy.resident_byte_budget()),
        max_wall_millis: product.compiled().model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_wall_time_milliseconds,
        max_artifact_bytes: ObjectInventoryLimits::default().max_total_bytes,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    };
    let preparation = FluidStudyPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        policy,
        product,
        mesh: fixture.mesh.clone(),
        frozen_project_digest: None,
        artifact_access: session.artifact_access().clone(),
        program_revision: revision(),
        output_policy: AnalysisOutputPolicy::new(8192, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    };
    let done = execute_fluid_study_locally(
        session,
        preparation,
        FluidStudyExecutionPolicy {
            resources,
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .await
    .unwrap();
    let preparation = done.preparation();
    let closures = done.result_closures().to_vec();
    // The execution API has consumed and terminated its session. Reopen its filesystem store;
    // the public source guard retains directory lifetime, without retaining decoded fields.
    let store = FilesystemObjectStore::open(
        store_root,
        ObjectInventoryLimits::default().max_object_bytes,
    )
    .unwrap();
    {
        let mut reordered = closures.clone();
        reordered.swap(0, 1);
        assert!(import_fluid_study_completion(&store, done.preparation(), reordered).is_err());
        let mut missing = closures.clone();
        missing.pop();
        assert!(import_fluid_study_completion(&store, done.preparation(), missing).is_err());
        let mut foreign = done.preparation();
        let resident = foreign.policy.numerical.maximum_resident_bytes;
        foreign.policy.numerical.maximum_resident_bytes =
            Some(resident.map_or(1, |bytes| bytes - 1));
        assert!(import_fluid_study_completion(&store, foreign, closures.clone()).is_err());
        if fixture
            .actions
            .iter()
            .any(|a| matches!(a, ExcitationAction::MassFlow(_)))
        {
            let mut foreign = done.preparation();
            // A different balanced physical source preserves shape and requested work;
            // it still cannot inherit the original fields, receipt or publication.
            let mut compiled = foreign.product.compiled().clone();
            {
                let model = &mut compiled.model;
                for signal in &mut model.signals {
                    if let PhysicalSignal::MassFlowRate(signal) = signal {
                        let SignalRepresentation::Constant(SignalValue::Scalar(rate)) =
                            &mut signal.representation
                        else {
                            unreachable!()
                        };
                        *rate =
                            runmat_types::physical::Quantity::new(rate.canonical_value() * (1.1));
                    }
                }
            }
            foreign.product = runmat_analysis_fea::study::plan_fluid_study_product(
                compiled,
                &foreign.mesh,
                foreign.policy,
            )
            .unwrap();
            assert!(import_fluid_study_completion(&store, foreign, closures.clone()).is_err());
        }
    }
    let cold = import_fluid_study_completion(&store, preparation, closures).unwrap();
    assert_eq!(cold.result.result.original(), done.result.result.original());
    assert_eq!(cold.fields, done.fields);
    assert_eq!(cold.accuracy, done.accuracy);
    assert_eq!(cold.publication, done.publication);
    assert_eq!(cold.states, done.states);
    assert_eq!(cold.fields.manifest.fields.len(), 3);
    assert_eq!(cold.product.workload().policy, planned_policy);
    let request = cold
        .result
        .result
        .native_field_request(cold.product.workload(), &fixture.mesh)
        .unwrap();
    let expected = cold
        .result
        .result
        .recover_native_fields(cold.product.workload(), &fixture.mesh, &request)
        .unwrap();
    for (entry, original) in cold.fields.manifest.fields.iter().zip(expected.fields()) {
        let partitions = entry
            .partitions
            .iter()
            .map(|descriptor| {
                let objects = runmat_analysis_execution::import_analysis_partition(
                    &store,
                    runmat_analysis_execution::AnalysisPartitionImport {
                        root: &cold.fields.output.root,
                        expected_kind: AnalysisNumericalObjectKind::Field,
                        expected_logical_identity: cold.fields.output.logical_identity,
                        expected_payload_digest: cold.fields.output.payload_digest,
                        partition_identity: descriptor.partition.digest(),
                        access: &cold.preparation().unwrap().artifact_access,
                        limits: ObjectInventoryLimits::default(),
                    },
                )
                .unwrap();
                RecoveredFieldPartition::from_canonical_bytes(&objects.payload()).unwrap()
            })
            .collect();
        let actual = reconstruct_recovered_field(
            entry.field,
            entry.definition.clone(),
            entry.sample,
            partitions,
            entry
                .definition
                .stored_entities(cold.fields.manifest.entity_counts),
        )
        .unwrap();
        assert_eq!(&actual, original);
    }
    assert_eq!(
        cold.accuracy.artifact.evaluation().report().status,
        runmat_analysis_fea::results::accuracy::AccuracyStatus::Indeterminate
    );
    let gate = runmat_analysis_fea::results::accuracy::publication_gate(
        &cold.product.workload().model.sequence.steps[0].accuracy,
        cold.accuracy.artifact.evaluation(),
    )
    .unwrap();
    assert_eq!(
        gate.decision,
        runmat_analysis_fea::results::accuracy::PublicationDecision::Block
    );

    cold.result
        .numerical_receipt
        .validate_against(cold.product.workload(), &cold.result.result, &fixture.mesh)
        .unwrap();
    cold
}
