//! Analytic equilibrium checks through the shared published-field observation programs.
use super::{execute, planning_policy, preparation};
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    import_analysis_output_set, import_observation_partition_contribution,
    import_reduced_observation, import_thermal_result, prepare_observation_fanout,
    prepare_observation_reduction, prepare_thermal_task, ObservationPreparation,
    ObservationPublishedSource, ObservationReductionPreparation, ObservationWorkloadSource,
    ThermalTaskPreparation,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind, SequenceIdentity},
    results::observation::ObservationLimits,
    study::plan_thermal_study,
};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::FilesystemObjectStore;
use runmat_meshing_core::ElementOrder;

fn observations(transient: bool) -> Vec<ObservationRequest> {
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
        let spec = ObservationSpec {
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
            sampling: if transient {
                SamplingPolicy::EveryAcceptedIncrement
            } else {
                SamplingPolicy::Final
            },
            reduction: None,
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access: PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostPolicy::Automatic,
        };
        if transient {
            ObservationRequest::History(spec)
        } else {
            ObservationRequest::Field(spec)
        }
    })
    .collect()
}

#[test]
fn shared_thermal_observations_read_all_published_samples_without_another_solve() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for transient in [false, true] {
            check_equilibrium_observations(order, transient);
        }
    }
}

fn check_equilibrium_observations(order: ElementOrder, transient: bool) {
    let mut setup = preparation(order, transient);
    setup.product.compiled.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = if transient { 4 } else { 1 };
    setup.product.compiled.model.sequence.steps[0].observations = observations(transient);
    setup.product.workload =
        plan_thermal_study(&setup.product.compiled, planning_policy()).unwrap();
    let sequence =
        SequenceIdentity::from_solver_contract(&setup.product.workload.model.sequence).unwrap();
    let step = setup.product.workload.model.sequence.steps[0].clone();
    let revision = setup.program_revision.clone();
    let access = setup.artifact_access.clone();
    let limits = setup.output_policy.inventory_limits();
    let chunk_bytes = setup.output_policy.chunk_bytes();
    let solve = prepare_thermal_task(ThermalTaskPreparation {
        maximum_resident_bytes: u64::MAX,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        field_source: None,
        authored_study_digest: Digest::sha256("thermal-observation-native-test"),
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
    // This is the only numerical solve submission. Every following program consumes its fields.
    let (response, _) = execute(
        &mut store,
        &solve.host,
        &solve.program,
        &solve.resource_plan,
        0,
    );
    let outputs = import_analysis_output_set(&store, response.clone(), &solve.host).unwrap();
    let result = import_thermal_result(&store, response, &solve, limits).unwrap();
    let output = outputs
        .manifest
        .outputs
        .iter()
        .find(|output| output.kind == AnalysisNumericalObjectKind::SolverResult)
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
    let observation_preparation = |sample_ordinal| ObservationPreparation {
        source_host: solve.host.clone(),
        source: ObservationPublishedSource::Thermal {
            manifest: result.manifest.clone(),
            output: output.clone(),
            access: access.clone(),
            sample_ordinal,
        },
        source_workload: ObservationWorkloadSource::Thermal(input(
            AnalysisNumericalObjectKind::ThermalWorkload,
        )),
        mesh: Some(input(AnalysisNumericalObjectKind::Mesh)),
        accelerator_requests: Vec::new(),
        sequence,
        step: step.clone(),
        limits: ObservationLimits::default(),
        program_revision: revision.clone(),
        chunk_bytes,
        inventory_limits: limits,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
    };
    assert_eq!(result.manifest.fields.len(), if transient { 4 } else { 1 });
    assert!(prepare_observation_fanout(
        &store,
        observation_preparation(result.manifest.fields.len()),
        result.manifest.fields[0][0].sample,
        None,
    )
    .is_err());
    if transient {
        // A valid time from the same result is still foreign to a different selected sample.
        assert!(prepare_observation_fanout(
            &store,
            observation_preparation(0),
            result.manifest.fields[1][0].sample,
            None,
        )
        .is_err());
    }
    let mut ordinal = 1;
    let mut history_plan = None;
    for (sample_ordinal, fields) in result.manifest.fields.iter().enumerate() {
        let sample = fields[0].sample;
        let fanout = prepare_observation_fanout(
            &store,
            observation_preparation(sample_ordinal),
            sample,
            None,
        )
        .unwrap();
        // Snapshot display labels vary, but the scientific plan must be stable across time.
        if let Some(expected) = &history_plan {
            assert_eq!(&fanout.source_request.plan, expected);
        } else {
            history_plan = Some(fanout.source_request.plan.clone());
        }
        assert_eq!(
            fanout.partitions.len(),
            4,
            "serial policy gives one partition per quantity"
        );
        let mut observed = Vec::new();
        for partition in &fanout.partitions {
            assert_eq!(
                partition.host.workload.kind,
                runmat_analysis_fea::execution::AnalysisWorkKind::EvaluateObservationPartition
            );
            let (response, _) = execute(
                &mut store,
                &partition.host,
                &partition.program,
                &partition.resource_plan,
                ordinal,
            );
            ordinal += 1;
            let contribution =
                import_observation_partition_contribution(&store, response, partition, limits)
                    .unwrap();
            let reduction = prepare_observation_reduction(ObservationReductionPreparation {
                source_host: solve.host.clone(),
                source_request: fanout.source_request.clone(),
                partition_plan: partition.partition_plan.clone(),
                contributions: vec![contribution],
                program_revision: revision.clone(),
                chunk_bytes,
                inventory_limits: limits,
            })
            .unwrap();
            let (response, _) = execute(
                &mut store,
                &reduction.host,
                &reduction.program,
                &reduction.resource_plan,
                ordinal,
            );
            ordinal += 1;
            let observation = import_reduced_observation(&store, response, &reduction, limits)
                .unwrap()
                .result;
            assert_eq!(observation.sample, sample);
            let expected = if observation.quantity == ObservationQuantity::Temperature {
                293.15
            } else {
                0.0
            };
            let tolerance =
                if observation.quantity == ObservationQuantity::ThermalEnergyStorageRateDensity {
                    1e-5
                } else {
                    1e-8
                };
            let values = observation.field.as_host_f64().unwrap();
            assert!(!values.is_empty());
            for value in values {
                assert!((value - expected).abs() < tolerance,
                    "{order:?} transient={transient}, sample={sample:?}, quantity={:?}: {value} != {expected}", observation.quantity);
            }
            observed.push(observation.quantity);
        }
        assert_eq!(observed.len(), 4);
        for field in fields {
            assert!(observed.contains(&field.definition.quantity()));
        }
    }
}
