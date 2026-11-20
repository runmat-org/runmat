//! Publication memory is admitted with the plan: a study that cannot publish is refused before
//! its solve, and the planned requirement covers the memory publication actually holds.
use super::standalone_publication::{standalone_preparation, standalone_preparation_with};
use super::*;
use runmat_analysis_execution::import_thermal_study_completion;
use runmat_analysis_fea::planning::resource::ResourcePlanningError;
use runmat_runtime::analysis::{
    publish_thermal_study_result, thermal_publication_retained_bytes,
    AnalysisStudyCompletionIdentity, AnalysisThermalRunOptions, PrecisionMode, QualityPolicy,
};
use runmat_runtime::operations::OperationContext;

fn policy(publication_memory_bytes: u64) -> crate::ThermalStudyExecutionPolicy {
    crate::ThermalStudyExecutionPolicy {
        publication_memory_bytes,
        resources: ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 64 * 1024 * 1024,
            scratch_bytes: 0,
            max_wall_millis: 60_000,
            max_artifact_bytes: 64 * 1024 * 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: Default::default(),
        },
        retry: RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    }
}

/// The root wire's storage as the publication source charges it (`owned`).
fn root_wire_bytes(root: &runmat_analysis_execution::PreparedThermalStudyRoot) -> u64 {
    let codec = runmat_canonical_codec::CanonicalLimits::content_bounded(usize::MAX, 65_536, 64);
    let catalog = runmat_canonical_codec::profile_serialized(&root.catalog, codec)
        .unwrap()
        .materialization_resident_bytes_estimate;
    root.objects.iter().fold(catalog, |total, object| {
        total + object.bytes.capacity() as u64
    })
}

#[test]
fn thermal_publication_is_admitted_with_the_plan() {
    check(
        |allocation| standalone_preparation(ElementOrder::Tet10, false, allocation),
        true,
    );
}

/// The transient study shape of a warm-up example: 201 samples of every recovered field,
/// terminal temperature states and a final temperature field observation. Each of those
/// single-sample products is charged as one sample of the result plus its entity identities,
/// not as a further copy of the whole result, and the plan still covers what publication holds.
#[test]
fn transient_publication_charges_single_sample_products_as_one_sample() {
    let plan = |field: bool| {
        let preparation = standalone_preparation_with(
            ElementOrder::Tet10,
            Some((200, false, field)),
            "thermal-publication-transient-plan",
        );
        let entities = preparation
            .mesh
            .topology
            .nodes
            .len()
            .max(preparation.mesh.topology.volume_elements.len()) as u64;
        let mut coordinator =
            runmat_analysis_execution::ThermalStudyCoordinator::new(preparation).unwrap();
        let step = coordinator.selected_step();
        assert!(!step.state_outputs.is_empty());
        assert_eq!(step.observations.len(), usize::from(field));
        let task = coordinator.prepare_solve().unwrap();
        let runmat_analysis_fea::planning::resource::FeaResourceDemand::ThermalSolve(demand) =
            &task.resource_plan.input.demand
        else {
            panic!("thermal solve demand");
        };
        assert_eq!(demand.sample_count.0, 201);
        let result = demand.result_output.transport_closure_bytes_upper_bound.0;
        let sample = task.sample_output_bytes;
        assert!(
            sample < result,
            "one sample {sample} of the result {result}"
        );
        assert!(task.result_import_bytes > 3 * result);
        let required = coordinator.planned_publication_bytes().unwrap();
        (required, result, sample, entities)
    };
    let (without, result, sample, entities) = plan(false);
    let (with, ..) = plan(true);
    eprintln!(
        "transient publication plan {with} bytes ({without} without the field observation); \
         result bound {result}, one sample {sample}"
    );
    // One product is held typed by the projection and its copy (2 x 25, the codec's per-byte
    // typed decode bound), encoded and typed by the completion and its replay (2 x 26), and
    // imported at three transport copies plus its typed decode (28): 130 times its encoding.
    // Its phase adds four fixed contract allowances to the projections, completions and import.
    let added = with - without;
    let identities = entities * 128;
    let phase = 5 * 4 * runmat_canonical_codec::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT;
    assert!(
        added >= 130 * sample && added <= 130 * (sample + identities) + phase,
        "a field observation added {added} bytes; one sample is {sample}, the result {result}"
    );
    // The runtime's default staging budget in tests is below this study's import workspace,
    // so the plan's coverage of the source phase is checked without publishing.
    check(
        |allocation| {
            standalone_preparation_with(ElementOrder::Tet10, Some((200, false, true)), allocation)
        },
        false,
    );
}

fn check(
    preparation: impl Fn(&str) -> runmat_analysis_execution::ThermalStudyPreparation,
    publish: bool,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let allocation = format!(
        "thermal-publication-admission-{}",
        directory.path().display()
    );
    let execute = |publication_memory_bytes: u64, pool: &mut RecordingPool| {
        runtime.block_on(Box::pin(crate::analysis_thermal::execute_with_pool(
            pool,
            preparation(&allocation),
            policy(publication_memory_bytes),
        )))
    };

    // (a) A publication that cannot fit is refused at planning, with the typed resource error
    // naming its requirement, and no task (no solve) runs.
    let mut refused_pool = RecordingPool::new(directory.path());
    let Err(crate::NativeExecutionError::Resource(ResourcePlanningError::AccuracyCeiling {
        resource,
        required,
        maximum,
    })) = execute(1, &mut refused_pool)
    else {
        panic!("a publication above the admitted memory must be refused at planning");
    };
    assert_eq!(resource, "thermal result publication");
    assert_eq!(maximum, 1);
    assert!(required > 1);
    assert!(
        refused_pool.tasks.is_empty(),
        "refusal must precede the solve"
    );

    // (b) The same study with exactly its planned publication memory solves and publishes.
    let mut pool = RecordingPool::new(directory.path());
    let executed = execute(required, &mut pool).unwrap();
    assert!(!pool.tasks.is_empty());
    let root = prepare_thermal_study_root(&executed.completed).unwrap();
    let owned = root_wire_bytes(&root);
    let source = RetainedThermalSource {
        inventory_limits: executed.completed.output_policy().inventory_limits(),
        catalog: root.catalog,
        completed: executed.completed,
        store: pool.store.clone(),
        objects: root
            .objects
            .into_iter()
            .map(|object| (object.descriptor.digest, object.bytes))
            .collect(),
    };
    let dto = completion(&source.completed).unwrap();

    // (c) The source reservation counts the owners that coexist: the retained completion, an
    // equal replay, and the rebuilt root wire, which equals the retained one.
    let completion_bytes = source.completed.retained_storage_bytes(u64::MAX).unwrap();
    let replay = import_thermal_study_completion(
        &source,
        source.completed.preparation(),
        source.completed.result_closures().to_vec(),
    )
    .unwrap();
    assert!(replay == source.completed);
    assert_eq!(
        replay.retained_storage_bytes(u64::MAX).unwrap(),
        completion_bytes
    );
    assert_eq!(
        root_wire_bytes(&prepare_thermal_study_root(&replay).unwrap()),
        owned
    );
    drop(replay);
    let reservation = source.publication_workspace(&dto, u64::MAX).unwrap();
    // Publication re-checks the source under the allowance it admitted (`source_replay_allowance`,
    // at least this reservation). The estimate must not depend on the maximum it is checked
    // against, so the re-check under exactly the reservation reproduces it.
    assert_eq!(
        source.publication_workspace(&dto, reservation).unwrap(),
        reservation
    );
    assert!(source.publication_workspace(&dto, reservation - 1).is_err());
    let import = reservation - owned - 2 * completion_bytes;
    assert!(
        import >= owned,
        "the import workspace term covers the rebuilt root wire"
    );
    // The retired bound charged a third completion and added the root wire on top.
    assert!(reservation < owned + 3 * completion_bytes + import);

    let mesh = source.completed.preparation().mesh;
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
    let options = AnalysisThermalRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        quality_policy: QualityPolicy::Strict,
        snapshot_count: 1,
        residual_warn_threshold: dto.workload.residual_target,
        solver_mesh_artifact_path: None,
    };
    let staging = thermal_publication_retained_bytes(&dto, &mesh, &evidence, &options).unwrap();
    let actual = staging + reservation;
    eprintln!(
        "publication plan {required} bytes; actual source phase {actual} bytes \
         (staging {staging}, root wire {owned}, completion {completion_bytes}, import {import})"
    );
    assert!(
        required >= actual,
        "planned publication {required} must cover the source phase {actual}"
    );

    if !publish {
        return;
    }
    let definition = dto.compiled.definition_digest().unwrap();
    let reader = ThermalStudyObjectSource::new(source);
    publish_thermal_study_result(
        &dto,
        &reader,
        &mesh,
        &evidence,
        &options,
        definition,
        OperationContext::new(None, None),
        AnalysisStudyCompletionIdentity::new(&allocation).unwrap(),
    )
    .unwrap();
}
