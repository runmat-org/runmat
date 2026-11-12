//! Publication memory is admitted with the plan: a study that cannot publish is refused before
//! its solve, and the planned requirement covers the memory publication actually holds.
use super::standalone_publication::standalone_preparation;
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
            standalone_preparation(ElementOrder::Tet10, false, &allocation),
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
