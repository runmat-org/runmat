//! Real same-pool numerical closures, cold replayed through the composed-prefix importer.
//! This isolates composition; it does not manufacture exact-CAD meshing evidence.
use super::*;
use crate::analysis_thermal_prestress::{
    execute_with_pool,
    tests::{pool::RecordingPool, preparation},
};
use runmat_execution_artifact::cache::{CacheExport, CacheImport, FilesystemObjectStore};
use runmat_meshing_core::ElementOrder;

fn policy() -> crate::StructuralSequenceExecutionPolicy {
    crate::StructuralSequenceExecutionPolicy {
        resources: runmat_execution::resource::ResourceRequest {
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
        retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    }
}

fn copy_declared(
    source: &impl CacheImport,
    target: &mut FilesystemObjectStore,
    references: &[ValueRef],
) {
    for reference in references {
        let bytes = source
            .read_verified(reference.logical_digest)
            .unwrap()
            .unwrap();
        assert_eq!(bytes.len() as u64, reference.encoded_length);
        let object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "analysis/cold-composed-test",
            &reference.media_type,
            bytes,
        )
        .unwrap();
        assert_eq!(object.descriptor.digest, reference.logical_digest);
        target.write_verified(&object).unwrap();
    }
}

#[tokio::test]
async fn all_four_thermal_prestress_closures_replay_cold_and_reject_foreign_heat() {
    let mut foreign: Option<(thermal_prefix::ThermalPrefixClosure, tempfile::TempDir)> = None;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for branch in 0..4 {
            let setup = preparation(branch, order);
            let compiled = setup.compiled.clone();
            let mesh = setup.structural.mesh.clone();
            let expected = ExactStructuralStudyResultExpectation {
                program_revision: setup.structural.program_revision.clone(),
                artifact_access: setup.structural.artifact_access.clone(),
                // Deliberately distinct from the compiled definition used by the preload.
                authored_study_digest: Digest::sha256(b"source-neutral-outer-study"),
                frozen_project_digest: setup.structural.frozen_project_digest,
                geometry_source_digest: Digest::sha256(b"source-neutral-outer-geometry"),
                workloads: setup.structural.workloads.clone(),
                local_execution: setup.structural.local_execution,
                analysis_accelerator_requests: setup.structural.accelerator_requests.clone(),
                input_materializations: Vec::new(),
                expected_meshing_request: mesh.resolved_request.clone(),
            };
            let warm_dir = tempfile::tempdir().unwrap();
            let mut pool = RecordingPool::new(warm_dir.path());
            let completed = execute_with_pool(&mut pool, setup, policy()).await.unwrap();
            let analysis = match &completed.structural {
                crate::ThermalPrestressStructuralCompletion::Buckling(value) => {
                    buckling_closure(value, &expected.artifact_access)
                }
                crate::ThermalPrestressStructuralCompletion::Modal(value) => {
                    prestressed_modal_closure(value, &expected.artifact_access)
                }
                crate::ThermalPrestressStructuralCompletion::Frequency(value) => {
                    prestressed_frequency_closure(value, &expected.artifact_access)
                }
                crate::ThermalPrestressStructuralCompletion::ModalFrequency(value) => {
                    prestressed_modal_frequency_closure(value, &expected.artifact_access)
                }
            }
            .unwrap();
            let thermal =
                runmat_analysis_execution::prepare_thermal_study_root(&completed.thermal).unwrap();
            for object in &thermal.objects {
                pool.store.write_verified(object).unwrap();
            }
            let closure = thermal_prefix::ThermalPrefixClosure {
                root: thermal.root,
                catalog: thermal.catalog,
            };
            let inventory = unique_references(
                analysis
                    .objects()
                    .iter()
                    .cloned()
                    .chain(closure.catalog.objects.iter().cloned()),
            )
            .unwrap();
            reference_inventory(&inventory).unwrap();
            let cold_dir = tempfile::tempdir().unwrap();
            let limits = crate::NativeAnalysisHostLimits::default().inventory;
            let mut cold =
                FilesystemObjectStore::open(cold_dir.path(), limits.max_object_bytes).unwrap();
            copy_declared(&pool, &mut cold, &inventory);
            let replay = |source: &FilesystemObjectStore, thermal| {
                thermal_prefix::import(thermal_prefix::ImportContext {
                    source,
                    compiled: Some(&compiled),
                    analysis: &analysis,
                    thermal,
                    expected: &expected,
                    workloads: &expected.workloads,
                    mesh: &mesh,
                    limits,
                })
            };
            let recovered = replay(&cold, Some(&closure)).unwrap().unwrap();
            assert_eq!(*recovered, completed.thermal);
            assert_eq!(recovered.states[1].state.state_id.0, "second-temperature");
            assert!(replay(&cold, None).is_err());
            // A complete, independently replayable thermal root from another actual study
            // cannot be paired with this unchanged structural closure.
            if let Some((foreign_closure, foreign_dir)) = &foreign {
                let foreign_store =
                    FilesystemObjectStore::open(foreign_dir.path(), limits.max_object_bytes)
                        .unwrap();
                copy_declared(&foreign_store, &mut cold, &foreign_closure.catalog.objects);
                runmat_analysis_execution::import_thermal_study_root(
                    &cold,
                    &foreign_closure.root,
                    foreign_closure.root.logical_digest,
                    &expected.artifact_access,
                    limits,
                )
                .unwrap();
                assert!(replay(&cold, Some(foreign_closure)).is_err());
            } else {
                foreign = Some((closure.clone(), cold_dir));
            }
        }
    }
}
