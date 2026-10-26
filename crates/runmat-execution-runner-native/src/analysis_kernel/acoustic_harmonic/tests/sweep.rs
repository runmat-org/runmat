//! Real native host admission and cold filesystem entry import, not process restart.
use super::*;
use runmat_analysis_execution::{
    execute_serial_analysis, import_acoustic_harmonic_result, AnalysisHostResponse,
};
use runmat_analysis_fea::execution::{AcousticFrequencyResult, AcousticFrequencyWorkload};
use runmat_execution_artifact::cache::FilesystemObjectStore;

#[test]
fn native_acoustic_frequency_entries_fan_in_after_cold_reopen_for_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut expected = None;
        for chunk in [1024, 8192] {
            let (mut template, mesh, _) = task(order, chunk);
            template.model.sequence.steps[0].independent_domain =
                IndependentDomain::FrequencyHertz(vec![0.1, 0.2, 0.3]);
            let sweep = AcousticFrequencyWorkload::new(template).unwrap();
            let digest = sweep.identity().unwrap().digest();
            let limits = ObjectInventoryLimits::default();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("objects");
            let mut results = Vec::new();
            let mut numerical = std::collections::BTreeSet::new();
            let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            // Deliberately reverse completion order. Each task has distinct admitted
            // numerical intent and resource accounting; this is task fan-out, not
            // a cooperative/distributed Helmholtz solve.
            for entry in sweep.plan().entries().unwrap().into_iter().rev() {
                let task = prepare_acoustic_harmonic_task(AcousticHarmonicTaskPreparation {
                    local_execution:
                        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                    authored_study_digest: digest,
                    frozen_project_digest: None,
                    program_revision: ProgramRevision::new(
                        digest,
                        digest,
                        ProgramEnvironment::new(1, 1, digest, digest, "matlab").unwrap(),
                    )
                    .unwrap(),
                    artifact_access: ArtifactAccess {
                        authorization_scope: "study:acoustic-frequency".into(),
                        encryption_context: digest,
                    },
                    workload: sweep.entry(entry).unwrap(),
                    mesh: mesh.clone(),
                    output_policy: AnalysisOutputPolicy::new(chunk, limits).unwrap(),
                    required_capabilities: Default::default(),
                })
                .unwrap();
                assert!(numerical.insert(task.host.workload.execution_plan));
                task.program.write_to(&mut store).unwrap();
                let completed = execute_serial_analysis(
                    &task.program.request,
                    &mut store,
                    &NativeAnalysisKernel,
                    chunk,
                    limits,
                )
                .unwrap();
                let response = AnalysisHostResponse::completed(&task.host, &completed)
                    .unwrap()
                    .program_response();
                drop(store);
                store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
                let imported =
                    import_acoustic_harmonic_result(&store, response, &task, limits).unwrap();
                imported
                    .result
                    .validate_for(&sweep.entry(entry).unwrap(), &mesh)
                    .unwrap();
                results.push(imported.result);
            }
            assert_eq!(numerical.len(), 3);
            let assembled =
                AcousticFrequencyResult::assemble(sweep.clone(), &mesh, results.clone()).unwrap();
            results.reverse();
            assert_eq!(
                assembled,
                AcousticFrequencyResult::assemble(sweep.clone(), &mesh, results).unwrap()
            );
            let reopened = AcousticFrequencyResult::from_canonical_bytes(
                &assembled.canonical_bytes().unwrap(),
            )
            .unwrap();
            reopened.validate_for(&sweep, &mesh).unwrap();
            assert_eq!(
                reopened
                    .results()
                    .iter()
                    .map(|r| r.frequency().canonical_value())
                    .collect::<Vec<_>>(),
                vec![0.1, 0.2, 0.3]
            );
            let identity = reopened.source_result_identity().unwrap();
            if let Some(expected) = expected {
                assert_eq!(identity, expected);
            } else {
                expected = Some(identity);
            }
        }
    }
}
