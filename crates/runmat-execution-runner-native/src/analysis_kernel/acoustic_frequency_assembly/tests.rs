use super::*;
use crate::analysis_kernel::NativeAnalysisKernel;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    execute_serial_analysis, import_acoustic_frequency_assembly, import_acoustic_harmonic_result,
    prepare_acoustic_frequency_assembly_task, prepare_acoustic_harmonic_task,
    AcousticFrequencyAssemblyPreparation, AcousticHarmonicTaskPreparation, AnalysisHostResponse,
    AnalysisKernel, AnalysisOutputPolicy,
};
use runmat_analysis_fea::execution::{AcousticHarmonicWorkload, AnalysisWorkKind};
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{
    cache::FilesystemObjectStore, object::ObjectInventoryLimits, ArtifactAccess,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

struct DamagedObject<'a> {
    source: &'a FilesystemObjectStore,
    target: Digest,
    missing: bool,
}
impl runmat_execution_artifact::cache::CacheImport for DamagedObject<'_> {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        if digest == self.target {
            if self.missing {
                return Ok(None);
            }
            return Ok(Some(vec![0]));
        }
        runmat_execution_artifact::cache::CacheImport::read_verified(self.source, digest)
    }
}

fn template(frequencies: Vec<f64>) -> AcousticHarmonicWorkload {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(3.0));
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    let mut step = ProcedureStep::acoustic_harmonic("acoustic");
    step.independent_domain = IndependentDomain::FrequencyHertz(frequencies);
    model.sequence.steps = vec![step];
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                value: PrescribedValue::Constant(Pressure::new(1.0)),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "matched".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:3")),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    });
    AcousticHarmonicWorkload {
        model,
        step_id: ProcedureStepId("acoustic".into()),
        frequency: Frequency::new(0.2),
        solver: runmat_analysis_fea::physics::acoustic::AcousticHarmonicPolicy::default(),
        power_balance_relative_tolerance: 1e-8,
    }
}

// witness: FEA-ACOUSTIC-EXECUTION-001/inventory @ acoustic_harmonic/tet4|tet10/native_process;
//   boundary: Tet4/Tet10, two/three serial-native frequencies, reversed completion and cold
//   filesystem fields; no actual parallel/distributed/GPU or producer-termination claim.
#[test]
fn native_admitted_frequency_assembly_reopens_original_complete_fields_and_metadata() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for count in [2, 3] {
            let frequencies = vec![0.1, 0.2, 0.3][..count].to_vec();
            let workload = AcousticFrequencyWorkload::new(template(frequencies.clone())).unwrap();
            let digest = workload.identity().unwrap().digest();
            let revision = ProgramRevision::new(
                digest,
                digest,
                ProgramEnvironment::new(1, 1, digest, digest, "matlab").unwrap(),
            )
            .unwrap();
            let access = ArtifactAccess {
                authorization_scope: "acoustic-frequency-assembly-test".into(),
                encryption_context: digest,
            };
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let limits = ObjectInventoryLimits::default();
            let policy = AnalysisOutputPolicy::new(1024, limits).unwrap();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("objects");
            let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let mut originals = Vec::new();
            let mut objects = Vec::new();
            for entry in workload.plan().entries().unwrap().into_iter().rev() {
                let solve = prepare_acoustic_harmonic_task(AcousticHarmonicTaskPreparation {
                    local_execution:
                        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                    authored_study_digest: digest,
                    frozen_project_digest: None,
                    program_revision: revision.clone(),
                    artifact_access: access.clone(),
                    workload: workload.entry(entry).unwrap(),
                    mesh: mesh.clone(),
                    output_policy: policy,
                    required_capabilities: Default::default(),
                })
                .unwrap();
                solve.program.write_to(&mut store).unwrap();
                let executed = execute_serial_analysis(
                    &solve.program.request,
                    &mut store,
                    &NativeAnalysisKernel,
                    policy.chunk_bytes(),
                    limits,
                )
                .unwrap();
                let response = AnalysisHostResponse::completed(&solve.host, &executed)
                    .unwrap()
                    .program_response();
                let imported =
                    import_acoustic_harmonic_result(&store, response, &solve, limits).unwrap();
                originals.push(imported.result);
                objects.push(imported.objects);
            }
            let request =
                runmat_analysis_fea::execution::AcousticFrequencyAssemblyRequest::new(vec![
                    None;
                    count
                ])
                .unwrap();
            let preparation = |entries| AcousticFrequencyAssemblyPreparation {
                workload: workload.clone(),
                mesh: mesh.clone(),
                entries,
                request: request.clone(),
                artifact_access: access.clone(),
                program_revision: revision.clone(),
                output_policy: policy,
                authored_study_digest: digest,
                frozen_project_digest: None,
            };
            assert!(prepare_acoustic_frequency_assembly_task(preparation(
                objects[..count - 1].to_vec()
            ))
            .is_err());
            let mut duplicate = objects.clone();
            duplicate[1] = duplicate[0].clone();
            assert!(prepare_acoustic_frequency_assembly_task(preparation(duplicate)).is_err());
            let task =
                prepare_acoustic_frequency_assembly_task(preparation(objects.clone())).unwrap();
            assert_eq!(
                task.host.workload.kind,
                AnalysisWorkKind::AssembleAcousticFrequencyResult
            );
            assert!(task.resource_plan.request.accelerators.is_empty());
            task.program.write_to(&mut store).unwrap();
            // Worker transport chunking may differ from preparation. Import must preserve
            // its original physical field roots while verifying canonical partitions.
            let assembly_chunk = if count == 3 {
                8192
            } else {
                policy.chunk_bytes()
            };
            let executed = execute_serial_analysis(
                &task.program.request,
                &mut store,
                &NativeAnalysisKernel,
                assembly_chunk,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&task.host, &executed)
                .unwrap()
                .program_response();
            drop(store);
            let store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let imported =
                import_acoustic_frequency_assembly(&store, response.clone(), &task, limits)
                    .unwrap();
            for missing in [true, false] {
                let damaged = DamagedObject {
                    source: &store,
                    target: imported.result_objects.objects[0].descriptor.digest,
                    missing,
                };
                assert!(import_acoustic_frequency_assembly(
                    &damaged,
                    response.clone(),
                    &task,
                    limits
                )
                .is_err());
            }
            assert_eq!(
                imported
                    .result
                    .results()
                    .iter()
                    .map(|result| result.frequency().canonical_value())
                    .collect::<Vec<_>>(),
                frequencies
            );
            imported
                .publication
                .validate_for(&imported.result.publication_authority().unwrap())
                .unwrap();
            // Fields are recovered and published by each frequency entry's own task; the fan-in
            // retains only the assembled result and its publication manifest.
            for (result, sample) in imported
                .result
                .results()
                .iter()
                .zip(imported.publication.samples())
            {
                assert!(originals.contains(result));
                assert_eq!(
                    sample.recovery_request.source.solver_result(),
                    Some(result.source_result_identity().unwrap())
                );
            }
            let mut ordered = objects;
            ordered.reverse();
            let ordered_task =
                prepare_acoustic_frequency_assembly_task(preparation(ordered)).unwrap();
            assert_eq!(task.resource_plan, ordered_task.resource_plan);
            let mut tampered = task.inputs.clone();
            let entry = tampered
                .iter_mut()
                .find(|object| object.manifest.kind == AnalysisNumericalObjectKind::SolverResult)
                .unwrap();
            entry.manifest.logical_identity = Digest::sha256("foreign");
            assert!(NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    kind: task.host.workload.kind,
                    host: &task.host,
                    inputs: &tampered,
                    output_policy: policy
                })
                .is_err());
            let mut resource_tamper = task.inputs.clone();
            resource_tamper
                .iter_mut()
                .find(|object| object.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
                .unwrap()
                .manifest
                .logical_identity = Digest::sha256("wrong-plan");
            assert!(NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    kind: task.host.workload.kind,
                    host: &task.host,
                    inputs: &resource_tamper,
                    output_policy: policy
                })
                .is_err());
        }
    }
}
