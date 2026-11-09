use super::*;
mod electrothermal;
mod initial_fields;
mod profile_rejection;
mod publication_admission;
use crate::analysis_kernel::NativeAnalysisKernel;
use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
use runmat_analysis_execution::{
    decode_prepared_thermal_manifest, prepare_thermal_task, AnalysisKernel, AnalysisOutputPolicy,
    PreparedThermalTask, ThermalTaskPreparation,
};
use runmat_analysis_fea::{
    execution::{AnalysisWorkKind, ThermalNumericalReceiptArtifact},
    results::field::RecoveredFieldPartition,
    ComputeBackend, ThermalProcedureControl,
};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn policy(chunk: usize) -> AnalysisOutputPolicy {
    AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap()
}

fn revision() -> ProgramRevision {
    let digest = Digest::sha256("thermal-native-test");
    ProgramRevision::new(
        digest,
        digest,
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

fn task(order: ElementOrder, chunk: usize) -> (ThermalWorkload, PreparedThermalTask) {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_transient("heat")]);
    let workload = ThermalWorkload {
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    };
    let digest = Digest::sha256("thermal-native-test");
    let prepared = prepare_thermal_task(ThermalTaskPreparation {
        maximum_resident_bytes: u64::MAX,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        field_source: None,
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "study:thermal".into(),
            encryption_context: digest,
        },
        workload: workload.clone(),
        mesh: canonical_tetrahedron_solver_mesh(order),
        output_policy: policy(chunk),
        required_capabilities: Default::default(),
    })
    .unwrap();
    (workload, prepared)
}

// witness: FEA-THERMAL-EXECUTION-001/publication @ thermal_transient/tet4|tet10/native_process;
//   boundary: Both-order serial native artifact import and semantic terminal state roundtrip; no
//   producer-death, remote, partitioned or GPU credit.
#[test]
fn native_thermal_serial_transport_import_and_terminal_state_round_trip() {
    use runmat_analysis_execution::{
        execute_serial_analysis, import_semantic_state, import_thermal_result,
        prepare_semantic_state_publication, AnalysisHostResponse,
        SemanticStatePublicationPreparation,
    };
    use runmat_analysis_fea::execution::SemanticStatePublicationSource;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (workload, task) = task(order, 1024);
        let limits = ObjectInventoryLimits::default();
        let directory = tempfile::tempdir().unwrap();
        let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
            directory.path().join("objects"),
            limits.max_object_bytes,
        )
        .unwrap();
        task.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            1024,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&task.host, &completed)
            .unwrap()
            .program_response();
        let imported = import_thermal_result(&store, response, &task, limits).unwrap();
        let topology_digest = imported.manifest.topology_digest;
        let publication = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: task.host,
            source_result: imported.objects,
            source_profiles: imported.state_publication_profile_objects,
            source: SemanticStatePublicationSource::ThermalPrimaryField,
            declaration: workload.model.sequence.steps[0].state_outputs[0].clone(),
            topology_digest,
            program_revision: revision(),
            chunk_bytes: 1024,
            inventory_limits: limits,
        })
        .unwrap();
        publication.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &publication.program.request,
            &mut store,
            &NativeAnalysisKernel,
            1024,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&publication.host, &completed)
            .unwrap()
            .program_response();
        let state = import_semantic_state(&store, response, &publication, limits).unwrap();
        assert_eq!(state.topology_digest, topology_digest);
        assert_eq!(state.kind, runmat_analysis_core::StateKind::Thermal);
        let fields = state.payload.fields().unwrap();
        assert_eq!(fields.len(), 1);
        assert!(fields[0]
            .as_host_f64()
            .unwrap()
            .iter()
            .all(|value| (value - 293.15).abs() < 1e-8));
    }
}

#[test]
fn native_thermal_preserves_uniform_equilibrium_and_source_bound_receipt() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut semantic_identity = None;
        for chunk in [1024, 8192] {
            let (workload, task) = task(order, chunk);
            let outputs = NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    kind: AnalysisWorkKind::SolveThermal,
                    host: &task.host,
                    inputs: &task.inputs,
                    output_policy: policy(chunk),
                })
                .unwrap();
            let mut outputs = outputs.into_iter();
            let result = outputs
                .next()
                .unwrap()
                .into_prepared(chunk, ObjectInventoryLimits::default())
                .unwrap();
            let receipt = outputs
                .next()
                .unwrap()
                .into_prepared(chunk, ObjectInventoryLimits::default())
                .unwrap();
            let profiles = outputs
                .next()
                .unwrap()
                .into_prepared(chunk, ObjectInventoryLimits::default())
                .unwrap();
            assert!(outputs.next().is_none());
            let catalog =
                runmat_analysis_fea::execution::StatePublicationProfiles::from_canonical_bytes(
                    &profiles.payload(),
                )
                .unwrap();
            assert_eq!(
                catalog,
                runmat_analysis_execution::thermal_state_publication_profiles(&workload, &result)
                    .unwrap()
            );
            let manifest = decode_prepared_thermal_manifest(&result).unwrap();
            if let Some(expected) = semantic_identity {
                assert_eq!(manifest.identity(), expected);
            } else {
                semantic_identity = Some(manifest.identity());
            }
            assert_eq!(manifest.time_points_s.len(), 3);
            for sample in &manifest.fields {
                for (field_index, field) in sample.iter().enumerate() {
                    let expected = if field_index == 0 { 293.15 } else { 0.0 };
                    for descriptor in &field.partitions {
                        let partition = RecoveredFieldPartition::from_canonical_bytes(
                            &result
                                .partition_payload(descriptor.partition.digest())
                                .unwrap(),
                        )
                        .unwrap();
                        assert!(partition
                            .storage
                            .as_host_f64()
                            .unwrap()
                            .iter()
                            .all(|value| (value - expected).abs() < 1e-8));
                    }
                }
            }
            ThermalNumericalReceiptArtifact::from_canonical_bytes(&receipt.payload())
                .unwrap()
                .validate_against(&workload, &manifest)
                .unwrap();
        }
    }
}

#[test]
fn native_thermal_rejects_substituted_identity_resource_policy_and_inventory() {
    let (_, task) = task(ElementOrder::Tet4, 1024);
    let invoke = |inputs: &[runmat_analysis_execution::PreparedAnalysisObjects], output_policy| {
        NativeAnalysisKernel.execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveThermal,
            host: &task.host,
            inputs,
            output_policy,
        })
    };
    assert!(invoke(&task.inputs, policy(8192)).is_err());
    let (_, other) = self::task(ElementOrder::Tet10, 1024);
    let mut host = task.host.clone();
    host.workload.execution_plan = other.host.workload.execution_plan;
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveThermal,
            host: &host,
            inputs: &task.inputs,
            output_policy: policy(1024),
        })
        .is_err());
    let mut inputs = task.inputs.clone();
    let index = inputs
        .iter()
        .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    inputs[index] = other
        .inputs
        .into_iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    assert!(invoke(&inputs, policy(1024)).is_err());
    for index in 0..task.inputs.len() {
        let mut inputs = task.inputs.clone();
        inputs[index].manifest.logical_identity = Digest::sha256("substituted");
        assert!(invoke(&inputs, policy(1024)).is_err());
    }
    let mut inputs = task.inputs.clone();
    inputs.push(inputs[0].clone());
    assert!(invoke(&inputs, policy(1024)).is_err());
    assert!(invoke(&task.inputs[1..], policy(1024)).is_err());
}
