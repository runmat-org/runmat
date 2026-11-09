//! Native thermal-only execution of retained electrical loss; not scheduler/restart evidence.
use super::*;
use crate::analysis_kernel::electrical_static::fixture::electrical_fixture;
use runmat_analysis_fea::execution::RetainedJouleHeatSource;
use runmat_execution_artifact::cache::FilesystemObjectStore;
use runmat_meshing_core::SolverMeshArtifact;

#[path = "electrothermal/fixture.rs"]
mod fixture;
use fixture::fixture;

fn prepare(
    source: Option<RetainedJouleHeatSource>,
    workload: ThermalWorkload,
    mesh: SolverMeshArtifact,
) -> PreparedThermalTask {
    let digest = Digest::sha256("native-retained-joule-source");
    prepare_thermal_task(ThermalTaskPreparation {
        maximum_resident_bytes: u64::MAX,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        field_source: source,
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "thermal-retained-joule".into(),
            encryption_context: digest,
        },
        workload,
        mesh,
        output_policy: policy(1024),
        required_capabilities: Default::default(),
    })
    .unwrap()
}

#[test]
fn native_retained_joule_source_executes_only_thermal_and_cold_imports_both_orders() {
    use runmat_analysis_execution::{
        execute_serial_analysis, import_thermal_completion, AnalysisHostResponse,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (source, bound, mesh) = fixture(order);
        let task = prepare(Some(source), bound.clone(), mesh.clone());
        assert_eq!(task.inputs.len(), 4);
        let directory = tempfile::tempdir().unwrap();
        let object_root = directory.path().join("objects");
        let limits = ObjectInventoryLimits::default();
        let mut store = FilesystemObjectStore::open(&object_root, limits.max_object_bytes).unwrap();
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
        let (submission, _) = crate::analysis_task::prepare_analysis_task(
            runmat_execution::ExecutionScopeId::derive(&[b"retained-joule-fixture"]),
            runmat_execution::PoolId::derive(&[b"retained-joule-fixture"]),
            &task.host,
            &task.program,
            vec![],
            Default::default(),
            crate::analysis_task::AnalysisTaskPolicy {
                resources: task.resource_plan.request.clone(),
                resource_plan: Some(&task.resource_plan),
                prepared_closure: None,
                retry: runmat_execution::task::RetryPolicy::Never,
                deadline_unix_millis: None,
                priority: 0,
                permitted_trust_tiers: crate::program_host::local_trust(),
            },
        )
        .unwrap();
        let accounting =
            crate::analysis_task_pool::phase_fixture::successful_phase(submission, &response, 1);
        drop(completed);
        drop(store);
        let cold = FilesystemObjectStore::open(&object_root, limits.max_object_bytes).unwrap();
        let imported =
            import_thermal_completion(&cold, response, accounting, &task, policy(1024)).unwrap();
        let manifest = &imported.result.manifest;
        assert_eq!(manifest.source_workload, bound.identity().unwrap());
        assert!(
            (manifest.energy_balances[0].applied_power_w
                - runmat_types::physical::Quantity::new(12.0))
            .abs()
                < runmat_types::physical::Quantity::new(1e-8)
        );
        assert!(
            (manifest.energy_balances[0].prescribed_boundary_power_w
                - runmat_types::physical::Quantity::new(12.0))
            .abs()
                < runmat_types::physical::Quantity::new(1e-8)
        );
        let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
        let temperature = &manifest.fields[0][0];
        let mut values = Vec::new();
        for descriptor in &temperature.partitions {
            let partition = RecoveredFieldPartition::from_canonical_bytes(
                &imported
                    .result
                    .objects
                    .partition_payload(descriptor.partition.digest())
                    .unwrap(),
            )
            .unwrap();
            values.extend_from_slice(partition.storage.as_host_f64().unwrap());
        }
        assert_eq!(values.len(), projection.nodes.len());
        if order == ElementOrder::Tet10 {
            assert!(projection.nodes.iter().zip(&values).all(|(node, value)| {
                let x = node.coordinates_m.canonical_value()[0];
                (value - (300.0 + 6.0 * x * (1.0 - x))).abs() < 1e-8
            }));
        }
    }
}

#[test]
fn native_retained_joule_source_rejects_missing_foreign_and_understated_authority() {
    let (source, bound, mesh) = fixture(ElementOrder::Tet4);
    let unbound = source.thermal_intent().clone();
    let task = prepare(Some(source), bound.clone(), mesh.clone());
    let invoke = |inputs: &[runmat_analysis_execution::PreparedAnalysisObjects]| {
        NativeAnalysisKernel.execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveThermal,
            host: &task.host,
            inputs,
            output_policy: policy(1024),
        })
    };
    for index in 0..task.inputs.len() {
        let mut inputs = task.inputs.clone();
        inputs.remove(index);
        assert!(invoke(&inputs).is_err());
        let mut inputs = task.inputs.clone();
        inputs[index].manifest.logical_identity = Digest::sha256("foreign-authority");
        assert!(invoke(&inputs).is_err());
    }
    let ordinary = prepare(None, unbound, mesh);
    assert!(task.resource_plan.request.memory_bytes > ordinary.resource_plan.request.memory_bytes);
    let mut inputs = task.inputs.clone();
    let resource = inputs
        .iter()
        .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    inputs[resource] = ordinary
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap()
        .clone();
    assert!(invoke(&inputs).is_err());
    let (foreign, _, _) = fixture(ElementOrder::Tet10);
    let source_index = inputs
        .iter()
        .position(|input| {
            !matches!(
                input.manifest.kind,
                AnalysisNumericalObjectKind::ThermalWorkload
                    | AnalysisNumericalObjectKind::Mesh
                    | AnalysisNumericalObjectKind::ResourcePlan
            )
        })
        .unwrap();
    let mut corrupted = task.inputs[source_index].payload();
    corrupted[0] ^= 1;
    inputs = task.inputs.clone();
    inputs[source_index] = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldSourceBinding,
        task.inputs[source_index].manifest.logical_identity,
        &corrupted,
        1024,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert!(invoke(&inputs).is_err());
    inputs = task.inputs.clone();
    inputs[source_index] = runmat_analysis_execution::prepare_analysis_objects(
        inputs[source_index].manifest.kind,
        foreign.identity().unwrap(),
        &foreign.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert!(invoke(&inputs).is_err());
}
