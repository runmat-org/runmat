use super::*;
use crate::{prepare_thermal_task, ThermalTaskPreparation};
use runmat_analysis_core::*;

fn preparation() -> ThermalTaskPreparation {
    let p = crate::thermal_study::tests::preparation();
    ThermalTaskPreparation {
        maximum_resident_bytes: u64::MAX,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: p.product.compiled.definition_digest().unwrap(),
        frozen_project_digest: p.frozen_project_digest,
        program_revision: p.program_revision,
        artifact_access: p.artifact_access,
        workload: p.product.workload,
        mesh: p.mesh,
        field_source: None,
        output_policy: p.output_policy,
        required_capabilities: p.required_capabilities,
    }
}

#[test]
fn placement_budget_changes_resource_identity_without_rewriting_thermal_intent() {
    let mut plans = Vec::new();
    for maximum in [64 * 1024 * 1024, 128 * 1024 * 1024] {
        let mut input = preparation();
        let original = input.workload.clone();
        input.maximum_resident_bytes = maximum;
        let mesh = input.mesh.clone();
        let task = prepare_thermal_task(input).unwrap();
        assert_eq!(
            task.resource_plan
                .input
                .accuracy_ceiling
                .resident_byte_budget(),
            maximum
        );
        assert!(task.resource_plan.request.memory_bytes <= maximum);
        let stored = task
            .inputs
            .iter()
            .find(|objects| objects.manifest.kind == AnalysisNumericalObjectKind::ThermalWorkload)
            .unwrap();
        let restored = ThermalWorkload::from_canonical_bytes(&stored.payload()).unwrap();
        assert_eq!(restored, original);
        assert_eq!(
            rederive_thermal_resource_plan(&restored, &mesh, &task.inputs, task.output_policy)
                .unwrap(),
            task.resource_plan
        );
        plans.push(task.resource_plan.identity);
    }
    assert_ne!(plans[0], plans[1]);
    let mut refused = preparation();
    refused.maximum_resident_bytes = 1;
    assert!(prepare_thermal_task(refused).is_err());
}

fn heat_field(p: &mut ThermalTaskPreparation, provenance: SignalProvenance) {
    p.workload
        .model
        .signals
        .push(PhysicalSignal::HeatGenerationDensity(Signal {
            signal_id: SignalId("generic.heat".into()),
            shape: SignalShape::TopologyField {
                topology_id: Digest::from_bytes(*p.mesh.canonical_digest.bytes()).to_string(),
                location: FieldLocation::Element,
                component_count: 1,
            },
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Tensor {
                shape: vec![p.mesh.topology.volume_elements.len(), 1],
                values: vec![
                    HeatGenerationDensity::new(2.0);
                    p.mesh.topology.volume_elements.len()
                ],
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance,
        }));
    p.workload
        .model
        .excitations
        .push(ExcitationAction::HeatGeneration(AnalysisExcitation {
            excitation_id: ExcitationId("generic.heat".into()),
            target: ExcitationTarget::WholeModel,
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::TopologyField,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("generic.heat".into()),
            active_step_ids: vec![p.workload.step_id.0.clone()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        }));
    p.workload.model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("generic.heat".into()),
        action_ids: vec![ExcitationId("generic.heat".into())],
        activation: ExcitationCaseActivation::Always,
    });
}

#[test]
fn standalone_and_ordinary_imported_heat_fields_keep_three_input_admission() {
    for provenance in [
        None,
        Some(SignalProvenance::Authored),
        Some(SignalProvenance::Imported {
            source_digest: Digest::sha256("ordinary-input").to_string(),
        }),
    ] {
        let mut p = preparation();
        if let Some(provenance) = provenance {
            heat_field(&mut p, provenance);
        }
        let workload = p.workload.clone();
        let mesh = p.mesh.clone();
        let policy = p.output_policy;
        let task = prepare_thermal_task(p).unwrap();
        assert_eq!(task.inputs.len(), 3);
        assert_eq!(
            rederive_thermal_resource_plan(&workload, &mesh, &task.inputs, policy).unwrap(),
            task.resource_plan
        );
        assert!(validate_thermal_task_source(&workload, &mesh, &task.inputs)
            .unwrap()
            .is_none());
    }
}

#[test]
fn typed_retained_field_cannot_lose_binding_even_with_rehashed_workload() {
    let mut p = preparation();
    heat_field(
        &mut p,
        SignalProvenance::Retained {
            source_digest: Digest::sha256("original-retained-field").to_string(),
        },
    );
    assert!(prepare_thermal_inputs(&p.workload, &p.mesh, p.output_policy).is_err());
    assert!(prepare_thermal_task(p).is_err());
    let baseline = preparation();
    let mut workload = baseline.workload.clone();
    let mesh = baseline.mesh.clone();
    let policy = baseline.output_policy;
    let task = prepare_thermal_task(baseline).unwrap();
    let mut retained = preparation();
    heat_field(
        &mut retained,
        SignalProvenance::Retained {
            source_digest: Digest::sha256("retained-input").to_string(),
        },
    );
    workload.model = retained.workload.model;
    let mut inputs = task.inputs.clone();
    inputs[0] = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ThermalWorkload,
        workload.identity().unwrap().digest(),
        &workload.canonical_bytes().unwrap(),
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )
    .unwrap();
    assert!(rederive_thermal_resource_plan(&workload, &mesh, &inputs, policy).is_err());
    assert!(validate_thermal_task_source(&workload, &mesh, &inputs).is_err());
}

#[test]
fn source_closure_rejects_extra_duplicate_and_malformed_typed_objects() {
    let p = preparation();
    let workload = p.workload.clone();
    let mesh = p.mesh.clone();
    let policy = p.output_policy;
    let task = prepare_thermal_task(p).unwrap();
    let mut extra = task.inputs.clone();
    extra.push(task.inputs[1].clone());
    assert!(rederive_thermal_resource_plan(&workload, &mesh, &extra, policy).is_err());
    let fake = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldSourceBinding,
        Digest::sha256("not-source"),
        b"not-canonical-retained-source",
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )
    .unwrap();
    let mut malformed = task.inputs.clone();
    malformed.push(fake.clone());
    assert!(rederive_thermal_resource_plan(&workload, &mesh, &malformed, policy).is_err());
    assert!(validate_thermal_task_source(&workload, &mesh, &malformed).is_err());
    malformed.push(fake);
    assert!(rederive_thermal_resource_plan(&workload, &mesh, &malformed, policy).is_err());
}

#[test]
fn cpu_allocation_is_bound_by_resource_identity_and_rederived_after_cold_decode() {
    use runmat_analysis_fea::planning::resource::{LocalExecutionPlan, LocalThreadCount};
    let serial = prepare_thermal_task(preparation()).unwrap();
    let mut requested = preparation();
    requested.local_execution =
        LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap();
    let workload = requested.workload.clone();
    let mesh = requested.mesh.clone();
    let parallel = prepare_thermal_task(requested).unwrap();
    assert_eq!(parallel.resource_plan.request.cpu_millicores, 4_000);
    assert_ne!(
        parallel.resource_plan.identity,
        serial.resource_plan.identity
    );
    assert!(
        parallel.resource_plan.request.memory_bytes
            >= serial.resource_plan.request.memory_bytes
                + 4 * runmat_analysis_fea::planning::resource::CPU_POOL_STACK_BYTES_PER_THREAD
    );
    assert_eq!(
        rederive_thermal_resource_plan(&workload, &mesh, &parallel.inputs, parallel.output_policy)
            .unwrap(),
        parallel.resource_plan
    );
    let cold =
        FeaResourcePlan::from_canonical_bytes(&parallel.resource_plan.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(cold, parallel.resource_plan);
}
