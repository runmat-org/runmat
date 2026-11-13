//! Checked-recipe source authority through the physical candidate planner.
use super::*;
use crate::placement::tensor_representation;
use runmat_accelerate_api::{
    reduction::{
        ProviderReductionRecipe, ReductionDispatch, ReductionGeometry, ReductionInputBinding,
        ReductionPartialLayout, ReductionRecipeConfig, ReductionRecipeDescriptor,
        ReductionRecipeWork, ReductionSelection, ReductionShaderSources, ReductionStrategy,
    },
    scalar_program::{ScalarNode, ScalarProgram, SCALAR_PROGRAM_REVISION},
    GpuTensorLayout, GpuTensorStorage, NumericElementType, ProviderTypedReductionWorkload,
};
use runmat_types::{
    FusionReductionContract, FusionReductionCount, FusionReductionEmpty, FusionReductionOperation,
    FusionReductionTies, NumericClass, ReductionNaN,
};

fn checked_recipe() -> ProviderReductionRecipe {
    let dispatch = |groups| ReductionDispatch {
        workgroups: [groups, 1, 1],
        logical_groups: groups,
    };
    let descriptor = ReductionRecipeDescriptor {
        contract: FusionReductionContract {
            operation: FusionReductionOperation::Sum,
            input_class: NumericClass::Single,
            accumulator_class: NumericClass::Double,
            output_class: NumericClass::Double,
            input_shape: vec![4, 2],
            output_shape: vec![1, 2],
            axes: vec![0],
            identity: false,
            nan_mode: ReductionNaN::Include,
            count: FusionReductionCount::None,
            empty: FusionReductionEmpty::PositiveZero,
            ties: FusionReductionTies::NotApplicable,
        },
        execution_precision: ProviderPrecision::F64,
        producer_transport: runmat_accelerate_api::scalar_program::ScalarTransport::NativeFloat,
        producer_implementations: Vec::new(),
        config: ReductionRecipeConfig {
            workgroup_size: 8,
            chunk_elements: 2,
            max_workgroup_size: 256,
            max_workgroup_storage_bytes: 32768,
            max_dispatch_dimension: 65535,
            max_storage_binding_bytes: 1024,
            max_rank: 8,
        },
        inputs: vec![ReductionInputBinding {
            precision: ProviderPrecision::F32,
            shape: vec![4, 2],
            layout: GpuTensorLayout::Contiguous,
            broadcast_shape: vec![4, 2],
            strides: vec![1, 4],
            payload_bytes: 32,
        }],
        iteration_shape: vec![4, 2],
        output_shape: vec![1, 2],
        output_precision: ProviderPrecision::F64,
        geometry: ReductionGeometry {
            reduce_elements: 4,
            slices: 2,
            chunks: 2,
            partial_elements: 4,
            single: dispatch(2),
            pass1: dispatch(4),
            pass2: dispatch(2),
        },
        partial: ReductionPartialLayout {
            precision: ProviderPrecision::F64,
            alignment: 8,
            stride: 16,
            count_offset: 8,
            flags_offset: 12,
        },
        work: ReductionRecipeWork {
            producer_scalar_work_units: 8,
            output_payload_bytes: 16,
            output_requested_bytes: 16,
            partial_payload_bytes: 64,
            partial_requested_bytes: 64,
            workgroup_bytes: 128,
            single_dispatches: 1,
            two_pass_dispatches: 2,
            uniform_bytes: 0,
        },
    };
    let scalar = ScalarProgram::new(
        Digest::sha256(b"checked scalar fixture"),
        SCALAR_PROGRAM_REVISION,
        vec![ScalarNode::Input {
            id: 7,
            class: NumericClass::Single,
        }],
        7,
    )
    .unwrap();
    let source = "@compute @workgroup_size(8) fn main() {}";
    ProviderReductionRecipe::new_checked(
        Digest::sha256(b"checked reduction fixture"),
        descriptor,
        ReductionShaderSources {
            single: source.into(),
            pass1: source.into(),
            pass2: source.into(),
        },
        scalar,
        vec![7],
    )
    .unwrap()
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn altered_checked_selection_workload_refuses_before_provider_query() {
    for altered_field in ["strategy", "work", "output"] {
        let recipe = checked_recipe();
        let selection = ReductionSelection::new(&recipe, ReductionStrategy::SinglePass).unwrap();
        let mut typed = ProviderTypedReductionWorkload::from_selection(&selection);
        match altered_field {
            "strategy" => typed.strategy = ReductionStrategy::TwoPass,
            "work" => typed.work.dispatches += 1,
            "output" => typed.output.shape = vec![2, 1],
            _ => unreachable!(),
        }
        let provider = registered_fixture();
        let allocation = provider
            .owner
            .as_ref()
            .expect("registered fixture retains its allocation owner")
            .allocate_zeroed(vec![4, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let value = Value::GpuTensor(allocation.handle.clone());
        let leaves = [&value];
        let result = try_plan_physical_set(
            provider,
            &Rc::new(Capture::default()),
            1_000_000,
            vec![LocalPhysicalCandidate {
                authority: None,
                physical_identity: Some(selection.identity()),
                provider_kind: ExecutionCandidateKind::ProviderFusion,
                data_inputs: &leaves,
                source_inputs: Some(typed.inputs.clone()),
                prepared_inputs: Some(typed.inputs.clone()),
                input_adaptation: None,
                outputs: vec![typed.output.clone()],
                workload: ProviderWorkload {
                    scalar_work_units: typed.work.checked_total_scalar_work_units(),
                    typed_reduction: Some(typed),
                    ..Default::default()
                },
                preparation: CandidatePreparationState::Cold,
                preparation_program: Some(
                    runmat_accelerate_api::ProviderPreparationProgram::TypedReduction {
                        selection: &selection,
                        host_metadata: None,
                    },
                ),
                shader_requirements: None,
                fallback: runmat_execution::CandidateCostEvidence::Estimated(cost(10_000, 0)),
                required_upload_bytes: Some(0),
                required_download_bytes: 0,
                downstream_materialization: false,
            }],
        );
        assert!(result.is_err(), "altered {altered_field} must be rejected");
        assert_eq!(
            provider.queries.load(Ordering::SeqCst),
            0,
            "altered {altered_field} reached provider feasibility"
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unchanged_checked_typed_selection_reaches_same_physical_query() {
    let recipe = checked_recipe();
    let selection = ReductionSelection::new(&recipe, ReductionStrategy::SinglePass).unwrap();
    let typed = ProviderTypedReductionWorkload::from_selection(&selection);
    let provider = registered_fixture();
    let allocation = provider
        .owner
        .as_ref()
        .expect("registered fixture retains its allocation owner")
        .allocate_zeroed(vec![4, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let value = Value::GpuTensor(allocation.handle.clone());
    let leaves = [&value];
    assert_eq!(
        tensor_representation(&value, ProviderPrecision::F64),
        Some(typed.inputs[0].clone())
    );
    // The same registered facade retains the fixture allocation producer.
    // This proves routing/lease admission, not backend buffer authentication.
    let selected = plan_physical_set(
        provider,
        &Rc::new(Capture::default()),
        1_000_000,
        vec![LocalPhysicalCandidate {
            authority: None,
            physical_identity: Some(selection.identity()),
            provider_kind: ExecutionCandidateKind::ProviderFusion,
            data_inputs: &leaves,
            source_inputs: Some(typed.inputs.clone()),
            prepared_inputs: Some(typed.inputs.clone()),
            input_adaptation: None,
            outputs: vec![typed.output.clone()],
            workload: ProviderWorkload {
                scalar_work_units: selection.work().checked_total_scalar_work_units(),
                typed_reduction: Some(typed.clone()),
                ..Default::default()
            },
            preparation: CandidatePreparationState::Cold,
            preparation_program: Some(
                runmat_accelerate_api::ProviderPreparationProgram::TypedReduction {
                    selection: &selection,
                    host_metadata: None,
                },
            ),
            shader_requirements: None,
            fallback: runmat_execution::CandidateCostEvidence::Estimated(cost(10_000, 0)),
            required_upload_bytes: Some(0),
            required_download_bytes: 0,
            downstream_materialization: false,
        }],
    );
    assert!(matches!(
        selected,
        LocalPlacementOutcome::Selected {
            provider_index: Some(0),
            provider_query,
            ..
        } if provider_query.workload.typed_reduction == Some(typed)
    ));
    assert_eq!(provider.queries.load(Ordering::SeqCst), 1);
}

fn try_plan_resident(
    provider: &MultiProvider,
    selection: &ReductionSelection<'_>,
    value: &Value,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    let typed = ProviderTypedReductionWorkload::from_selection(selection);
    let leaves = [value];
    try_plan_physical_set(
        provider,
        &Rc::new(Capture::default()),
        1_000_000,
        vec![LocalPhysicalCandidate {
            authority: None,
            physical_identity: Some(selection.identity()),
            provider_kind: ExecutionCandidateKind::ProviderFusion,
            data_inputs: &leaves,
            source_inputs: Some(typed.inputs.clone()),
            prepared_inputs: Some(typed.inputs.clone()),
            input_adaptation: None,
            outputs: vec![typed.output.clone()],
            workload: ProviderWorkload {
                scalar_work_units: selection.work().checked_total_scalar_work_units(),
                typed_reduction: Some(typed),
                ..Default::default()
            },
            preparation: CandidatePreparationState::Cold,
            preparation_program: None,
            shader_requirements: None,
            fallback: runmat_execution::CandidateCostEvidence::Estimated(cost(10_000, 0)),
            required_upload_bytes: Some(0),
            required_download_bytes: 0,
            downstream_materialization: false,
        }],
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_resident_foreign_expired_and_wrong_facade_refuse_before_query() {
    let recipe = checked_recipe();
    let selection = ReductionSelection::new(&recipe, ReductionStrategy::SinglePass).unwrap();
    let registered = registered_fixture();
    let foreign = registered_fixture();
    let allocation = registered
        .owner
        .as_ref()
        .expect("registered fixture retains its allocation owner")
        .allocate_zeroed(vec![4, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let value = Value::GpuTensor(allocation.handle.clone());
    assert!(try_plan_resident(foreign, &selection, &value).is_err());
    assert_eq!(foreign.queries.load(Ordering::SeqCst), 0);

    // Same advertised device ID is insufficient: this facade was not
    // registered as the handle's owner and never owns an allocation.
    let wrong_facade = MultiProvider {
        device: registered.device_id(),
        owner: None,
        reject: None,
        queries: AtomicUsize::new(0),
    };
    assert!(try_plan_resident(&wrong_facade, &selection, &value).is_err());
    assert_eq!(wrong_facade.queries.load(Ordering::SeqCst), 0);

    allocation.probe.mark_unavailable();
    assert!(allocation
        .handle
        .allocation_lease()
        .validate_live()
        .is_err());
    assert!(try_plan_resident(registered, &selection, &value).is_err());
    assert_eq!(registered.queries.load(Ordering::SeqCst), 0);
}
