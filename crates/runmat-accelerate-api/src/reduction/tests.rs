use super::*;
use crate::scalar_program::{ScalarNode, ScalarProgram, SCALAR_PROGRAM_REVISION};
use crate::{
    test_support::TestAllocationOwner, GpuTensorLayout, GpuTensorStorage, NumericElementType,
    ProviderPrecision, TransposeInfo,
};
use runmat_execution::Digest;
use runmat_types::NumericClass;
use runmat_types::{
    FusionReductionContract, FusionReductionCount, FusionReductionEmpty, FusionReductionOperation,
    FusionReductionTies, ReductionNaN,
};

struct NoReductionProvider;
impl crate::AccelProvider for NoReductionProvider {
    fn device_info(&self) -> String {
        "no reduction provider".into()
    }
    fn upload(&self, _: &crate::HostTensorView) -> anyhow::Result<crate::GpuTensorHandle> {
        panic!("default typed reduction must not upload")
    }
    fn download<'a>(&'a self, _: &'a crate::GpuTensorHandle) -> crate::AccelDownloadFuture<'a> {
        panic!("default typed reduction must not download")
    }
}

#[test]
fn default_typed_reduction_declines_before_effects_and_keeps_input() {
    use crate::AccelProvider;
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![4, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let recipe = recipe(descriptor());
    {
        let inputs = [input.handle.clone()];
        let selected = ReductionSelection::new(&recipe, ReductionStrategy::SinglePass).unwrap();
        let submitted =
            ReductionSubmission::new(selected, &inputs, owner.device_id(), None, None).unwrap();
        let error = NoReductionProvider
            .submit_reduction_recipe(&submitted)
            .unwrap_err();
        assert!(error.is_unsupported());
        assert!(!error.is_after_effect());
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn default_physical_selection_is_pre_effect_unsupported() {
    use crate::AccelProvider;
    let provider = NoReductionProvider;
    let config = provider.reduction_recipe_config(4, 2).unwrap_err();
    assert!(config.is_unsupported());
    assert!(!config.is_after_effect());
    let recipe = recipe(descriptor());
    let candidates = provider
        .reduction_recipe_candidates(&recipe, None)
        .unwrap_err();
    assert!(candidates.is_unsupported());
    assert!(!candidates.is_after_effect());
}

#[test]
fn projected_workload_retains_exact_selection_identity_and_representations() {
    let recipe = recipe(descriptor());
    let first = ReductionSelection::new(&recipe, ReductionStrategy::SinglePass).unwrap();
    let second = ReductionSelection::new(&recipe, ReductionStrategy::TwoPass).unwrap();
    let first_work = crate::ProviderTypedReductionWorkload::from_selection(&first);
    let second_work = crate::ProviderTypedReductionWorkload::from_selection(&second);
    assert_eq!(first_work.recipe_identity, recipe.identity());
    assert_eq!(second_work.recipe_identity, recipe.identity());
    assert_ne!(
        first_work.selection_identity,
        second_work.selection_identity
    );
    assert_eq!(first_work.work.partial_requested_bytes, 0);
    assert_eq!(second_work.work.partial_requested_bytes, 64);
    assert_eq!(first_work.inputs[0].shape, vec![4, 2]);
    assert_eq!(
        first_work.inputs[0].element_type,
        crate::ProviderElementType::F32
    );
    assert_eq!(second_work.output.shape, vec![1, 2]);
    assert_eq!(
        second_work.output.element_type,
        crate::ProviderElementType::F64
    );
}

fn descriptor() -> ReductionRecipeDescriptor {
    let dispatch = |logical_groups| ReductionDispatch {
        workgroups: [logical_groups, 1, 1],
        logical_groups,
    };
    ReductionRecipeDescriptor {
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
        producer_transport: crate::scalar_program::ScalarTransport::NativeFloat,
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
    }
}

fn sources() -> ReductionShaderSources {
    // API validation is deliberately not a shader compiler or semantic proof.
    // The accelerator's Naga tests validate real generated sources separately.
    let source = "@compute @workgroup_size(8) fn main() {}";
    ReductionShaderSources {
        single: source.into(),
        pass1: source.into(),
        pass2: source.into(),
    }
}

fn recipe(d: ReductionRecipeDescriptor) -> ProviderReductionRecipe {
    ProviderReductionRecipe::new(Digest::sha256(b"captured semantics"), d, sources()).unwrap()
}

fn checked_input(id: u32, class: NumericClass) -> ScalarProgram {
    ScalarProgram::new(
        Digest::sha256(b"captured scalar source"),
        SCALAR_PROGRAM_REVISION,
        vec![ScalarNode::Input { id, class }],
        id,
    )
    .unwrap()
}

#[test]
fn checked_recipe_binds_scalar_identity_class_and_ordered_physical_input() {
    let make = |program, inputs| {
        ProviderReductionRecipe::new_checked(
            Digest::sha256(b"captured semantics"),
            descriptor(),
            sources(),
            program,
            inputs,
        )
    };
    let first = make(checked_input(7, NumericClass::Single), vec![7]).unwrap();
    let changed = make(checked_input(8, NumericClass::Single), vec![8]).unwrap();
    assert_ne!(first.identity(), changed.identity());
    assert_eq!(first.scalar_inputs(), &[7]);
    assert_eq!(first.checked_scalar().unwrap().wire().root, 7);
    assert!(make(checked_input(7, NumericClass::Single), vec![8]).is_err());
    assert!(make(checked_input(7, NumericClass::Double), vec![7]).is_err());
    assert!(make(checked_input(7, NumericClass::Single), vec![]).is_err());
    assert!(recipe(descriptor()).checked_scalar().is_none());
}

#[test]
fn owner_contract_axes_and_classes_must_match_physical_recipe() {
    let mut bad_axis = descriptor();
    bad_axis.contract.axes = vec![1];
    assert!(ProviderReductionRecipe::new(Digest::sha256(b"source"), bad_axis, sources()).is_err());
    let mut swapped_output = descriptor();
    swapped_output.contract.output_shape = vec![2, 1];
    swapped_output.output_shape = vec![2, 1];
    assert!(
        ProviderReductionRecipe::new(Digest::sha256(b"source"), swapped_output, sources()).is_err()
    );
    let mut duplicate = descriptor();
    duplicate.contract.axes = vec![0, 0];
    assert!(ProviderReductionRecipe::new(Digest::sha256(b"source"), duplicate, sources()).is_err());
    let mut wrong_class = descriptor();
    wrong_class.contract.output_class = NumericClass::Single;
    assert!(
        ProviderReductionRecipe::new(Digest::sha256(b"source"), wrong_class, sources()).is_err()
    );
    let mut wrong_policy = descriptor();
    wrong_policy.contract.count = FusionReductionCount::NonNan;
    assert!(
        ProviderReductionRecipe::new(Digest::sha256(b"source"), wrong_policy, sources()).is_err()
    );
    let mut wrong_accumulator = descriptor();
    wrong_accumulator.contract.accumulator_class = NumericClass::Single;
    assert!(
        ProviderReductionRecipe::new(Digest::sha256(b"source"), wrong_accumulator, sources())
            .is_err()
    );
    let mut wrong_mean_count = descriptor();
    wrong_mean_count.contract.operation = FusionReductionOperation::Mean;
    wrong_mean_count.contract.empty = FusionReductionEmpty::NaN;
    wrong_mean_count.contract.count = FusionReductionCount::NonNan;
    assert!(
        ProviderReductionRecipe::new(Digest::sha256(b"source"), wrong_mean_count, sources())
            .is_err()
    );
}

#[test]
fn owner_shape_accepts_ranked_identity_and_scalar_normalization() {
    let mut identity = descriptor();
    identity.contract.input_shape = vec![2, 3, 1];
    identity.contract.output_shape = vec![2, 3, 1];
    identity.contract.axes = vec![2];
    identity.contract.identity = true;
    identity.iteration_shape = vec![2, 3, 1];
    identity.output_shape = vec![2, 3, 1];
    identity.inputs[0].shape = vec![2, 3, 1];
    identity.inputs[0].broadcast_shape = vec![2, 3, 1];
    identity.inputs[0].strides = vec![1, 2, 0];
    identity.inputs[0].payload_bytes = 24;
    identity.partial.precision = ProviderPrecision::F32;
    identity.partial.alignment = 4;
    identity.partial.stride = 12;
    identity.partial.count_offset = 4;
    identity.partial.flags_offset = 8;
    identity.geometry.reduce_elements = 1;
    identity.geometry.slices = 6;
    identity.geometry.chunks = 1;
    identity.geometry.partial_elements = 6;
    for dispatch in [
        &mut identity.geometry.single,
        &mut identity.geometry.pass1,
        &mut identity.geometry.pass2,
    ] {
        dispatch.workgroups = [6, 1, 1];
        dispatch.logical_groups = 6;
    }
    identity.work.producer_scalar_work_units = 6;
    identity.work.output_payload_bytes = 48;
    identity.work.output_requested_bytes = 48;
    identity.work.partial_payload_bytes = 72;
    identity.work.partial_requested_bytes = 72;
    identity.work.workgroup_bytes = 96;
    let identity = ProviderReductionRecipe::new(Digest::sha256(b"identity"), identity, sources())
        .expect("rank-preserving identity must retain zero singleton strides");
    let single = identity
        .terminal_work_prior(ReductionStrategy::SinglePass)
        .unwrap();
    let two = identity
        .terminal_work_prior(ReductionStrategy::TwoPass)
        .unwrap();
    assert_eq!(
        (
            single.sample_visits,
            single.combines,
            single.finalizations,
            single.conversions
        ),
        (0, 0, 6, 6)
    );
    assert_eq!(
        (
            two.sample_visits,
            two.combines,
            two.finalizations,
            two.conversions
        ),
        (6, 0, 6, 6)
    );

    let mut scalar = descriptor();
    scalar.contract.input_shape = vec![1, 1, 1];
    scalar.contract.output_shape = vec![1, 1];
    scalar.contract.axes = vec![2];
    scalar.iteration_shape = vec![1, 1, 1];
    scalar.output_shape = vec![1, 1];
    scalar.inputs[0].shape = vec![1, 1, 1];
    scalar.inputs[0].broadcast_shape = vec![1, 1, 1];
    scalar.inputs[0].strides = vec![0, 0, 0];
    scalar.inputs[0].payload_bytes = 4;
    scalar.geometry.reduce_elements = 1;
    scalar.geometry.slices = 1;
    scalar.geometry.chunks = 1;
    scalar.geometry.partial_elements = 1;
    for dispatch in [
        &mut scalar.geometry.single,
        &mut scalar.geometry.pass1,
        &mut scalar.geometry.pass2,
    ] {
        dispatch.workgroups = [1, 1, 1];
        dispatch.logical_groups = 1;
    }
    scalar.work.producer_scalar_work_units = 1;
    scalar.work.output_payload_bytes = 8;
    scalar.work.output_requested_bytes = 8;
    scalar.work.partial_payload_bytes = 16;
    scalar.work.partial_requested_bytes = 16;
    ProviderReductionRecipe::new(Digest::sha256(b"scalar"), scalar, sources())
        .expect("scalar-normalized reduction must retain zero singleton strides");
}

#[test]
fn strategy_is_frozen_in_identity_and_exact_selected_work() {
    let r = recipe(descriptor());
    let single = ReductionSelection::new(&r, ReductionStrategy::SinglePass).unwrap();
    let two = ReductionSelection::new(&r, ReductionStrategy::TwoPass).unwrap();
    assert_ne!(single.identity(), two.identity());
    assert_eq!(single.work().partial_requested_bytes, 0);
    assert_eq!(two.work().partial_requested_bytes, 64);
    assert_eq!(single.work().dispatches, 1);
    assert_eq!(two.work().dispatches, 2);
    assert_eq!(single.work().output_requested_bytes, 16);
    let first = r
        .terminal_work_prior(ReductionStrategy::SinglePass)
        .unwrap();
    let second = r.terminal_work_prior(ReductionStrategy::TwoPass).unwrap();
    assert_eq!(
        (first.sample_visits, first.combines, first.finalizations),
        (8, 8 + 2 * 7, 2)
    );
    assert_eq!(
        (second.sample_visits, second.combines, second.finalizations),
        (8, 8 + 4 + 6 * 7, 2)
    );
    assert_eq!(first.count_updates, first.combines);
    assert_eq!(second.count_updates, second.combines);
    assert_eq!(
        single.work().terminal_scalar_work_units,
        first.checked_units().unwrap()
    );
    assert_eq!(
        two.work().checked_total_scalar_work_units(),
        Some(8 + second.checked_units().unwrap())
    );
    assert!(std::ptr::eq(two.recipe(), &r));
    let mut changed = sources();
    changed.pass2.push_str("\n// distinct compiled pipeline");
    let changed =
        ProviderReductionRecipe::new(Digest::sha256(b"captured semantics"), descriptor(), changed)
            .unwrap();
    assert_ne!(r.identity(), changed.identity());
    let mut geometry = descriptor();
    geometry.geometry.single.workgroups = [1, 2, 1];
    assert_ne!(r.identity(), recipe(geometry).identity());
}

#[test]
fn word_boundary_conversion_is_not_priced_twice_and_total_overflow_declines() {
    let native = recipe(descriptor());
    let mut words = descriptor();
    words.execution_precision = ProviderPrecision::F32;
    words.producer_transport = crate::scalar_program::ScalarTransport::Binary32Words;
    let words = recipe(words);
    for strategy in [ReductionStrategy::SinglePass, ReductionStrategy::TwoPass] {
        let native = native.terminal_work_prior(strategy).unwrap();
        let decoded = words.terminal_work_prior(strategy).unwrap();
        assert_eq!(native.conversions, 8);
        assert_eq!(decoded.conversions, 0);
        assert_eq!(
            native.checked_units().unwrap(),
            decoded.checked_units().unwrap() + 8
        );
    }
    let mut overflow = descriptor();
    overflow.work.producer_scalar_work_units = u64::MAX;
    let overflow = recipe(overflow);
    assert!(ReductionSelection::new(&overflow, ReductionStrategy::SinglePass).is_err());
}

#[test]
fn malformed_physical_packets_fail_closed_before_submission() {
    for case in 0..14 {
        let mut d = descriptor();
        match case {
            0 => d.output_shape = vec![2, 2],
            1 => d.iteration_shape = vec![3, 2],
            2 => d.geometry.chunks = 1,
            3 => d.geometry.partial_elements = 2,
            4 => d.geometry.pass1.workgroups = [1, 1, 1],
            5 => d.geometry.single.logical_groups = 3,
            6 => d.partial.count_offset = 4,
            7 => d.work.partial_requested_bytes = 32,
            8 => d.inputs[0].payload_bytes = 16,
            9 => d.inputs[0].strides = vec![1, 100],
            10 => d.config.workgroup_size = 3,
            11 => d.config.max_storage_binding_bytes = 16,
            12 => d.work.uniform_bytes = 16,
            13 => d.iteration_shape = vec![usize::MAX, usize::MAX],
            _ => unreachable!(),
        }
        assert!(
            ProviderReductionRecipe::new(Digest::sha256(b"s"), d, sources()).is_err(),
            "case {case}"
        );
    }
    let mut missing = sources();
    missing.pass1.clear();
    assert!(ProviderReductionRecipe::new(Digest::sha256(b"s"), descriptor(), missing).is_err());
    assert_eq!(
        super::validation::elements(&[usize::MAX, usize::MAX, 0]).unwrap(),
        0
    );
}

#[test]
fn empty_slices_price_tree_and_output_fill_without_sampling_or_mean_division() {
    let mut d = descriptor();
    d.inputs.clear();
    d.iteration_shape[0] = 0;
    d.contract.input_shape = d.iteration_shape.clone();
    d.contract.operation = FusionReductionOperation::Mean;
    d.contract.count = FusionReductionCount::SliceLength;
    d.contract.empty = FusionReductionEmpty::NaN;
    d.geometry.reduce_elements = 0;
    d.geometry.chunks = 1;
    d.geometry.partial_elements = 2;
    d.geometry.pass1 = d.geometry.single;
    d.work.partial_payload_bytes = 32;
    d.work.partial_requested_bytes = 32;
    d.work.producer_scalar_work_units = 0;
    let r = recipe(d);
    for strategy in [ReductionStrategy::SinglePass, ReductionStrategy::TwoPass] {
        let prior = r.terminal_work_prior(strategy).unwrap();
        assert_eq!(
            (prior.sample_visits, prior.divisions, prior.finalizations),
            (0, 0, 2)
        );
        assert!(prior.combines > 0);
        assert!(
            ReductionSelection::new(&r, strategy)
                .unwrap()
                .work()
                .checked_total_scalar_work_units()
                .unwrap()
                > 0
        );
    }
}

#[test]
fn empty_output_keeps_minimum_output_allocation_but_no_scratch_or_dispatch() {
    let mut d = descriptor();
    d.inputs.clear(); // A constant producer need not bind a data operand.
    d.iteration_shape = vec![4, 0];
    d.output_shape = vec![1, 0];
    d.contract.input_shape = d.iteration_shape.clone();
    d.contract.output_shape = d.output_shape.clone();
    d.geometry.slices = 0;
    d.geometry.partial_elements = 0;
    let absent = ReductionDispatch {
        workgroups: [0; 3],
        logical_groups: 0,
    };
    d.geometry.single = absent;
    d.geometry.pass1 = absent;
    d.geometry.pass2 = absent;
    d.work.output_payload_bytes = 0;
    d.work.output_requested_bytes = 4;
    d.work.partial_payload_bytes = 0;
    d.work.partial_requested_bytes = 0;
    d.work.single_dispatches = 0;
    d.work.two_pass_dispatches = 0;
    d.work.producer_scalar_work_units = 0;
    let r = recipe(d);
    for strategy in [ReductionStrategy::SinglePass, ReductionStrategy::TwoPass] {
        let selection = ReductionSelection::new(&r, strategy).unwrap();
        assert_eq!(selection.work().dispatches, 0);
        assert_eq!(selection.work().partial_requested_bytes, 0);
        assert_eq!(selection.work().output_requested_bytes, 4);
        assert_eq!(selection.work().terminal_scalar_work_units, 0);
    }
}

#[test]
fn authentic_inputs_are_borrowed_without_a_shadow_lifetime_manager() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![4, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let r = recipe(descriptor());
    let inputs = [input.handle];
    {
        let selection = ReductionSelection::new(&r, ReductionStrategy::TwoPass).unwrap();
        let submission =
            ReductionSubmission::new(selection, &inputs, owner.device_id(), None, None).unwrap();
        assert!(std::ptr::eq(&submission.inputs()[0], &inputs[0]));
        assert_eq!(submission.device_id(), owner.device_id());
        assert_eq!(
            submission.selection().strategy(),
            ReductionStrategy::TwoPass
        );
    }
    assert_eq!(input.probe.retirements(), 0);
    drop(inputs);
    // The recipe is still live, but it cannot hold an allocation alive.
    assert_eq!(input.probe.retirements(), 1);
    assert_eq!(r.descriptor().inputs[0].precision, ProviderPrecision::F32);
}

#[test]
fn submission_checks_owner_native_storage_layout_and_actual_allocation_extent() {
    let owner = TestAllocationOwner::new();
    let foreign = TestAllocationOwner::new();
    let r = recipe(descriptor());
    let bad = [
        foreign
            .allocate_zeroed(vec![4, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap(),
        owner
            .allocate_zeroed(vec![4, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap(),
        owner
            .allocate_zeroed(
                vec![4, 2],
                NumericElementType::F32,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap(),
        owner
            .allocate_zeroed(vec![2, 4], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap(),
    ];
    for input in bad {
        let inputs = [input.handle];
        assert!(ReductionSubmission::new(
            ReductionSelection::new(&r, ReductionStrategy::SinglePass).unwrap(),
            &inputs,
            owner.device_id(),
            None,
            None
        )
        .is_err());
        assert_eq!(input.probe.retirements(), 0);
        drop(inputs);
        assert_eq!(input.probe.retirements(), 1);
    }
    let allocation = owner
        .allocate_zeroed(
            vec![512, 1],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let view = allocation
        .handle
        .with_view(allocation.handle.view().clone().with_shape(vec![4, 2]))
        .unwrap();
    assert!(ReductionSubmission::new(
        ReductionSelection::new(&r, ReductionStrategy::SinglePass).unwrap(),
        &[view],
        owner.device_id(),
        None,
        None
    )
    .is_err());
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);

    let allocation = owner
        .allocate_zeroed(vec![2, 4], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let layout = GpuTensorLayout::Transpose(TransposeInfo {
        base_rows: 2,
        base_cols: 4,
    });
    let transposed = allocation
        .handle
        .with_view(
            allocation
                .handle
                .view()
                .clone()
                .with_shape(vec![4, 2])
                .with_layout(layout),
        )
        .unwrap();
    let inputs = [transposed];
    assert!(ReductionSubmission::new(
        ReductionSelection::new(&r, ReductionStrategy::SinglePass).unwrap(),
        &inputs,
        owner.device_id(),
        None,
        None
    )
    .is_err());
    let mut d = descriptor();
    d.inputs[0].layout = layout;
    d.inputs[0].strides = vec![2, 1];
    let transpose_recipe = recipe(d);
    {
        let submission = ReductionSubmission::new(
            ReductionSelection::new(&transpose_recipe, ReductionStrategy::SinglePass).unwrap(),
            &inputs,
            owner.device_id(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(submission.inputs()[0].view().layout(), layout);
    }
    drop(inputs);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn logical_f32_leaf_is_rejected_without_releasing_the_callers_allocation() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_logical_zeroed(vec![4, 2], ProviderPrecision::F32)
        .unwrap();
    let caller = input.handle.clone();
    let wire = caller.wire_descriptor();
    let r = recipe(descriptor());
    let inputs = [input.handle];
    // Native encoding, dimensions, device and byte extent all match. Only the
    // authentic logical producer interpretation disqualifies this leaf.
    assert_eq!(caller.physical().element_type(), NumericElementType::F32);
    assert_eq!(caller.physical().payload_bytes(), 32);
    assert!(caller.view().is_logical());
    for strategy in [ReductionStrategy::SinglePass, ReductionStrategy::TwoPass] {
        let error = match ReductionSubmission::new(
            ReductionSelection::new(&r, strategy).unwrap(),
            &inputs,
            owner.device_id(),
            None,
            None,
        ) {
            Ok(_) => panic!("logical storage must not be interpreted as a numerical leaf"),
            Err(error) => error,
        };
        assert!(!error.is_unsupported());
        assert!(error.to_string().contains("prepared input view mismatch"));
        assert_eq!(caller.wire_descriptor(), wire);
        assert_eq!(input.probe.retirements(), 0);
    }
    drop(inputs);
    assert_eq!(input.probe.retirements(), 0);
    drop(caller);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn unavailable_leaf_retains_typed_not_live_source_and_caller_ownership() {
    use crate::allocation::AllocationAcquisitionError;
    use std::error::Error;

    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![4, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let caller = input.handle.clone();
    let wire = caller.wire_descriptor();
    let r = recipe(descriptor());
    let inputs = [input.handle];
    // The exact same native leaf is admissible before owner availability is
    // lost. No descriptor forgery or premature manual free constructs this case.
    assert!(ReductionSubmission::new(
        ReductionSelection::new(&r, ReductionStrategy::SinglePass).unwrap(),
        &inputs,
        owner.device_id(),
        None,
        None,
    )
    .is_ok());
    input.probe.mark_unavailable();
    for strategy in [ReductionStrategy::SinglePass, ReductionStrategy::TwoPass] {
        let error = match ReductionSubmission::new(
            ReductionSelection::new(&r, strategy).unwrap(),
            &inputs,
            owner.device_id(),
            None,
            None,
        ) {
            Ok(_) => panic!("unavailable allocation must not be submitted"),
            Err(error) => error,
        };
        assert!(
            !error.is_unsupported(),
            "device failure is not permission to retry"
        );
        assert_eq!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<AllocationAcquisitionError>()),
            Some(&AllocationAcquisitionError::NotLive),
        );
        assert_eq!(caller.wire_descriptor(), wire);
        assert_eq!(input.probe.retirements(), 0);
        drop(error);
        assert_eq!(input.probe.retirements(), 0);
    }
    drop(inputs);
    assert_eq!(input.probe.retirements(), 0);
    drop(caller);
    assert_eq!(input.probe.retirements(), 1);
}
