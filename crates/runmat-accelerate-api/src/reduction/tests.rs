use super::*;
use crate::{
    test_support::TestAllocationOwner, GpuTensorLayout, GpuTensorStorage, NumericElementType,
    ProviderPrecision, TransposeInfo,
};
use runmat_execution::Digest;

fn descriptor() -> ReductionRecipeDescriptor {
    let dispatch = |logical_groups| ReductionDispatch {
        workgroups: [logical_groups, 1, 1],
        logical_groups,
    };
    ReductionRecipeDescriptor {
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
fn empty_output_keeps_minimum_output_allocation_but_no_scratch_or_dispatch() {
    let mut d = descriptor();
    d.inputs.clear(); // A constant producer need not bind a data operand.
    d.iteration_shape = vec![4, 0];
    d.output_shape = vec![1, 0];
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
        let submission = ReductionSubmission::new(selection, &inputs, owner.device_id()).unwrap();
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
            owner.device_id()
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
        owner.device_id()
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
        owner.device_id()
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
        )
        .unwrap();
        assert_eq!(submission.inputs()[0].view().layout(), layout);
    }
    drop(inputs);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}
