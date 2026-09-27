//! Recipe and actual WGSL validation. Numerical GPU execution is a later gate.
#[path = "support/typed_fusion_plan.rs"]
mod fixture;
use futures::executor::block_on;
use runmat_accelerate::{
    fusion::{
        reduction_recipe::{build, ReductionRecipe, ReductionRecipeConfig},
        FusionKind, FusionOp,
    },
    fusion_exec::{resolve_reduction_program, FusionExecutionRequest},
    graph::PrimitiveOp,
};
use runmat_accelerate_api::{
    GpuTensorLayout, GpuTensorStorage, NumericElementType, ProviderPrecision, TransposeInfo,
};
use runmat_value::{Tensor, Value};

fn config() -> ReductionRecipeConfig {
    ReductionRecipeConfig {
        workgroup_size: 8,
        chunk_elements: 12,
        max_workgroup_size: 256,
        max_workgroup_storage_bytes: 32_768,
        max_dispatch_dimension: 65_535,
        max_storage_binding_bytes: 1 << 30,
        max_rank: 8,
    }
}
fn array(single: bool, shape: &[usize]) -> Value {
    let count = shape.iter().product();
    Value::Tensor(if single {
        Tensor::from_f32(vec![1.0; count], shape.to_vec()).unwrap()
    } else {
        Tensor::new(vec![1.0; count], shape.to_vec()).unwrap()
    })
}
fn call(name: &str, inputs: Vec<u32>, output: u32) -> FusionOp {
    if let [input] = inputs.as_slice() {
        return fixture::cast(name, *input, output);
    }
    FusionOp::Builtin {
        name: name.into(),
        origin: Default::default(),
        inputs,
        output: Some(output),
    }
}
fn recipe(
    name: &str,
    inputs: Vec<Value>,
    controls: Vec<Value>,
    producer: Option<PrimitiveOp>,
    cfg: ReductionRecipeConfig,
) -> ReductionRecipe {
    try_recipe(name, inputs, controls, producer, cfg).unwrap()
}
fn try_recipe(
    name: &str,
    inputs: Vec<Value>,
    controls: Vec<Value>,
    producer: Option<PrimitiveOp>,
    cfg: ReductionRecipeConfig,
) -> anyhow::Result<ReductionRecipe> {
    let count = inputs.len();
    let mut operations = Vec::new();
    let root = if let Some(op) = producer {
        operations.push(FusionOp::Primitive {
            op,
            inputs: (0..count as u32).collect(),
            output: Some(100),
        });
        100
    } else {
        assert_eq!(count, 1);
        0
    };
    let mut terminal = vec![root];
    terminal.extend((count..count + controls.len()).map(|i| i as u32));
    operations.push(call(name, terminal, 101));
    let mut plan = fixture::plan(operations, 101);
    plan.inputs = (0..count + controls.len()).map(|i| i as u32).collect();
    plan.group.kind = FusionKind::Reduction;
    plan.kernel.kind = FusionKind::Reduction;
    plan.reduction_data = Some(root);
    let mut values = inputs;
    values.extend(controls);
    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: values,
        placement: None,
        runtime: None,
    };
    let program = block_on(resolve_reduction_program(&request, ProviderPrecision::F64))
        .unwrap()
        .expect("owner-proven reduction");
    build(&program, cfg)
}
fn axes(name: &str, axis: Value, omit: bool) -> Vec<Value> {
    let mut controls = Vec::new();
    if matches!(name, "min" | "max") {
        controls.push(array(false, &[0, 0]));
    }
    controls.push(axis);
    controls.push(Value::String(
        if omit { "omitnan" } else { "includenan" }.into(),
    ));
    controls
}

#[test]
fn native_leaf_output_and_partial_types_do_not_follow_preferred_precision() {
    let single_sum = recipe(
        "sum",
        vec![array(true, &[33, 2]), array(true, &[33, 2])],
        vec![Value::Num(1.0)],
        Some(PrimitiveOp::ElemMul),
        config(),
    );
    assert!(single_sum
        .inputs()
        .iter()
        .all(|input| input.source_precision == ProviderPrecision::F32
            && input.bound_precision == ProviderPrecision::F32
            && !input.requires_cast()));
    assert_eq!(
        single_sum.partial_layout().precision,
        ProviderPrecision::F64
    );
    assert_eq!(single_sum.partial_layout().stride, 16);
    assert_eq!(single_sum.partial_layout().count_offset, 8);
    assert_eq!(single_sum.partial_layout().flags_offset, 12);
    for source in [single_sum.single_source(), single_sum.pass1_source()] {
        assert!(source.contains("RRInput0 { data: array<f32>"));
        assert!(source.contains("RRInput1 { data: array<f32>"));
        assert!(source.contains("let typed0: f32"));
        assert!(source.contains("struct RRPartial { value: f64"));
    }
    assert!(single_sum
        .single_source()
        .contains("RROutput { data: array<f64>"));
    assert_eq!(single_sum.geometry().chunks, 3);
    assert_eq!(single_sum.work().partial_payload_bytes, 2 * 3 * 16);
    assert_eq!(single_sum.work().producer_scalar_work_units, 66);
    let single_mean = recipe("mean", vec![array(true, &[33, 2])], vec![], None, config());
    assert!(single_mean
        .single_source()
        .contains("RROutput { data: array<f32>"));
    assert!(single_mean
        .pass2_source()
        .contains("return f32(a.value / f64(denominator))"));
    assert_eq!(single_mean.work().output_payload_bytes, 8);
    let single_min = recipe("min", vec![array(true, &[33, 2])], vec![], None, config());
    assert_eq!(single_min.partial_layout().stride, 12);
    assert_eq!(
        single_min.partial_layout().precision,
        ProviderPrecision::F32
    );
}

#[test]
fn broadcast_and_authentic_transpose_share_the_existing_addressing_policy() {
    let broadcast = recipe(
        "sum",
        vec![array(false, &[2, 1]), array(false, &[1, 3])],
        vec![Value::Num(2.0)],
        Some(PrimitiveOp::Add),
        config(),
    );
    assert_eq!(broadcast.contract().input_shape, vec![2, 3]);
    assert_eq!(broadcast.inputs()[0].strides, vec![1, 0]);
    assert_eq!(broadcast.inputs()[1].strides, vec![0, 1]);
    assert_eq!(broadcast.geometry().slices, 2);
    assert_eq!(broadcast.geometry().reduce_elements, 3);
    for source in [broadcast.single_source(), broadcast.pass1_source()] {
        assert!(source.contains("((slice / 1u) % 2u) * 1u"));
        assert!(source.contains("((r / 1u) % 3u) * 1u"));
    }
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 3], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let view = allocation
        .handle
        .with_view(
            allocation
                .handle
                .view()
                .clone()
                .with_shape(vec![3, 2])
                .with_layout(GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 3,
                })),
        )
        .unwrap();
    let transposed = recipe(
        "sum",
        vec![Value::GpuTensor(view)],
        vec![Value::Num(1.0)],
        None,
        config(),
    );
    assert_eq!(transposed.inputs()[0].strides, vec![2, 1]);
    let logical_indices: Vec<_> = (0..2)
        .flat_map(|column| (0..3).map(move |row| row * 2 + column))
        .collect();
    assert_eq!(logical_indices, vec![0, 2, 4, 1, 3, 5]);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(
        allocation.probe.retirements(),
        1,
        "recipe descriptors do not hold execution leases"
    );
}

#[test]
fn all_axes_nd_identity_and_empty_geometry_are_explicit() {
    let all = recipe(
        "sum",
        vec![array(false, &[2, 3, 4])],
        vec![Value::String("all".into())],
        None,
        config(),
    );
    assert_eq!(all.geometry().reduce_elements, 24);
    assert_eq!(all.geometry().slices, 1);
    assert_eq!(all.geometry().chunks, 2);
    let noop = recipe(
        "sum",
        vec![array(true, &[2, 3])],
        vec![Value::Num(4.0)],
        None,
        config(),
    );
    assert!(noop.contract().identity);
    assert_eq!(noop.geometry().reduce_elements, 1);
    assert_eq!(noop.geometry().chunks, 1);
    assert_eq!(noop.geometry().slices, 6);
    assert_eq!(noop.work().workgroup_bytes, 0);
    assert!(!noop.single_source().contains("var<workgroup>"));
    assert!(noop
        .single_source()
        .contains("output.data[slice] = f32(rr_value(slice, 0u))"));
    assert!(noop
        .pass2_source()
        .contains("rr_finish(partials.data[slice])"));
    let empty_slices = recipe(
        "sum",
        vec![array(false, &[2, 0])],
        vec![Value::Num(1.0)],
        None,
        config(),
    );
    assert_eq!(empty_slices.geometry().slices, 0);
    assert_eq!(empty_slices.work().single_dispatches, 0);
    assert_eq!(empty_slices.work().two_pass_dispatches, 0);
    assert_eq!(empty_slices.work().partial_requested_bytes, 0);
    let empty_reduction = recipe(
        "mean",
        vec![array(false, &[0, 3])],
        vec![Value::Num(1.0)],
        None,
        config(),
    );
    assert_eq!(empty_reduction.geometry().reduce_elements, 0);
    assert_eq!(empty_reduction.geometry().slices, 3);
    assert_eq!(empty_reduction.geometry().chunks, 1);
    assert!(empty_reduction
        .single_source()
        .contains("let denominator = 0u"));
    assert!(empty_reduction
        .pass1_source()
        .contains("let end = start + min(12u, 0u - start)"));
}

#[test]
fn every_pass_carries_nan_counts_validity_and_signed_zero_ties() {
    let mean = recipe(
        "mean",
        vec![array(false, &[25, 2])],
        axes("mean", Value::Num(1.0), true),
        None,
        config(),
    );
    for source in [
        mean.single_source(),
        mean.pass1_source(),
        mean.pass2_source(),
    ] {
        assert!(source.contains("let denominator = a.count"));
        assert!(source.contains("a.count + b.count"));
        assert!(source.contains("a.flags | b.flags"));
        assert!(source.contains("if denominator == 0u"));
    }
    for (name, expected) in [
        (
            "min",
            "rr_negative_zero(b.value) && !rr_negative_zero(a.value)",
        ),
        (
            "max",
            "!rr_negative_zero(b.value) && rr_negative_zero(a.value)",
        ),
    ] {
        let extrema = recipe(
            name,
            vec![array(false, &[25, 2])],
            axes(name, Value::Num(1.0), false),
            None,
            config(),
        );
        for source in [
            extrema.single_source(),
            extrema.pass1_source(),
            extrema.pass2_source(),
        ] {
            assert!(source.contains(expected));
            assert!(source.contains("if (a.flags & 1u) == 0u"));
            assert!(source.contains("if (a.flags & 2u) != 0u"));
        }
    }
}

#[test]
fn f64_zero_sign_encoding_is_exact_without_narrowing_nonzero_comparisons() {
    // Independent IEEE bit-pattern oracle for the conversion used only after
    // the shader's zero guard. This is not a GPU execution claim; the Naga
    // matrix below validates the actual F64 shaders with the same helper.
    for (bits, expected) in [(0u64, 0u32), (1u64 << 63, 1u32 << 31)] {
        assert_eq!((f64::from_bits(bits) as f32).to_bits(), expected);
    }
    assert!((f32::from_bits(0x7fc00000) as f64).is_nan());
    for name in ["min", "max"] {
        let extrema = recipe(
            name,
            vec![array(false, &[25, 2])],
            axes(name, Value::Num(1.0), false),
            None,
            config(),
        );
        for source in [
            extrema.single_source(),
            extrema.pass1_source(),
            extrema.pass2_source(),
        ] {
            assert!(source.contains(
                "if value != f64(0.0) { return false; } return (bitcast<u32>(f32(value))"
            ));
            assert!(source.contains("return f64(bitcast<f32>(0x7fc00000u))"));
            assert!(!source.contains("bitcast<vec2<u32>>"));
            assert!(!source.contains("bitcast<f64>"));
            // Infinity and every nonzero comparison still use native F64
            // operands, never the sign helper's temporary f32 zero.
            assert!(source.contains(if name == "min" {
                "b.value < a.value"
            } else {
                "b.value > a.value"
            }));
        }
    }
}

#[test]
fn geometry_and_layout_are_part_of_the_immutable_recipe_identity() {
    let a = recipe("sum", vec![array(false, &[25, 2])], vec![], None, config());
    let mut different = config();
    different.chunk_elements = 16;
    let b = recipe("sum", vec![array(false, &[25, 2])], vec![], None, different);
    assert_ne!(a.identity(), b.identity());
    assert_eq!(a.geometry().chunks, 3);
    assert_eq!(b.geometry().chunks, 2);
    assert_eq!(a.work().uniform_bytes, 0);
    assert_eq!(a.work().single_dispatches, 1);
    assert_eq!(a.work().two_pass_dispatches, 2);
}

#[test]
fn provider_limits_and_invalid_geometry_are_rejected_without_execution() {
    for case in 0..6 {
        let mut cfg = config();
        match case {
            0 => cfg.workgroup_size = 0,
            1 => cfg.workgroup_size = 3,
            2 => cfg.chunk_elements = 0,
            3 => cfg.max_workgroup_storage_bytes = 8,
            4 => cfg.max_storage_binding_bytes = 4,
            5 => cfg.max_rank = 1,
            _ => unreachable!(),
        }
        assert!(
            try_recipe("sum", vec![array(false, &[25, 2])], vec![], None, cfg).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn resident_binding_limits_use_authentic_allocation_not_smaller_view_bytes() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![16, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let view = allocation
        .handle
        .with_view(allocation.handle.view().clone().with_shape(vec![4, 1]))
        .unwrap();
    let mut cfg = config();
    cfg.max_storage_binding_bytes = 64;
    let result = try_recipe(
        "sum",
        vec![Value::GpuTensor(view.clone())],
        vec![],
        None,
        cfg,
    );
    assert!(
        result.is_err(),
        "32 logical bytes cannot hide a 128-byte full-buffer binding"
    );
    cfg.max_storage_binding_bytes = 128;
    let accepted = try_recipe("sum", vec![Value::GpuTensor(view)], vec![], None, cfg).unwrap();
    assert_eq!(accepted.inputs()[0].source_payload_bytes, 32);
    assert_eq!(
        accepted.inputs()[0]
            .source_physical
            .unwrap()
            .requested_bytes(),
        128
    );
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[cfg(feature = "wgpu")]
#[test]
fn naga_validates_all_three_real_shaders_at_both_precisions() {
    for single in [true, false] {
        for name in ["sum", "mean", "min", "max"] {
            for omit in [false, true] {
                let recipe = recipe(
                    name,
                    vec![array(single, &[25, 3])],
                    axes(name, Value::Num(2.0), omit),
                    None,
                    config(),
                );
                validate_shaders(&recipe);
            }
        }
    }
    for extra in [
        recipe(
            "sum",
            vec![array(true, &[2, 3])],
            vec![Value::Num(4.0)],
            None,
            config(),
        ),
        recipe(
            "sum",
            vec![array(false, &[2, 0])],
            vec![Value::Num(1.0)],
            None,
            config(),
        ),
        recipe(
            "mean",
            vec![array(false, &[0, 3])],
            vec![Value::Num(1.0)],
            None,
            config(),
        ),
        recipe(
            "sum",
            vec![array(false, &[2, 3, 4])],
            vec![Value::String("all".into())],
            None,
            config(),
        ),
        recipe(
            "sum",
            vec![array(true, &[2, 1]), array(true, &[1, 3])],
            vec![],
            Some(PrimitiveOp::ElemMul),
            config(),
        ),
    ] {
        validate_shaders(&extra);
    }
}

#[cfg(feature = "wgpu")]
fn validate_shaders(recipe: &ReductionRecipe) {
    for source in [
        recipe.single_source(),
        recipe.pass1_source(),
        recipe.pass2_source(),
    ] {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}\n{source}"));
        let partial = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("RRPartial"))
            .unwrap()
            .1;
        let wgpu::naga::TypeInner::Struct { members, span } = &partial.inner else {
            panic!("partial state struct");
        };
        assert_eq!(*span, recipe.partial_layout().stride);
        assert_eq!(members[1].offset, recipe.partial_layout().count_offset);
        assert_eq!(members[2].offset, recipe.partial_layout().flags_offset);
    }
}
