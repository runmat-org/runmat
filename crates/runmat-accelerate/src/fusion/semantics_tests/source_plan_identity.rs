use super::*;
use crate::fusion::{
    FusionPattern, FusionStackLayout, FusionStackValueBinding, FusionStoreMaterialization,
    ImageNormalizePattern, ImageScalar,
};
use crate::graph::{VarBinding, VarKind};

fn add_plan() -> FusionGroupPlan {
    plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn source_plan_identity_is_insertion_order_independent_and_bit_exact() {
    let mut left = add_plan();
    left.const_values.insert(0, Value::Num(-0.0));
    left.const_values.insert(
        1,
        Value::Tensor(
            runmat_value::Tensor::from_f32(vec![f32::from_bits(0x7fc0_0042)], vec![1, 1]).unwrap(),
        ),
    );
    let mut reordered = add_plan();
    reordered.const_values.insert(
        1,
        Value::Tensor(
            runmat_value::Tensor::from_f32(vec![f32::from_bits(0x7fc0_0042)], vec![1, 1]).unwrap(),
        ),
    );
    reordered.const_values.insert(0, Value::Num(-0.0));
    assert_eq!(
        exact_source_plan_identity(&left).unwrap(),
        exact_source_plan_identity(&reordered).unwrap()
    );

    let mut positive_zero = reordered.clone();
    positive_zero.const_values.insert(0, Value::Num(0.0));
    assert_ne!(
        exact_source_plan_identity(&left).unwrap(),
        exact_source_plan_identity(&positive_zero).unwrap()
    );
    let mut other_nan = reordered.clone();
    other_nan.const_values.insert(
        1,
        Value::Tensor(
            runmat_value::Tensor::from_f32(vec![f32::from_bits(0x7fc0_0043)], vec![1, 1]).unwrap(),
        ),
    );
    assert_ne!(
        exact_source_plan_identity(&left).unwrap(),
        exact_source_plan_identity(&other_nan).unwrap()
    );
    let mut wider = reordered;
    wider.const_values.insert(
        1,
        Value::Tensor(runmat_value::Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap()),
    );
    assert_ne!(
        exact_source_plan_identity(&left).unwrap(),
        exact_source_plan_identity(&wider).unwrap()
    );
    let mut reshaped = left.clone();
    reshaped.const_values.insert(
        1,
        Value::Tensor(
            runmat_value::Tensor::from_f32(vec![f32::from_bits(0x7fc0_0042)], vec![1, 1, 1])
                .unwrap(),
        ),
    );
    assert_ne!(
        exact_source_plan_identity(&left).unwrap(),
        exact_source_plan_identity(&reshaped).unwrap()
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn source_plan_identity_covers_stack_store_and_rejects_unsupported_constants() {
    let base = add_plan();
    let baseline = exact_source_plan_identity(&base).unwrap();
    let mut stack = base.clone();
    stack.group.stack_layout = Some(FusionStackLayout {
        required_stack_operands: 1,
        bindings: vec![FusionStackValueBinding {
            value_id: 0,
            stack_offset: 0,
        }],
        compiler_variable_bindings: Vec::new(),
    });
    assert_ne!(baseline, exact_source_plan_identity(&stack).unwrap());
    let legacy = serde_json::to_value(stack.group.stack_layout.as_ref().unwrap()).unwrap();
    assert!(legacy.get("compiler_variable_bindings").is_none());
    let decoded: FusionStackLayout = serde_json::from_value(legacy.clone()).unwrap();
    assert!(decoded.compiler_variable_bindings.is_empty());
    assert_eq!(serde_json::to_value(&decoded).unwrap(), legacy);
    let mut certified = stack.clone();
    let layout = certified.group.stack_layout.as_mut().unwrap();
    layout.compiler_variable_bindings = layout.bindings.clone();
    assert_ne!(
        exact_source_plan_identity(&stack).unwrap(),
        exact_source_plan_identity(&certified).unwrap()
    );

    let mut store = base.clone();
    store.materialized_stores.push(FusionStoreMaterialization {
        value_id: 2,
        binding: VarBinding {
            kind: VarKind::Global,
            index: 3,
        },
    });
    assert_ne!(baseline, exact_source_plan_identity(&store).unwrap());

    let mut unsupported = base;
    unsupported.const_values.insert(
        0,
        Value::Cell(runmat_value::CellArray::new(vec![], 0, 0).unwrap()),
    );
    assert!(exact_source_plan_identity(&unsupported).is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn source_plan_identity_bit_encodes_every_pattern_float_boundary() {
    let mut negative_zero = add_plan();
    negative_zero.group.pattern = Some(FusionPattern::PowerStepNormalize {
        lhs: 0,
        rhs: 1,
        epsilon: -0.0,
    });
    let mut positive_zero = negative_zero.clone();
    positive_zero.group.pattern = Some(FusionPattern::PowerStepNormalize {
        lhs: 0,
        rhs: 1,
        epsilon: 0.0,
    });
    assert_ne!(
        exact_source_plan_identity(&negative_zero).unwrap(),
        exact_source_plan_identity(&positive_zero).unwrap()
    );

    let image = |bits| {
        FusionPattern::ImageNormalize(ImageNormalizePattern {
            input: 0,
            epsilon: ImageScalar::Constant(f64::from_bits(bits)),
            gain: Some(ImageScalar::Constant(-0.0)),
            bias: Some(ImageScalar::Value(1)),
            gamma: None,
            clamp_zero: true,
            clamp_zero_value: None,
            bound_mean_axes: Vec::new(),
            requires_single_input: false,
            reduction_routes: None,
        })
    };
    let mut nan_a = add_plan();
    nan_a.pattern = Some(image(0x7ff8_0000_0000_0042));
    let mut nan_b = nan_a.clone();
    nan_b.pattern = Some(image(0x7ff8_0000_0000_0043));
    assert_ne!(
        exact_source_plan_identity(&nan_a).unwrap(),
        exact_source_plan_identity(&nan_b).unwrap()
    );
}
