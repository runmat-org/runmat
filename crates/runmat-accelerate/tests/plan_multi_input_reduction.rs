#![cfg(feature = "wgpu")]

use runmat_accelerate::fusion::{detect_fusion_groups, FusionPlan};
use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, PrimitiveOp,
    ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarBinding, VarKind,
};
use runmat_builtins::Value;
use std::collections::HashMap;

#[test]
fn plan_collects_two_parents_for_mul_before_sum_dim1() {
    // Build a minimal graph encoding: y = sum(x .* w, 1)
    //
    // Values:
    //   v_x, v_w: variables
    //   v_mul: node output (ElemMul)
    //   v_dim: constant 1.0
    //   v_sum: node output (Builtin sum)
    //
    // Nodes:
    //   mul (ElemMul) inputs [v_x, v_w] -> output v_mul
    //   sum (Builtin "sum") inputs [v_mul, v_dim] -> output v_sum
    //
    // Expect: detect a Reduction group; FusionPlan should record reduction_data=v_mul,
    // and plan.inputs should include both v_x and v_w (two external tensor dependencies).

    // Value ids
    let v_x: ValueId = 0;
    let v_w: ValueId = 1;
    let v_mul: ValueId = 2;
    let v_dim: ValueId = 3;
    let v_sum: ValueId = 4;

    // Build values table
    let mut values: Vec<ValueInfo> = Vec::new();
    // x variable
    values.push(ValueInfo {
        id: v_x,
        origin: ValueOrigin::Variable {
            kind: VarKind::Global,
            index: 0,
        },
        ty: runmat_builtins::Type::tensor(),
        shape: ShapeInfo::Tensor(vec![Some(8), Some(16)]), // rows=8, cols=16 (example)
        constant: None,
    });
    // w variable
    values.push(ValueInfo {
        id: v_w,
        origin: ValueOrigin::Variable {
            kind: VarKind::Global,
            index: 1,
        },
        ty: runmat_builtins::Type::tensor(),
        shape: ShapeInfo::Tensor(vec![Some(8), Some(16)]),
        constant: None,
    });
    // mul output placeholder (origin NodeOutput filled by node)
    values.push(ValueInfo {
        id: v_mul,
        origin: ValueOrigin::Unknown,
        ty: runmat_builtins::Type::tensor(),
        shape: ShapeInfo::Tensor(vec![Some(8), Some(16)]),
        constant: None,
    });
    // dim constant 1.0
    values.push(ValueInfo {
        id: v_dim,
        origin: ValueOrigin::Constant,
        ty: runmat_builtins::Type::Num,
        shape: ShapeInfo::Scalar,
        constant: Some(Value::Num(1.0)),
    });
    // sum output placeholder
    values.push(ValueInfo {
        id: v_sum,
        origin: ValueOrigin::Unknown,
        ty: runmat_builtins::Type::tensor(),
        shape: ShapeInfo::Tensor(vec![Some(16)]), // sum over rows -> vector of cols
        constant: None,
    });

    // Nodes
    let mul_node = AccelNode {
        id: 0,
        label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
        category: AccelOpCategory::Elementwise,
        inputs: vec![v_x, v_w],
        outputs: vec![v_mul],
        span: InstrSpan { start: 0, end: 0 },
        tags: vec![AccelGraphTag::Elementwise],
    };
    let sum_node = AccelNode {
        id: 1,
        label: AccelNodeLabel::Builtin {
            name: "sum".to_string(),
        },
        category: AccelOpCategory::Reduction,
        inputs: vec![v_mul, v_dim],
        outputs: vec![v_sum],
        span: InstrSpan { start: 1, end: 1 },
        tags: vec![AccelGraphTag::Reduction],
    };

    // Fix origins for node outputs
    values[v_mul as usize].origin = ValueOrigin::NodeOutput { node: 0, output: 0 };
    values[v_sum as usize].origin = ValueOrigin::NodeOutput { node: 1, output: 0 };

    // Var bindings for x and w
    let mut var_bindings: HashMap<ValueId, VarBinding> = HashMap::new();
    var_bindings.insert(
        v_x,
        VarBinding {
            kind: VarKind::Global,
            index: 0,
        },
    );
    var_bindings.insert(
        v_w,
        VarBinding {
            kind: VarKind::Global,
            index: 1,
        },
    );

    let graph = AccelGraph {
        nodes: vec![mul_node, sum_node],
        values,
        var_bindings,
    };

    let groups = detect_fusion_groups(&graph);
    assert!(
        groups.iter().any(|g| g.kind.is_reduction()),
        "should detect a reduction group"
    );
    let plan = FusionPlan::from_graph(&graph, &groups);
    let red = plan
        .groups
        .iter()
        .find(|g| g.group.kind.is_reduction())
        .expect("reduction group plan");

    // reduction_data should be the mul output
    assert_eq!(
        red.reduction_data,
        Some(v_mul),
        "reduction_data should be the mul output"
    );

    // Inputs should include both x and w (two external tensor deps)
    // Order is not critical; just check presence and len
    let inputs: Vec<ValueId> = red.inputs.clone();
    assert!(
        inputs.contains(&v_x) && inputs.contains(&v_w),
        "plan.inputs must include both x and w; got {:?}",
        inputs
    );
    assert_eq!(
        inputs.len(),
        2,
        "plan.inputs for sum(x.*w,1) should have two entries"
    );
}
