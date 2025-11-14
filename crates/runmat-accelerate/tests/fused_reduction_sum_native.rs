#![cfg(feature = "wgpu")]

use runmat_accelerate::fusion::{detect_fusion_groups, FusionPlan};
use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, PrimitiveOp,
    ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarBinding, VarKind,
};
use runmat_builtins::{Type, Value};
use std::collections::HashMap;

fn sum_native_graph() -> AccelGraph {
    let p = 8usize;
    let c = 16usize;
    let mut values: Vec<ValueInfo> = Vec::new();

    // x input
    let v_x: ValueId = 0;
    values.push(ValueInfo {
        id: v_x,
        origin: ValueOrigin::Variable {
            kind: VarKind::Global,
            index: 0,
        },
        ty: Type::tensor(),
        shape: ShapeInfo::Tensor(vec![Some(p), Some(c)]),
        constant: None,
    });

    // mul output
    let v_mul: ValueId = 1;
    values.push(ValueInfo {
        id: v_mul,
        origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
        ty: Type::tensor(),
        shape: ShapeInfo::Tensor(vec![Some(p), Some(c)]),
        constant: None,
    });

    // dim constant (1)
    let v_dim: ValueId = 2;
    values.push(ValueInfo {
        id: v_dim,
        origin: ValueOrigin::Constant,
        ty: Type::Num,
        shape: ShapeInfo::Scalar,
        constant: Some(Value::Num(1.0)),
    });

    // 'native' string constant
    let v_native: ValueId = 3;
    values.push(ValueInfo {
        id: v_native,
        origin: ValueOrigin::Constant,
        ty: Type::String,
        shape: ShapeInfo::Scalar,
        constant: Some(Value::String("native".into())),
    });

    // sum output
    let v_sum: ValueId = 4;
    values.push(ValueInfo {
        id: v_sum,
        origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
        ty: Type::tensor(),
        shape: ShapeInfo::Tensor(vec![Some(c)]),
        constant: None,
    });

    let mul_node = AccelNode {
        id: 0,
        label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
        category: AccelOpCategory::Elementwise,
        inputs: vec![v_x, v_x],
        outputs: vec![v_mul],
        span: InstrSpan { start: 0, end: 0 },
        tags: vec![AccelGraphTag::Elementwise],
    };

    let sum_node = AccelNode {
        id: 1,
        label: AccelNodeLabel::Builtin { name: "sum".into() },
        category: AccelOpCategory::Reduction,
        inputs: vec![v_mul, v_dim, v_native],
        outputs: vec![v_sum],
        span: InstrSpan { start: 1, end: 1 },
        tags: vec![AccelGraphTag::Reduction],
    };

    let mut var_bindings = HashMap::new();
    var_bindings.insert(
        v_x,
        VarBinding {
            kind: VarKind::Global,
            index: 0,
        },
    );

    AccelGraph {
        nodes: vec![mul_node, sum_node],
        values,
        var_bindings,
    }
}

#[test]
fn fused_sum_with_native_arg_supported() {
    let graph = sum_native_graph();
    let groups = detect_fusion_groups(&graph);
    assert_eq!(groups.len(), 1);
    let fusion_plan = FusionPlan::from_graph(&graph, &groups);
    let plan = fusion_plan.groups.first().expect("plan present").clone();
    assert!(plan.group.kind.is_reduction());
    assert_eq!(plan.reduction_data, Some(1));
    assert!(
        plan.kernel.supported,
        "reduction plan should be supported even with 'native' argument"
    );
    let shader = plan
        .generate_reduction_wgsl("f32")
        .expect("WGSL generation succeeds");
    assert!(
        shader.contains("@compute"),
        "reduction WGSL should contain kernel entry point"
    );
}
