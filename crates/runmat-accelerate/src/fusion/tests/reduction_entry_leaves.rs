use super::*;

#[test]
fn reduction_keeps_external_objective_array_result_as_entry_leaf() {
    let tensor_shape = ShapeInfo::Tensor(vec![Some(3), Some(1)]);
    let values = vec![
        ValueInfo {
            id: 0,
            origin: ValueOrigin::Variable {
                kind: VarKind::Local,
                index: 0,
            },
            ty: Type::tensor(),
            shape: tensor_shape.clone(),
            constant: None,
        },
        ValueInfo {
            id: 1,
            origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
            ty: Type::tensor(),
            shape: tensor_shape.clone(),
            constant: None,
        },
        ValueInfo {
            id: 2,
            origin: ValueOrigin::NodeOutput { node: 1, output: 0 },
            ty: Type::tensor(),
            shape: tensor_shape.clone(),
            constant: None,
        },
        ValueInfo {
            id: 3,
            origin: ValueOrigin::Constant,
            ty: Type::Num,
            shape: ShapeInfo::Scalar,
            constant: Some(Value::Num(2.0)),
        },
        ValueInfo {
            id: 4,
            origin: ValueOrigin::NodeOutput { node: 2, output: 0 },
            ty: Type::tensor(),
            shape: tensor_shape.clone(),
            constant: None,
        },
        ValueInfo {
            id: 5,
            origin: ValueOrigin::NodeOutput { node: 3, output: 0 },
            ty: Type::Num,
            shape: ShapeInfo::Scalar,
            constant: None,
        },
    ];
    let nodes = vec![
        AccelNode {
            id: 0,
            label: AccelNodeLabel::Primitive(PrimitiveOp::UPlus),
            category: AccelOpCategory::Other,
            inputs: vec![],
            outputs: vec![1],
            span: InstrSpan { start: 9, end: 9 },
            tags: vec![],
        },
        AccelNode {
            id: 1,
            label: AccelNodeLabel::Primitive(PrimitiveOp::Sub),
            category: AccelOpCategory::Elementwise,
            inputs: vec![0, 1],
            outputs: vec![2],
            span: InstrSpan { start: 10, end: 10 },
            tags: vec![AccelGraphTag::Elementwise],
        },
        AccelNode {
            id: 2,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemPow),
            category: AccelOpCategory::Elementwise,
            inputs: vec![2, 3],
            outputs: vec![4],
            span: InstrSpan { start: 11, end: 11 },
            tags: vec![AccelGraphTag::Elementwise],
        },
        AccelNode {
            id: 3,
            label: AccelNodeLabel::Builtin {
                name: "sum".into(),
                origin: FusionCallOrigin::named("sum"),
            },
            category: AccelOpCategory::Reduction,
            inputs: vec![4],
            outputs: vec![5],
            span: InstrSpan { start: 12, end: 12 },
            tags: vec![],
        },
    ];
    let graph = AccelGraph {
        nodes,
        values,
        var_bindings: StdHashMap::new(),
        node_bindings: StdHashMap::new(),
    };
    let reduction = FusionGroup {
        id: 0,
        kind: FusionKind::Reduction,
        nodes: vec![3],
        shape: ShapeInfo::Scalar,
        span: InstrSpan { start: 12, end: 12 },
        pattern: None,
        stack_layout: None,
    };
    let plan = FusionGroupPlan::new(0, reduction.clone(), &graph);
    assert_eq!(
        plan.inputs,
        vec![4],
        "sum must consume the already-materialized power result, not dead x/vector leaves"
    );
    assert_eq!(plan.stack_pattern, vec![0]);
    assert_eq!(plan.operations.len(), 1);

    let mut with_power = reduction;
    with_power.nodes = vec![2, 3];
    with_power.span.start = 11;
    let plan = FusionGroupPlan::new(0, with_power, &graph);
    assert_eq!(
        plan.inputs,
        vec![2],
        "an in-group power may still fuse over its live materialized subtraction input"
    );
    assert_eq!(plan.operations.len(), 2);
}
