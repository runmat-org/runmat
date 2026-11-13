use super::*;
use crate::graph::{
    AccelNode, AccelOpCategory, InstrSpan, ShapeInfo, ValueInfo, ValueOrigin, VarKind,
};
use crate::PrimitiveOp;
use runmat_builtins::Type;
use runmat_value::Tensor;
use std::collections::{HashMap, HashSet};

fn value(id: ValueId, origin: ValueOrigin, constant: Option<Value>) -> ValueInfo {
    ValueInfo {
        id,
        origin,
        ty: Type::tensor(),
        shape: ShapeInfo::Unknown,
        constant,
    }
}

fn dims(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).unwrap())
}

fn mean_node(id: NodeId, input: ValueId, axes: ValueId, output: ValueId) -> AccelNode {
    AccelNode {
        id,
        label: AccelNodeLabel::Builtin {
            name: "mean".into(),
            origin: FusionCallOrigin::named("mean"),
        },
        category: AccelOpCategory::Reduction,
        inputs: vec![input, axes],
        outputs: vec![output],
        span: InstrSpan {
            start: id as usize,
            end: id as usize,
        },
        tags: Vec::new(),
    }
}

fn vec_graph(axes: &[f64]) -> AccelGraph {
    AccelGraph {
        nodes: vec![mean_node(0, 0, 1, 2)],
        values: vec![
            value(
                0,
                ValueOrigin::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                None,
            ),
            value(1, ValueOrigin::Constant, Some(dims(axes))),
            value(2, ValueOrigin::NodeOutput { node: 0, output: 0 }, None),
        ],
        var_bindings: HashMap::new(),
        node_bindings: HashMap::new(),
    }
}

fn nested_graph() -> AccelGraph {
    AccelGraph {
        nodes: vec![mean_node(0, 0, 1, 3), mean_node(1, 3, 2, 4)],
        values: vec![
            value(
                0,
                ValueOrigin::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                None,
            ),
            value(1, ValueOrigin::Constant, Some(Value::Num(2.0))),
            value(2, ValueOrigin::Constant, Some(Value::Num(3.0))),
            value(3, ValueOrigin::NodeOutput { node: 0, output: 0 }, None),
            value(4, ValueOrigin::NodeOutput { node: 1, output: 0 }, None),
        ],
        var_bindings: HashMap::new(),
        node_bindings: HashMap::new(),
    }
}

#[test]
fn source_mean_form_and_each_call_boundary_survive_peeling() {
    let mut nodes = Vec::new();
    let vec = peel_mean_dims(
        &vec_graph(&[3.0, 2.0]),
        2,
        &[3.0, 2.0],
        &HashSet::new(),
        &mut nodes,
    )
    .unwrap();
    assert_eq!(vec.source, 0);
    assert_eq!(nodes, [0]);
    let reverse = vec.route.unwrap();
    assert_eq!(
        serde_json::to_value(&reverse).unwrap()["form"]["VecDims"]["axes"],
        serde_json::json!(["Width", "Height"])
    );
    let ImageMeanForm::VecDims { call, axes } = reverse.form else {
        panic!("vecdim source form lost");
    };
    assert_eq!(axes, [ImageMeanAxis::Width, ImageMeanAxis::Height]);
    assert_eq!(call.output_class, ImageMeanOutput::Default);
    assert_eq!((call.input, call.axes_value, call.output), (0, 1, 2));

    let forward = peel_mean_dims(
        &vec_graph(&[2.0, 3.0]),
        2,
        &[3.0, 2.0],
        &HashSet::new(),
        &mut Vec::new(),
    )
    .unwrap()
    .route
    .unwrap();
    let ImageMeanForm::VecDims { axes, .. } = &forward.form else {
        panic!("forward vecdim source form lost");
    };
    assert_eq!(*axes, [ImageMeanAxis::Height, ImageMeanAxis::Width]);
    assert_eq!(
        serde_json::to_value(&forward).unwrap()["form"]["VecDims"]["axes"],
        serde_json::json!(["Height", "Width"])
    );

    nodes.clear();
    let nested =
        peel_mean_dims(&nested_graph(), 4, &[3.0, 2.0], &HashSet::new(), &mut nodes).unwrap();
    assert_eq!(nested.source, 0);
    assert_eq!(nodes, [1, 0]);
    let ImageMeanForm::Nested { inner, outer, .. } = nested.route.unwrap().form else {
        panic!("nested source form lost");
    };
    assert_eq!((inner.input, inner.output), (0, 3));
    assert_eq!((outer.input, outer.output), (3, 4));
    assert_eq!(inner.output_class, ImageMeanOutput::Default);
    assert_eq!(outer.output_class, ImageMeanOutput::Default);
}

#[test]
fn direct_mean_hook_plan_keeps_vecdim_one_call_and_nested_source_order() {
    for axes in [[2.0, 3.0], [3.0, 2.0]] {
        let route = exact_route(&vec_graph(&axes), 2).expect("exact source vecdim");
        let ImageMeanForm::VecDims { axes: source, .. } = &route.form else {
            panic!("vecdim source form");
        };
        assert_eq!(source[0], ImageMeanAxis::from_exact_dim(axes[0]).unwrap());
        assert_eq!(source[1], ImageMeanAxis::from_exact_dim(axes[1]).unwrap());
        let plan = route.direct_provider_plan([2, 4, 5]).unwrap();
        assert_eq!(
            plan.calls,
            vec![ImageMeanProviderCall {
                hook: ImageMeanProviderHook::ReduceNd([1, 2]),
                output_class: ImageMeanOutput::Default,
                shape: [2, 1, 1],
            }]
        );
    }

    let mut graph = nested_graph();
    graph.values.push(value(
        5,
        ValueOrigin::Constant,
        Some(Value::String("native".into())),
    ));
    graph.nodes[0].inputs.push(5);
    let route = exact_route(&graph, 4).expect("exact nested source");
    let plan = route.direct_provider_plan([2, 4, 5]).unwrap();
    assert_eq!(
        plan.calls,
        vec![
            ImageMeanProviderCall {
                hook: ImageMeanProviderHook::ReduceDim(1),
                output_class: ImageMeanOutput::Native,
                shape: [2, 1, 5],
            },
            ImageMeanProviderCall {
                hook: ImageMeanProviderHook::ReduceDim(2),
                output_class: ImageMeanOutput::Default,
                shape: [2, 1, 1],
            },
        ]
    );
}

#[test]
fn source_power_transcript_retains_square_and_final_pow_even_for_identity_gamma() {
    let mut graph = vec_graph(&[2.0, 3.0]);
    graph.values.extend([
        value(3, ValueOrigin::Constant, Some(Value::Num(2.0))),
        value(4, ValueOrigin::NodeOutput { node: 1, output: 0 }, None),
        value(5, ValueOrigin::NodeOutput { node: 2, output: 0 }, None),
        value(6, ValueOrigin::Constant, Some(Value::Num(1.0))),
        value(7, ValueOrigin::NodeOutput { node: 3, output: 0 }, None),
    ]);
    let power = |id, base, exponent, output| AccelNode {
        id,
        label: AccelNodeLabel::Primitive(PrimitiveOp::ElemPow),
        category: AccelOpCategory::Elementwise,
        inputs: vec![base, exponent],
        outputs: vec![output],
        span: InstrSpan {
            start: id as usize,
            end: id as usize,
        },
        tags: Vec::new(),
    };
    graph.nodes.push(power(1, 0, 3, 4));
    graph.nodes.push(mean_node(2, 4, 1, 5));
    graph.nodes.push(power(3, 5, 6, 7));
    let routes = ImageReductionRoutes {
        mu: exact_route(&graph, 2).unwrap(),
        variance: exact_route(&graph, 5).unwrap(),
        stage_producers: Some(ImageStageProducers {
            mu: 2,
            variance: 5,
            sigma: 5,
            pre_clamp_output: None,
            clamp_output: None,
            output: 7,
        }),
    };
    let source = ImageMeanAndPowerSource::from_graph(&graph, &routes, [2, 4, 5]).unwrap();
    assert_eq!(source.square.node, 1);
    assert_eq!(
        source.square.represented_exponent.to_bits(),
        2.0f64.to_bits()
    );
    assert_eq!(source.final_power.node, 3);
    assert_eq!(
        source.final_power.represented_exponent.to_bits(),
        1.0f64.to_bits()
    );
    assert_eq!(source.mu.calls.len(), 1);
    assert_eq!(source.variance.calls.len(), 1);

    graph.values[6].constant = Some(Value::Num(f64::from_bits(1.0f64.to_bits() + 1)));
    let nearby = ImageMeanAndPowerSource::from_graph(&graph, &routes, [2, 4, 5]).unwrap();
    assert_eq!(
        nearby.final_power.represented_exponent.to_bits(),
        1.0f64.to_bits() + 1,
        "matcher identity simplification cannot erase the source power"
    );
    graph.nodes[3].label = AccelNodeLabel::Primitive(PrimitiveOp::ElemMul);
    assert!(ImageMeanAndPowerSource::from_graph(&graph, &routes, [2, 4, 5]).is_none());
}

#[test]
fn approximate_or_unresolved_axes_never_create_staged_evidence() {
    let mut nodes = Vec::new();
    let near = vec_graph(&[2.4, 3.4]);
    assert!(peel_mean_dims(&near, 2, &[3.0, 2.0], &HashSet::new(), &mut nodes).is_none());

    let mut unresolved = vec_graph(&[2.0, 3.0]);
    unresolved.nodes[0].label = AccelNodeLabel::Builtin {
        name: "mean".into(),
        origin: FusionCallOrigin::NamedUnresolved,
    };
    assert!(peel_mean_dims(
        &unresolved,
        2,
        &[3.0, 2.0],
        &HashSet::new(),
        &mut Vec::new(),
    )
    .is_none());

    unresolved.nodes[0].label = AccelNodeLabel::Builtin {
        name: "mean".into(),
        origin: FusionCallOrigin::named("single"),
    };
    assert!(peel_mean_dims(
        &unresolved,
        2,
        &[3.0, 2.0],
        &HashSet::new(),
        &mut Vec::new(),
    )
    .is_none());
}

#[test]
fn native_control_is_recorded_at_its_own_mean_call() {
    let mut graph = nested_graph();
    graph.values.push(value(
        5,
        ValueOrigin::Constant,
        Some(Value::String("native".into())),
    ));
    graph.nodes[0].inputs.push(5);
    let peeled = peel_mean_dims(&graph, 4, &[3.0, 2.0], &HashSet::new(), &mut Vec::new()).unwrap();
    let ImageMeanForm::Nested { inner, outer, .. } = peeled.route.unwrap().form else {
        panic!("nested route lost");
    };
    assert_eq!(inner.output_class, ImageMeanOutput::Native);
    assert_eq!(outer.output_class, ImageMeanOutput::Default);
}

#[test]
fn mu_and_variance_routes_remain_independent_and_legacy_is_unqualified() {
    let mu = peel_mean_dims(
        &nested_graph(),
        4,
        &[3.0, 2.0],
        &HashSet::new(),
        &mut Vec::new(),
    )
    .unwrap()
    .route
    .unwrap();
    let variance = peel_mean_dims(
        &vec_graph(&[2.0, 3.0]),
        2,
        &[3.0, 2.0],
        &HashSet::new(),
        &mut Vec::new(),
    )
    .unwrap()
    .route
    .unwrap();
    let routes = ImageReductionRoutes {
        mu,
        variance,
        stage_producers: None,
    };
    assert!(!serde_json::to_string(&routes)
        .unwrap()
        .contains("stage_producers"));
    assert!(matches!(routes.mu.form, ImageMeanForm::Nested { .. }));
    assert!(matches!(
        routes.variance.form,
        ImageMeanForm::VecDims { .. }
    ));

    let old: crate::fusion::ImageNormalizePattern = serde_json::from_str(
        r#"{"input":0,"epsilon":{"Constant":0.1},"gain":null,"bias":null,"gamma":null}"#,
    )
    .unwrap();
    assert!(old.reduction_routes.is_none());
}

#[test]
fn post_mean_cast_path_keeps_exact_observation_values() {
    let mut graph = vec_graph(&[2.0, 3.0]);
    graph.nodes.push(AccelNode {
        id: 1,
        label: AccelNodeLabel::Builtin {
            name: "single".into(),
            origin: FusionCallOrigin::Named {
                binding: runmat_builtins::SINGLE_CATALOG_ENTRY
                    .binding_identity(&runmat_builtins::SINGLE_CATALOG_ENTRY.bindings[0]),
            },
        },
        category: AccelOpCategory::Other,
        inputs: vec![2],
        outputs: vec![3],
        span: InstrSpan { start: 1, end: 1 },
        tags: Vec::new(),
    });
    graph.values.push(value(
        3,
        ValueOrigin::NodeOutput { node: 1, output: 0 },
        None,
    ));
    let route = peel_mean_dims(&graph, 2, &[3.0, 2.0], &HashSet::new(), &mut Vec::new())
        .unwrap()
        .route
        .unwrap()
        .with_cast_path(&graph, 3, 2)
        .unwrap();
    assert_eq!(route.cast_paths[0].consumer_value, 3);
    assert_eq!(route.cast_paths[0].peeled_value, 2);
    assert_eq!(route.cast_paths[0].nodes, [1]);

    let mut unresolved = graph.clone();
    let AccelNodeLabel::Builtin { origin, .. } = &mut unresolved.nodes[1].label else {
        unreachable!("selected single cast")
    };
    *origin = FusionCallOrigin::NamedUnresolved;
    assert!(route.clone().with_cast_path(&unresolved, 3, 2).is_none());

    graph.nodes[1].inputs[0] = 3;
    assert!(route.with_cast_path(&graph, 3, 2).is_none());
}
