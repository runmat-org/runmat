use super::*;

#[test]
fn activation_nodes_are_never_values_at_any_root_or_value_edge() {
    let rejects_value_reference = |graph: PayloadGraph| {
        let error = graph
            .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
            .unwrap_err();
        assert!(
            matches!(error, ContractError::Invalid { reason, .. } if reason == "value reference points to an activation"),
            "wrong edge role must fail before reachability or decoding"
        );
    };

    let mut root = shared_cell_graph();
    root.roots[0] = GraphNodeRef(1);
    rejects_value_reference(root);

    let mut cell = shared_cell_graph();
    cell.nodes[3] = GraphNode::Cell {
        shape: vec![1, 1],
        values: vec![GraphNodeRef(1)],
    };
    rejects_value_reference(cell);

    let mut structure = shared_cell_graph();
    let GraphNode::Struct { fields } = &mut structure.nodes[3] else {
        unreachable!()
    };
    fields[0].value = GraphNodeRef(1);
    rejects_value_reference(structure);

    let mut registered = shared_cell_graph();
    registered.nodes[3] = GraphNode::Registered(GraphRegistered {
        kind: GraphRegisteredKind::ImmutableValueClass,
        type_identity: "pkg.Value".into(),
        schema_version: 1,
        fields: vec![GraphField {
            name: "child".into(),
            value: GraphNodeRef(1),
        }],
    });
    rejects_value_reference(registered);

    let mut capture = shared_cell_graph();
    let GraphNode::Callable(callable) = &mut capture.nodes[0] else {
        unreachable!()
    };
    callable.captures.push(GraphNodeRef(1));
    rejects_value_reference(capture);

    let mut binding = shared_cell_graph();
    let GraphNode::Activation(activation) = &mut binding.nodes[1] else {
        unreachable!()
    };
    activation.bindings[0].value = Some(GraphNodeRef(1));
    rejects_value_reference(binding);

    shared_cell_graph()
        .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
        .expect("callable activation and activation parent remain valid owner edges");
}

#[test]
fn repeated_root_ordinals_consume_the_shared_wire_byte_budget() {
    let leaf = GraphNode::Leaf {
        value: ValuePayload::Inline(Box::new(InlineValue::Null)),
    };
    let mut graph = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![leaf],
    };
    let limit = ValueLimits {
        max_inline_bytes: 4,
        ..ValueLimits::default()
    };
    graph
        .validate_for_program(limit, None, &HashMap::new(), 1)
        .expect("one encoded root ordinal fits four bytes");
    graph.roots.push(GraphNodeRef(0));
    assert!(graph
        .validate_for_program(limit, None, &HashMap::new(), 1)
        .is_err());
}

#[test]
fn activation_cycle_does_not_hide_an_impossible_value_cycle() {
    let mut graph = shared_cell_graph();
    let GraphNode::Callable(callable) = &mut graph.nodes[0] else {
        unreachable!()
    };
    callable.captures.push(GraphNodeRef(4));
    graph.nodes.push(GraphNode::Cell {
        shape: vec![1, 1],
        values: vec![GraphNodeRef(4)],
    });
    let error = graph
        .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
        .unwrap_err();
    assert!(
        matches!(error, ContractError::Invalid { reason, .. } if reason == "value-only cycle cannot be constructed")
    );

    let legal = shared_cell_graph();
    legal
        .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
        .expect("the callable-activation cycle remains legal");
    let order = legal
        .validate_for_program_and_value_order(ValueLimits::default(), None, &layouts(), 1)
        .unwrap();
    assert!(
        order
            .iter()
            .position(|node| *node == GraphNodeRef(0))
            .unwrap()
            < order
                .iter()
                .position(|node| *node == GraphNodeRef(3))
                .unwrap()
    );
    assert!(!order.contains(&GraphNodeRef(1)));
}
