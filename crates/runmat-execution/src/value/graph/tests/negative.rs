use super::*;

#[path = "negative/edge_roles.rs"]
mod edge_roles;

#[test]
fn rejects_dangling_unused_and_malformed_graph_references() {
    let mut graph = shared_cell_graph();
    graph.nodes[3] = GraphNode::Cell {
        shape: vec![1, 2],
        values: vec![GraphNodeRef(0), GraphNodeRef(99)],
    };
    assert!(graph
        .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
        .is_err());
    graph.nodes[3] = GraphNode::Cell {
        shape: vec![1, 1],
        values: vec![GraphNodeRef(0)],
    };
    assert!(
        graph
            .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
            .is_err(),
        "unused leaf is rejected"
    );
    graph.roots.push(GraphNodeRef(2));
    graph
        .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
        .expect("reachable leaf");
    graph.nodes[1] = GraphNode::Activation(GraphActivation {
        owner: owner(1),
        parent: Some(GraphNodeRef(1)),
        bindings: vec![GraphBinding {
            binding: BindingId(7),
            value: Some(GraphNodeRef(0)),
        }],
    });
    assert!(
        graph
            .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
            .is_err(),
        "activation parent cannot cycle"
    );
}

#[test]
fn rejects_wrong_origin_layout_and_shared_budget_overruns() {
    let graph = shared_cell_graph();
    let mut wrong = layouts();
    wrong.remove(&owner(1));
    assert!(graph
        .validate_for_program(ValueLimits::default(), None, &wrong, 1)
        .is_err());
    let mut wrong = layouts();
    wrong.insert(
        owner(1),
        GraphLexicalLayout {
            activation_parent: None,
            bindings: vec![(BindingId(9), GraphBindingKind::Owned)],
        },
    );
    assert!(graph
        .validate_for_program(ValueLimits::default(), None, &wrong, 1)
        .is_err());
    let two_leaves = PayloadGraph {
        roots: vec![GraphNodeRef(0), GraphNodeRef(1)],
        nodes: vec![leaf("four"), leaf("more")],
    };
    assert!(two_leaves
        .validate_for_program(
            ValueLimits {
                max_text_bytes: 7,
                ..ValueLimits::default()
            },
            None,
            &HashMap::new(),
            1
        )
        .is_err());
    let field_and_leaf = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![
            GraphNode::Struct {
                fields: vec![GraphField {
                    name: "field".into(),
                    value: GraphNodeRef(1),
                }],
            },
            leaf("more"),
        ],
    };
    assert!(
        field_and_leaf
            .validate_for_program(
                ValueLimits {
                    max_text_bytes: 8,
                    ..ValueLimits::default()
                },
                None,
                &HashMap::new(),
                1
            )
            .is_err(),
        "metadata and leaf text share one limit"
    );
    let nested = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![
            GraphNode::Cell {
                shape: vec![1, 1],
                values: vec![GraphNodeRef(1)],
            },
            leaf("child"),
        ],
    };
    assert!(nested
        .validate_for_program(
            ValueLimits {
                max_depth: 0,
                ..ValueLimits::default()
            },
            None,
            &HashMap::new(),
            1
        )
        .is_err());
}

#[test]
fn rejects_unowned_value_cycles_and_hidden_recursive_leaf_values() {
    let value_cycle = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![GraphNode::Cell {
            shape: vec![1, 1],
            values: vec![GraphNodeRef(0)],
        }],
    };
    assert!(value_cycle
        .validate_for_program(ValueLimits::default(), None, &HashMap::new(), 1)
        .is_err());
    let hidden_cell = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![GraphNode::Leaf {
            value: ValuePayload::Inline(Box::new(InlineValue::Cell {
                shape: vec![1, 1],
                values: vec![ValuePayload::Inline(Box::new(InlineValue::Null))],
            })),
        }],
    };
    assert!(hidden_cell
        .validate_for_program(ValueLimits::default(), None, &HashMap::new(), 1)
        .is_err());
}

#[test]
fn graph_and_canonical_values_share_identity_validation() {
    let invalid_registered = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![GraphNode::Registered(GraphRegistered {
            kind: GraphRegisteredKind::ImmutableValueClass,
            type_identity: String::new(),
            schema_version: 1,
            fields: vec![],
        })],
    };
    assert!(invalid_registered
        .validate_for_program(ValueLimits::default(), None, &HashMap::new(), 1)
        .is_err());
    let canonical = ValuePayload::Inline(Box::new(InlineValue::ImmutableValueClass(
        super::super::super::RegisteredData {
            type_identity: String::new(),
            schema_version: 1,
            fields: vec![],
        },
    )));
    assert!(canonical.validate(ValueLimits::default()).is_err());

    let invalid_callable = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![GraphNode::Callable(GraphCallable {
            owner: GraphCallableOwner::Bound(owner(2)),
            qualified_name: String::new(),
            callable_digest: super::super::super::CallableValue::identity_digest(
                CallableOwner::Bound(origin(2)),
                "",
                CallableForm::Closure,
                runmat_types::FunctionSourceRole::Nested,
                None,
            ),
            form: CallableForm::Closure,
            source_role: runmat_types::FunctionSourceRole::Nested,
            source_file: None,
            captures: vec![],
            activation: None,
        })],
    };
    assert!(invalid_callable
        .validate_for_program(ValueLimits::default(), None, &layouts(), 1)
        .is_err());
}
