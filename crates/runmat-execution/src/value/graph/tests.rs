use super::*;
use crate::{Digest, ProgramFunctionId};

#[test]
fn callable_argument_revision_is_distinct_from_selected_value_revision() {
    assert_eq!(
        GraphRequiredSchema::from_version(GRAPH_REQUEST_SCHEMA_V10),
        Some(GraphRequiredSchema::V10)
    );
    assert_eq!(
        GraphRequiredSchema::from_version(GRAPH_REQUEST_SCHEMA_V11),
        Some(GraphRequiredSchema::SelectedValuesV11)
    );
    assert_eq!(
        GraphRequiredSchema::from_version(GRAPH_REQUEST_SCHEMA_V12),
        Some(GraphRequiredSchema::CallableArgumentsV12)
    );
    assert_eq!(
        GraphRequiredSchema::CallableArgumentsV12.required_capability(),
        Some(crate::resource::Capability::GraphCallableArgumentsV12)
    );
    assert!(GraphRequiredSchema::CallableArgumentsV12 > GraphRequiredSchema::SelectedValuesV11);
    assert_eq!(
        GraphRequiredSchema::from_version(GRAPH_REQUEST_SCHEMA_V13),
        Some(GraphRequiredSchema::ProjectRuntimePolicyV13)
    );
    assert_eq!(
        GraphRequiredSchema::ProjectRuntimePolicyV13.required_capability(),
        Some(crate::resource::Capability::GraphProjectRuntimePolicyV13)
    );
    assert_eq!(
        GraphRequiredSchema::from_version(GRAPH_REQUEST_SCHEMA_V14),
        Some(GraphRequiredSchema::InvocationControlsV14)
    );
    assert_eq!(
        GraphRequiredSchema::InvocationControlsV14.required_capability(),
        Some(crate::resource::Capability::GraphInvocationControlsV14)
    );
    assert!(
        GraphRequiredSchema::InvocationControlsV14 > GraphRequiredSchema::ProjectRuntimePolicyV13
    );
}

mod negative;

fn origin(function: u32) -> FunctionBindingOrigin {
    FunctionBindingOrigin {
        program: Digest::sha256(b"one-admitted-program"),
        function: ProgramFunctionId(function),
    }
}

fn owner(function: u32) -> GraphBoundOwner {
    GraphBoundOwner {
        unit: GraphProgramUnitId(0),
        origin: origin(function),
    }
}

fn selected_owner(unit: u16, function: u32) -> GraphBoundOwner {
    GraphBoundOwner {
        unit: GraphProgramUnitId(unit),
        origin: origin(function),
    }
}

fn leaf(value: &str) -> GraphNode {
    GraphNode::Leaf {
        value: ValuePayload::Inline(Box::new(InlineValue::String(value.into()))),
    }
}

fn layouts() -> HashMap<GraphBoundOwner, GraphLexicalLayout> {
    HashMap::from([
        (
            owner(1),
            GraphLexicalLayout {
                activation_parent: None,
                bindings: vec![(BindingId(7), GraphBindingKind::Owned)],
            },
        ),
        (
            owner(2),
            GraphLexicalLayout {
                activation_parent: Some(owner(1)),
                bindings: vec![(BindingId(7), GraphBindingKind::Inherited)],
            },
        ),
    ])
}

fn shared_cell_graph() -> PayloadGraph {
    let callable = GraphCallable {
        owner: GraphCallableOwner::Bound(owner(2)),
        qualified_name: "nested".into(),
        callable_digest: super::super::CallableValue::identity_digest(
            CallableOwner::Bound(origin(2)),
            "nested",
            CallableForm::Closure,
            runmat_types::FunctionSourceRole::Nested,
            None,
        ),
        form: CallableForm::Closure,
        source_role: runmat_types::FunctionSourceRole::Nested,
        source_file: None,
        captures: vec![],
        activation: Some(GraphNodeRef(1)),
    };
    PayloadGraph {
        // The target, argument, and nested structure all alias one callable.
        roots: vec![GraphNodeRef(0), GraphNodeRef(0), GraphNodeRef(3)],
        nodes: vec![
            GraphNode::Callable(callable),
            GraphNode::Activation(GraphActivation {
                owner: owner(1),
                parent: None,
                bindings: vec![GraphBinding {
                    binding: BindingId(7),
                    value: Some(GraphNodeRef(0)),
                }],
            }),
            leaf("shared"),
            GraphNode::Struct {
                fields: vec![
                    GraphField {
                        name: "callback".into(),
                        value: GraphNodeRef(0),
                    },
                    GraphField {
                        name: "label".into(),
                        value: GraphNodeRef(2),
                    },
                ],
            },
        ],
    }
}

#[test]
fn one_graph_keeps_target_argument_and_response_aliases_with_a_cycle() {
    let graph = shared_cell_graph();
    graph
        .validate_for_program(
            ValueLimits {
                max_depth: 2,
                ..ValueLimits::default()
            },
            None,
            &layouts(),
            1,
        )
        .expect("activation/callable cycle is one bounded graph component");
    assert_eq!(graph.roots[0], graph.roots[1]);
    assert!(graph
        .binding_origins(ValueLimits::default(), None, &layouts(), 1)
        .unwrap()
        .contains(&owner(2)));
    let output = PayloadGraph {
        roots: vec![GraphNodeRef(0), GraphNodeRef(0)],
        nodes: vec![leaf("one output object")],
    };
    output
        .validate_for_program(ValueLimits::default(), None, &HashMap::new(), 1)
        .unwrap();
    assert_eq!(output.roots[0], output.roots[1]);
}

#[test]
fn identical_program_origins_keep_distinct_selected_activations() {
    let nested = |unit, activation| {
        GraphNode::Callable(GraphCallable {
            owner: GraphCallableOwner::Bound(selected_owner(unit, 2)),
            qualified_name: "nested".into(),
            callable_digest: super::super::CallableValue::identity_digest(
                CallableOwner::Bound(origin(2)),
                "nested",
                CallableForm::Closure,
                runmat_types::FunctionSourceRole::Nested,
                None,
            ),
            form: CallableForm::Closure,
            source_role: runmat_types::FunctionSourceRole::Nested,
            source_file: None,
            captures: vec![],
            activation: Some(GraphNodeRef(activation)),
        })
    };
    let activation = |unit, value| {
        GraphNode::Activation(GraphActivation {
            owner: selected_owner(unit, 1),
            parent: None,
            bindings: vec![GraphBinding {
                binding: BindingId(7),
                value: Some(GraphNodeRef(value)),
            }],
        })
    };
    let graph = PayloadGraph {
        roots: vec![GraphNodeRef(0), GraphNodeRef(1)],
        nodes: vec![
            nested(0, 2),
            nested(1, 3),
            activation(0, 4),
            activation(1, 5),
            leaf("first issuance"),
            leaf("second issuance"),
        ],
    };
    let layouts = HashMap::from_iter([0, 1].into_iter().flat_map(|unit| {
        [
            (
                selected_owner(unit, 1),
                GraphLexicalLayout {
                    activation_parent: None,
                    bindings: vec![(BindingId(7), GraphBindingKind::Owned)],
                },
            ),
            (
                selected_owner(unit, 2),
                GraphLexicalLayout {
                    activation_parent: Some(selected_owner(unit, 1)),
                    bindings: vec![(BindingId(7), GraphBindingKind::Inherited)],
                },
            ),
        ]
    }));
    graph
        .validate_for_program(ValueLimits::default(), None, &layouts, 2)
        .expect("same origin in two admitted issuances is not one activation");
    let mut crossed = graph.clone();
    let GraphNode::Callable(callable) = &mut crossed.nodes[1] else {
        unreachable!()
    };
    callable.activation = Some(GraphNodeRef(2));
    assert!(crossed
        .validate_for_program(ValueLimits::default(), None, &layouts, 2)
        .is_err());
    assert!(graph
        .validate_for_program(ValueLimits::default(), None, &layouts, 1)
        .is_err());
}

#[test]
fn nested_registered_and_struct_graph_fields_use_sorted_typed_refs() {
    let graph = PayloadGraph {
        roots: vec![GraphNodeRef(0)],
        nodes: vec![
            GraphNode::Registered(GraphRegistered {
                kind: GraphRegisteredKind::ImmutableValueClass,
                type_identity: "pkg.Value".into(),
                schema_version: 1,
                fields: vec![GraphField {
                    name: "child".into(),
                    value: GraphNodeRef(1),
                }],
            }),
            GraphNode::Cell {
                shape: vec![1, 2],
                values: vec![GraphNodeRef(2), GraphNodeRef(2)],
            },
            leaf("same leaf"),
        ],
    };
    graph
        .validate_for_program(ValueLimits::default(), None, &HashMap::new(), 1)
        .expect("nested aliases");
    let mut bad = graph;
    let GraphNode::Registered(registered) = &mut bad.nodes[0] else {
        unreachable!()
    };
    registered.fields.push(GraphField {
        name: "child".into(),
        value: GraphNodeRef(2),
    });
    assert!(bad
        .validate_for_program(ValueLimits::default(), None, &HashMap::new(), 1)
        .is_err());
}

#[test]
fn quantitative_numeric_demand_expands_by_value_dag_fanout() {
    let graph = PayloadGraph {
        roots: vec![GraphNodeRef(0), GraphNodeRef(0)],
        nodes: vec![
            GraphNode::Cell {
                shape: vec![1, 2],
                values: vec![GraphNodeRef(1), GraphNodeRef(1)],
            },
            GraphNode::Leaf {
                value: ValuePayload::Inline(Box::new(InlineValue::Dense(
                    super::super::DenseValue {
                        element_type: super::super::ElementType::U8,
                        shape: vec![4],
                        little_endian_data: vec![1, 2, 3, 4],
                    },
                ))),
            },
        ],
    };
    graph
        .validate_for_program(ValueLimits::default(), None, &HashMap::new(), 1)
        .expect("valid by-value fanout");
    assert_eq!(graph.expanded_numeric_allocation_bytes().unwrap(), 16);

    let mut invalid = graph;
    invalid.nodes[1] = GraphNode::Cell {
        shape: vec![1],
        values: vec![GraphNodeRef(0)],
    };
    assert!(invalid.expanded_numeric_allocation_bytes().is_err());
}
