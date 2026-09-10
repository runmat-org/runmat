use runmat_runtime::call::arguments::{
    ArgumentExpansionSpec, ArgumentSpec, MaterializedArgument, MaterializedExpansionSource,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub async fn build_expanded_args_from_specs(
    stack: &mut Vec<Value>,
    specs: &[ArgumentSpec],
    sequence_state: &mut crate::interpreter::dispatch::SequenceState,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<Vec<Value>, RuntimeError> {
    let capture_slots = specs
        .iter()
        .filter_map(|spec| match spec {
            ArgumentSpec::CapturedSequence { slot } => Some(*slot),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut captures = sequence_state.take_captures(stack, &capture_slots)?;
    let mut arguments = Vec::with_capacity(specs.len());
    for spec in specs.iter().rev() {
        match spec {
            ArgumentSpec::Single => {
                arguments.push(MaterializedArgument::Single(pop_argument(stack)?));
            }
            ArgumentSpec::Expansion(ArgumentExpansionSpec::CellContents {
                num_indices,
                expand_all,
            }) => {
                let mut indices = Vec::with_capacity(*num_indices);
                for _ in 0..*num_indices {
                    indices.push(pop_argument(stack)?);
                }
                indices.reverse();
                let base = pop_argument(stack)?;
                arguments.push(MaterializedArgument::Expansion(
                    MaterializedExpansionSource::CellContents {
                        base,
                        indices,
                        expand_all: *expand_all,
                    },
                ));
            }
            ArgumentSpec::Expansion(ArgumentExpansionSpec::ReturnedOutputs) => {
                arguments.push(MaterializedArgument::Expansion(
                    MaterializedExpansionSource::ReturnedOutputs(pop_argument(stack)?),
                ));
            }
            ArgumentSpec::Expansion(ArgumentExpansionSpec::Member(member)) => {
                arguments.push(MaterializedArgument::Expansion(
                    MaterializedExpansionSource::Member {
                        base: pop_argument(stack)?,
                        member: member.clone(),
                    },
                ));
            }
            ArgumentSpec::Expansion(ArgumentExpansionSpec::DynamicMember) => {
                let member = pop_argument(stack)?;
                let base = pop_argument(stack)?;
                arguments.push(MaterializedArgument::Expansion(
                    MaterializedExpansionSource::DynamicMember { base, member },
                ));
            }
            ArgumentSpec::CapturedSequence { slot } => {
                let values = captures.remove(slot).ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "RunMat:CommaSeparatedListState",
                        "captured call argument was consumed more than once",
                    )
                })?;
                arguments.push(MaterializedArgument::Sequence(
                    runmat_runtime::sequence::ValueSequence::comma_separated(values),
                ));
            }
        }
    }
    arguments.reverse();
    runmat_runtime::call::arguments::expand_arguments(runtime, arguments).await
}

fn pop_argument(stack: &mut Vec<Value>) -> Result<Value, RuntimeError> {
    stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))
}

#[cfg(test)]
mod tests {
    use super::build_expanded_args_from_specs;
    use crate::bytecode::program::ExecutionContext;
    use futures::executor::block_on;
    use runmat_hir::CallableIdentity;
    use runmat_hir::{QualifiedName, SymbolName};
    use runmat_runtime::call::arguments::ArgumentSpec;
    use runmat_runtime::call::identity::{
        external_qualified_display_name, external_qualified_identity,
    };
    use runmat_runtime::object::indexing::{
        ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath, OBJECT_PROTOCOL_SUBSASGN,
        OBJECT_PROTOCOL_SUBSREF,
    };
    use runmat_types::MemberAccess;
    use runmat_value::Value;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    #[test]
    fn object_subscript_path_serializes_standard_substruct_once() {
        let path = ObjectSubscriptPath::single(ObjectSubscript::braces(
            ObjectIndexSelector::IndexValues {
                components: vec![Value::Num(2.0).into()],
            },
        ));

        let encoded = path
            .to_standard_substruct_value()
            .expect("standard substruct");
        let path = runmat_runtime::object::indexing::parse_standard_substruct(&encoded)
            .expect("decoded substruct");
        let step = &path.steps()[0];
        assert_eq!(
            step.kind(),
            runmat_runtime::object::indexing::ObjectIndexKind::Brace
        );
        match step.selector_value().expect("selector") {
            Value::Cell(cell) => assert_eq!(cell.data[0].clone(), Value::Num(2.0)),
            other => panic!("expected selector cell, got {other:?}"),
        }
    }

    #[test]
    fn object_member_path_carries_standard_substruct_without_assignment_payload() {
        let path = ObjectSubscriptPath::single(ObjectSubscript::member("field"));

        let encoded = path
            .to_standard_substruct_value()
            .expect("standard substruct");
        let path = runmat_runtime::object::indexing::parse_standard_substruct(&encoded)
            .expect("decoded substruct");
        assert_eq!(
            path.steps()[0].kind(),
            runmat_runtime::object::indexing::ObjectIndexKind::Member
        );
        assert_eq!(
            path.steps()[0].selector_value().unwrap(),
            Value::String("field".into())
        );
    }

    #[test]
    fn external_qualified_identity_preserves_malformed_base_segment() {
        let identity = external_qualified_identity("pkg..Point", "origin");
        let CallableIdentity::ExternalName(QualifiedName(segments)) = identity else {
            panic!("expected external qualified identity");
        };
        assert_eq!(
            segments,
            vec![
                SymbolName("pkg..Point".to_string()),
                SymbolName("origin".to_string())
            ]
        );
    }

    #[test]
    fn external_qualified_identity_splits_well_formed_base_segments() {
        let identity = external_qualified_identity("pkg.Point", "origin");
        let CallableIdentity::ExternalName(QualifiedName(segments)) = identity else {
            panic!("expected external qualified identity");
        };
        assert_eq!(
            segments,
            vec![
                SymbolName("pkg".to_string()),
                SymbolName("Point".to_string()),
                SymbolName("origin".to_string())
            ]
        );
    }

    #[test]
    fn external_qualified_display_name_preserves_malformed_base_shape() {
        assert_eq!(
            external_qualified_display_name("pkg..Point", "origin"),
            "pkg..Point.origin"
        );
    }

    #[test]
    fn external_qualified_display_name_renders_well_formed_qualified_name() {
        assert_eq!(
            external_qualified_display_name("pkg.Point", "origin"),
            "pkg.Point.origin"
        );
    }

    #[test]
    fn typed_subsref_resolution_includes_inherited_method_metadata() {
        let parent_name =
            runmat_types::ClassIdentity::new(unique_class_name("vm_subsref_parent")).unwrap();
        let child_name =
            runmat_types::ClassIdentity::new(unique_class_name("vm_subsref_child")).unwrap();
        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            OBJECT_PROTOCOL_SUBSREF.into(),
            runmat_runtime::class_registry::RuntimeMethod {
                name: OBJECT_PROTOCOL_SUBSREF.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "subsref_impl".into(),
                implicit_class_argument: None,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: parent_name.clone(),
                parent: None,
                properties: HashMap::new(),
                methods: parent_methods,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: child_name.clone(),
                parent: Some(parent_name),
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
        );

        let child = Value::Object(runmat_value::ObjectInstance::new(child_name));
        assert!(matches!(
            runmat_runtime::object::protocol::resolve_object_protocol(
                &child,
                runmat_runtime::object::protocol::ObjectProtocol::Subsref,
                &runmat_runtime::object::protocol::ObjectAccessContext::default(),
            )
            .unwrap(),
            runmat_runtime::object::protocol::ProtocolResolution::Method(_)
        ));
    }

    #[test]
    fn typed_subsasgn_resolution_includes_inherited_method_metadata() {
        let parent_name =
            runmat_types::ClassIdentity::new(unique_class_name("vm_subsasgn_parent")).unwrap();
        let child_name =
            runmat_types::ClassIdentity::new(unique_class_name("vm_subsasgn_child")).unwrap();
        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            OBJECT_PROTOCOL_SUBSASGN.into(),
            runmat_runtime::class_registry::RuntimeMethod {
                name: OBJECT_PROTOCOL_SUBSASGN.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "subsasgn_impl".into(),
                implicit_class_argument: None,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: parent_name.clone(),
                parent: None,
                properties: HashMap::new(),
                methods: parent_methods,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: child_name.clone(),
                parent: Some(parent_name),
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
        );

        let child = Value::Object(runmat_value::ObjectInstance::new(child_name));
        assert!(matches!(
            runmat_runtime::object::protocol::resolve_object_protocol(
                &child,
                runmat_runtime::object::protocol::ObjectProtocol::Subsasgn,
                &runmat_runtime::object::protocol::ObjectAccessContext::default(),
            )
            .unwrap(),
            runmat_runtime::object::protocol::ProtocolResolution::Method(_)
        ));
    }

    #[test]
    fn build_expanded_args_from_specs_supports_cell_index_expansion() {
        let mut stack = vec![
            Value::Cell(
                runmat_value::CellArray::new(vec![Value::Num(9.0), Value::Num(2.0)], 1, 2).unwrap(),
            ),
            Value::Num(1.0),
        ];
        let specs = vec![ArgumentSpec::Expansion(
            runmat_runtime::call::arguments::ArgumentExpansionSpec::CellContents {
                num_indices: 1,
                expand_all: false,
            },
        )];
        let mut sequences = crate::interpreter::dispatch::SequenceState::default();
        let expanded = block_on(build_expanded_args_from_specs(
            &mut stack,
            &specs,
            &mut sequences,
            &ExecutionContext::default().runtime,
        ))
        .expect("expanded args");
        assert_eq!(expanded, vec![Value::Num(9.0)]);

        let mut stack = vec![
            Value::Cell(
                runmat_value::CellArray::new(vec![Value::Num(9.0), Value::Num(2.0)], 1, 2).unwrap(),
            ),
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ];
        let mut sequences = crate::interpreter::dispatch::SequenceState::default();
        let expanded = block_on(build_expanded_args_from_specs(
            &mut stack,
            &specs,
            &mut sequences,
            &ExecutionContext::default().runtime,
        ))
        .expect("expanded args");
        assert_eq!(expanded, vec![Value::Num(9.0), Value::Num(2.0)]);
    }
}
