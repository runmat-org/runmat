use super::*;
use crate::object::indexing::ObjectIndexComponent;
use futures::executor::block_on;

#[test]
fn builder_cardinality_is_consumed_once_by_contextual_member_fallback_assignment() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::{cell::RefCell, rc::Rc, sync::Arc};
    struct Parallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for Parallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let state = self.0.snapshot();
            RuntimeParallelResources {
                cpu_millicores_available: 1000,
                memory_available_bytes: Some(state.free_bytes()),
                epoch: state.epoch,
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        );
    let _activation = runtime.enter();
    let trace = Rc::new(RefCell::new(vec!["before"]));
    let observed = Rc::clone(&trace);
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "member_fallback322").then_some(32201)
        })))
        .unwrap();
    let _invoker = crate::user_functions::install_local_semantic_function_invoker(Rc::new(
        move |runtime, function, arguments, requested| {
            assert_eq!(function, 32201);
            assert_eq!(requested, 1);
            assert!(runtime.is_some());
            assert_eq!(arguments.supplied().len(), 3);
            let Value::Struct(descriptor) = &arguments.supplied()[1] else {
                panic!("scalar descriptor")
            };
            assert_eq!(
                descriptor.fields.get("subs"),
                Some(&Value::String("field".into()))
            );
            observed.borrow_mut().push("cardinality");
            Box::pin(async move {
                let _arguments = arguments;
                crate::execution::RootedValueSequence::in_context(
                    crate::sequence::ValueSequence::single(Value::Num(1.))
                        .map_err(crate::sequence::sequence_error_to_runtime)?,
                    runtime.as_ref().unwrap(),
                )
            })
        },
    ))
    .unwrap();
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "MemberFallback322".into(),
        parent: None,
        properties: [(
            "field".into(),
            crate::class_registry::RuntimeProperty {
                name: "field".into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: runmat_types::MemberAccess::Public,
                set_access: runmat_types::MemberAccess::Public,
                default_value: None,
            },
        )]
        .into(),
        methods: [(
            "numArgumentsFromSubscript".into(),
            crate::class_registry::RuntimeMethod {
                name: "numArgumentsFromSubscript".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "member_fallback322".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]
        .into(),
    });
    let mut child = runmat_value::UnboundedObjectBuilder::new("MemberFallback322");
    child.insert("field".into(), Value::Num(0.));
    let mut root = runmat_value::UnboundedStructValueBuilder::new();
    root.insert("child", Value::Object(child.finish()));
    let root = Value::Struct(root.finish());
    let mut builder = SequenceDestinationBuilder::new_in_context(&runtime, &root).unwrap();
    block_on(builder.push_step_in_context(
        &runtime,
        AssignmentStepSpec::Member("child".into()),
        None,
    ))
    .unwrap();
    let prepared =
        block_on(builder.finish_in_context(&runtime, SequenceEndpointSpec::Member("field".into())))
            .unwrap();
    assert_eq!(prepared.cardinality(), 1);
    trace.borrow_mut().push("rhs");
    let result =
        block_on(prepared.assign_fixed_in_context(&runtime, root, Value::Num(9.), None)).unwrap();
    trace.borrow_mut().push("after");
    assert_eq!(*trace.borrow(), ["before", "cardinality", "rhs", "after"]);
    let Value::Struct(root) = result.value() else {
        panic!("preserved prefix")
    };
    let Some(Value::Object(child)) = root.fields.get("child") else {
        panic!("object child")
    };
    assert_eq!(child.property("field"), Some(&Value::Num(9.)));
    drop(result);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn colon_cannot_define_an_undefined_member_destination_shape() {
    let mut builder = SequenceDestinationBuilder::new(&Value::Num(0.0)).expect("root destination");
    block_on(builder.push_step(
        AssignmentStepSpec::Parentheses {
            selectors: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                ObjectIndexComponent::Colon,
            ]),
        },
        None,
    ))
    .expect("colon can be prepared against the placeholder scalar");
    let error = block_on(builder.finish(SequenceEndpointSpec::Member("value".into())))
        .expect_err("colon cannot define a new structure-array shape");
    assert_eq!(
        error.identifier(),
        Some("RunMat:UndefinedColonSequenceDestination")
    );
}
