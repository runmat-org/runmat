use super::*;

#[test]
fn contextual_standard_substruct_preserves_shapes_and_retires_physical_claims() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
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
    let context = |ledger: &ManagedHostEffectLedger| {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            )
    };
    let paths = [
        vec![ObjectSubscript::member("payload")],
        vec![ObjectSubscript::parentheses(
            ObjectIndexSelector::ScalarIndices {
                indices: vec![1, 3],
            },
        )],
        vec![
            ObjectSubscript::member("payload"),
            ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                    ObjectIndexComponent::Colon,
                    Value::Num(2.).into(),
                ]),
            }),
            ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::empty(),
            }),
        ],
    ];
    for steps in paths {
        let path = ObjectSubscriptPath::new(steps).unwrap();
        let expected = path.to_standard_substruct_value().unwrap();
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let runtime = context(&ledger);
        let result = path
            .to_standard_substruct_value_in_context(&runtime)
            .unwrap();
        assert_eq!(result.value(), &expected);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(result);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let denied = ManagedHostEffectLedger::new(1);
        assert!(path
            .to_standard_substruct_value_in_context(&context(&denied))
            .is_err());
        assert_eq!(denied.snapshot().live_claimed_bytes, 0);
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn standard_substruct_path_is_an_ordered_typed_structure_array() {
    let path = ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("payload"),
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                Value::Num(2.0).into(),
                ObjectIndexComponent::Colon,
            ]),
        }),
    ])
    .unwrap();
    let Value::StructArray(array) = path.to_standard_substruct_value().unwrap() else {
        panic!("a multi-step subscript must remain a typed structure array");
    };
    assert_eq!(array.shape(), &[1, 2]);
    assert_eq!(array.field_names().collect::<Vec<_>>(), ["type", "subs"]);
    assert_eq!(
        array.field_values("type").unwrap(),
        [Value::String(".".into()), Value::String("()".into())]
    );
    assert_eq!(
        array.field_values("subs").unwrap()[0],
        Value::String("payload".into())
    );
}

#[test]
fn object_subscript_paths_are_nonempty_and_preserve_multiple_steps() {
    let error = ObjectSubscriptPath::new(Vec::new()).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:InvalidObjectSubscriptPath")
    );

    let path = ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("first"),
        ObjectSubscript::member("second"),
    ])
    .unwrap();
    let encoded = path.to_standard_substruct_value().unwrap();
    let decoded = parse_standard_substruct(&encoded).unwrap();
    assert_eq!(decoded.steps().len(), 2);
}
