use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_target_and_selector_low_rank_shapes_keep_exact_cardinality() {
    for (shape, values) in [(Vec::new(), vec![1.0]), (vec![2], vec![1.0, 2.0])] {
        let value = Value::Tensor(Tensor::new(values.clone(), shape.clone()).unwrap());
        assert_eq!(target_shape_borrowed(&value).unwrap().as_slice(), shape);
        assert_eq!(
            single_subscript_count(&value, &value, 0, 1).unwrap(),
            values.len()
        );
    }
    let complex = Value::ComplexTensor(
        runmat_value::ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, 4.0)], vec![2]).unwrap(),
    );
    assert_eq!(target_shape_borrowed(&complex).unwrap().as_slice(), &[2]);
}

#[test]
fn final_colon_folds_trailing_dimensions_including_empty_extents() {
    for (shape, count) in [(vec![2, 3, 4], 24), (vec![2, 3, 0], 0)] {
        let target =
            Value::Cell(CellArray::new_with_shape(vec![Value::Num(0.0); count], shape).unwrap());
        let path = substruct("{}", cell(vec![Value::String(":".into()); 2]));
        assert_eq!(
            call(target, path, Value::String("Expression".into())),
            Value::Num(count as f64)
        );
    }
}

#[test]
fn character_colon_uses_full_nd_shape() {
    let target = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a'; 6], vec![1, 2, 3])
            .unwrap(),
    );
    assert_eq!(
        single_subscript_count(&target, &Value::String(":".into()), 0, 1).unwrap(),
        6
    );
    assert_eq!(
        single_subscript_count(&target, &Value::String(":".into()), 1, 2).unwrap(),
        6
    );
}

#[test]
fn typed_default_cardinality_reuses_public_selector_policy_without_roundtrip() {
    use crate::object::indexing::{
        ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
    };
    let target = Value::Tensor(Tensor::new(vec![1.; 6], vec![2, 3]).unwrap());
    for (selectors, expected) in [
        (vec![], 0),
        (vec![ObjectIndexComponent::Colon], 6),
        (
            vec![ObjectIndexComponent::Colon, ObjectIndexComponent::Colon],
            6,
        ),
        (vec![Value::Bool(false).into()], 0),
        (vec![Value::Bool(true).into()], 1),
        (
            vec![Value::CharArray(runmat_value::CharArray::new_row("\u{2003}:\t")).into()],
            6,
        ),
    ] {
        let path = ObjectSubscriptPath::new(vec![
            ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(selectors),
            }),
            ObjectSubscript::member("later_step"),
        ])
        .unwrap();
        assert_eq!(
            default_path_num_arguments(&target, &path).unwrap(),
            expected
        );
        let ordinary = path.to_standard_substruct_value().unwrap();
        let levels = parse_subscript_levels(&ordinary).unwrap();
        assert_eq!(
            default_num_arguments(&target, &levels, IndexingContext::Assignment).unwrap(),
            expected
        );
    }
    let malformed = Value::CharArray(
        runmat_value::CharArray::from_code_units(
            [0xD83D, 0xDE00, 0xD800]
                .into_iter()
                .map(runmat_value::Utf16CodeUnit::from_raw)
                .collect(),
            1,
            3,
        )
        .unwrap(),
    );
    let error = single_subscript_count(&target, &malformed, 0, 1).unwrap_err();
    let source = error
        .source
        .as_deref()
        .unwrap()
        .downcast_ref::<runmat_value::Utf16DecodeError>()
        .unwrap();
    assert_eq!(source.index, 2);
    assert_eq!(source.unpaired_surrogate, 0xD800);
    assert!(single_subscript_count(
        &target,
        &Value::CharArray(runmat_value::CharArray::new_row(": :")),
        0,
        1
    )
    .is_err());
}

#[test]
fn contextual_default_cardinality_requires_no_descriptor_or_shape_allocation() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use crate::object::indexing::{
        ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
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
    let ledger = ManagedHostEffectLedger::new(1);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
        );
    let target = Value::Tensor(Tensor::new(vec![1.; 6], vec![2, 3]).unwrap());
    let path = ObjectSubscriptPath::new(vec![ObjectSubscript::braces(
        ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                ObjectIndexComponent::Colon,
            ]),
        },
    )])
    .unwrap();
    let result = block_on(object_path_cardinality_in_context(
        &runtime,
        &target,
        &path,
        IndexingContext::Assignment,
        &crate::object::protocol::ObjectAccessContext::default(),
    ))
    .unwrap();
    assert_eq!(result.count, 6);
    assert!(result.method.is_none());
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}
