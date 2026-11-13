use super::*;
use crate::builtins::common::integer_conversion::IntegerClassExt;
use runmat_value::NumericScalar;

mod allocation;
mod lookup;
mod owned;
mod stream;
pub(in crate::builtins::table) use allocation::dictionary_protocol_ingress;
use stream::{element_at, element_count, element_shape, Element, Query};

/// The constructor's selected binding retains its explicit Unbounded metadata
/// frontier. Borrow native elements through normalization and duplicate
/// resolution; only surviving entries are materialized into the Cell storage.
pub(in crate::builtins::table) fn dictionary_from_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "dictionary: expected keys and values, or key/value pairs",
        ));
    }
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut keys_are_cells = args.len() > 2;
    let mut values_are_cells = args.len() > 2;
    for pair in 0..args.len() / 2 {
        let key_arg = &args[pair * 2];
        let value_arg = &args[pair * 2 + 1];
        if args.len() == 2 {
            keys_are_cells = matches!(key_arg, Value::Cell(_));
            values_are_cells = matches!(value_arg, Value::Cell(_));
        } else {
            keys_are_cells &= matches!(key_arg, Value::Cell(_));
            values_are_cells &= matches!(value_arg, Value::Cell(_));
        }
        let count = element_count(key_arg);
        let rhs_count = element_count(value_arg);
        if !(rhs_count == 1 && count != 1)
            && (count != rhs_count
                || element_shape(key_arg).as_slice() != element_shape(value_arg).as_slice())
        {
            return Err(invalid_argument(
                "dictionary: keys and values must have the same size unless values is scalar",
            ));
        }
        for index in 0..count {
            keys.push(element_at(key_arg, index)?);
            values.push(element_at(
                value_arg,
                if rhs_count == 1 { 0 } else { index },
            )?);
        }
    }
    if !keys_are_cells {
        let prototype = keys.first().copied();
        for value in &mut keys {
            *value = value.normalize(prototype)?;
        }
    }
    if !values_are_cells {
        let prototype = values.first().copied();
        for value in &mut values {
            *value = value.normalize(prototype)?;
        }
    }
    let mut unique_keys: Vec<Element<'_>> = Vec::new();
    let mut unique_values = Vec::new();
    for (key, value) in keys.into_iter().zip(values) {
        let mut duplicate = None;
        for (index, existing) in unique_keys.iter().enumerate() {
            if existing.equal(key)? {
                duplicate = Some(index);
                break;
            }
        }
        if let Some(index) = duplicate {
            unique_values[index] = value;
        } else {
            unique_keys.push(key);
            unique_values.push(value);
        }
    }
    let keys = unique_keys
        .into_iter()
        .map(|value| value.materialize(None, None))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let values = unique_values
        .into_iter()
        .map(|value| value.materialize(None, None))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let count = keys.len();
    let mut object =
        runmat_value::UnboundedObjectBuilder::with_property_capacity(DICTIONARY_CLASS, 2);
    object.insert(
        "Keys".to_string(),
        Value::Cell(CellArray::new(keys, 1, count).map_err(invalid_variable)?),
    );
    object.insert(
        "Values".to_string(),
        Value::Cell(CellArray::new(values, 1, count).map_err(invalid_variable)?),
    );
    Ok(Value::Object(object.finish()))
}

pub(in crate::builtins::table) fn into_dictionary_object(
    value: Value,
    context: &str,
) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(DICTIONARY_CLASS) => Ok(object),
        other => Err(invalid_argument(format!(
            "{context}: expected dictionary, got {other:?}"
        ))),
    }
}

pub(in crate::builtins::table) fn dictionary_cells<'a>(
    object: &'a ObjectInstance,
    field: &str,
) -> BuiltinResult<&'a CellArray> {
    match object.properties().get(field) {
        Some(Value::Cell(cell)) => Ok(cell),
        Some(other) => Err(invalid_variable(format!(
            "dictionary: {field} storage must be a cell array, got {other:?}"
        ))),
        None => Err(invalid_variable(format!(
            "dictionary: missing {field} storage"
        ))),
    }
}

pub(in crate::builtins::table) fn dictionary_lookup_selector_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &ObjectInstance,
    selector: &crate::object::indexing::ObjectIndexSelector,
) -> BuiltinResult<crate::execution::RootedValue> {
    lookup::lookup(runtime, object, Query::Typed(selector))
}

pub(in crate::builtins::table) fn dictionary_assign_selector_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    object: ObjectInstance,
    selector: &crate::object::indexing::ObjectIndexSelector,
    rhs: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    owned::assign(runtime, object, Query::Typed(selector), rhs)
}

#[cfg(test)]
pub(in crate::builtins::table) fn dictionary_lookup(
    object: &ObjectInstance,
    payload: &Value,
) -> BuiltinResult<Value> {
    lookup::lookup(None, object, Query::Payload(payload)).map(|rooted| rooted.value().clone())
}
#[cfg(test)]
pub(in crate::builtins::table) fn dictionary_assign(
    object: ObjectInstance,
    payload: &Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    owned::assign(None, object, Query::Payload(payload), rhs).map(|rooted| rooted.value().clone())
}
#[cfg(test)]
pub(in crate::builtins::table) fn dictionary_keys_equal(left: &Value, right: &Value) -> bool {
    Element::Borrowed(left)
        .equal(Element::Borrowed(right))
        .unwrap()
}
#[cfg(test)]
fn value_elements(value: &Value) -> BuiltinResult<Vec<Value>> {
    (0..element_count(value))
        .map(|index| element_at(value, index)?.materialize(None, None))
        .collect()
}
#[cfg(test)]
fn dictionary_lookup_output(values: Vec<Value>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if values.len() == 1 {
        return Ok(values[0].clone());
    }
    lookup::output(
        None,
        values.len(),
        &shape,
        |index| &values[index],
        || Ok(()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntValue, IntegerStorage};

    #[test]
    fn dictionary_exact_integer_keys_do_not_saturate_wide_uint64_values() {
        let first = Value::Int(IntValue::U64(i64::MAX as u64 + 1));
        let second = Value::Int(IntValue::U64(i64::MAX as u64 + 2));
        let repeated = Value::Int(IntValue::U64(i64::MAX as u64 + 1));

        assert!(dictionary_keys_equal(&first, &repeated));
        assert!(!dictionary_keys_equal(&first, &second));
    }

    #[test]
    fn dictionary_value_elements_preserve_exact_integer_tensor_keys() {
        let wide = i64::MAX as u64 + 1;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
            .expect("integer tensor");

        let elements = value_elements(&Value::Tensor(tensor)).expect("elements");

        assert_eq!(
            elements,
            vec![
                Value::Int(IntValue::U64(wide)),
                Value::Int(IntValue::U64(u64::MAX))
            ]
        );
    }

    #[test]
    fn dictionary_lookup_distinguishes_wide_uint64_tensor_keys() {
        let first = i64::MAX as u64 + 1;
        let second = i64::MAX as u64 + 2;
        let keys = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![first, second]), vec![1, 2])
                .expect("integer keys"),
        );
        let values = Value::Cell(
            CellArray::new(
                vec![
                    Value::String("first".to_string().into()),
                    Value::String("second".to_string().into()),
                ],
                1,
                2,
            )
            .expect("values"),
        );
        let dictionary = dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&[
            keys, values,
        ]))
        .expect("dictionary");
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary object");
        };

        let found = dictionary_lookup(&object, &Value::Int(IntValue::U64(second))).expect("lookup");

        assert_eq!(found, Value::String("second".to_string().into()));
    }

    #[test]
    fn dictionary_scalar_values_expand_and_duplicate_keys_keep_last_in_place() {
        let keys = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 1]), vec![1, 3]).unwrap(),
        );
        let dictionary = dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&[
            keys,
            Value::Int(IntValue::I8(7)),
        ]))
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        let keys = dictionary_cells(&object, "Keys").unwrap();
        let values = dictionary_cells(&object, "Values").unwrap();
        assert_eq!(
            keys.data(),
            vec![Value::Int(IntValue::U16(1)), Value::Int(IntValue::U16(2))]
        );
        assert_eq!(
            values.data(),
            vec![Value::Int(IntValue::I8(7)), Value::Int(IntValue::I8(7))]
        );
    }

    #[test]
    fn dictionary_array_pairs_expand_and_convert_to_first_integer_class() {
        let dictionary = dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&[
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![10, 20]), vec![1, 2]).unwrap(),
            ),
            Value::Int(IntValue::U8(3)),
            Value::Int(IntValue::I32(30)),
        ]))
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        assert_eq!(
            dictionary_cells(&object, "Keys").unwrap().data(),
            vec![
                Value::Int(IntValue::I16(1)),
                Value::Int(IntValue::I16(2)),
                Value::Int(IntValue::I16(3))
            ]
        );
        assert_eq!(
            dictionary_lookup(&object, &Value::Int(IntValue::U64(3))).unwrap(),
            Value::Int(IntValue::U64(30))
        );
    }

    #[test]
    fn dictionary_vector_assignment_and_removal_preserve_exact_integer_keys() {
        let dictionary = dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&[
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![10, 20]), vec![1, 2]).unwrap(),
            ),
        ]))
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        let assigned = dictionary_assign(
            object,
            &Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![2, 3]), vec![1, 2]).unwrap(),
            ),
            Value::Int(IntValue::I32(99)),
        )
        .unwrap();
        let Value::Object(object) = assigned else {
            panic!("expected dictionary");
        };
        assert_eq!(
            dictionary_lookup(&object, &Value::Int(IntValue::U64(3))).unwrap(),
            Value::Int(IntValue::I16(99))
        );
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        let removed = dictionary_assign(object, &Value::Int(IntValue::U64(2)), empty).unwrap();
        let Value::Object(object) = removed else {
            panic!("expected dictionary");
        };
        assert!(dictionary_lookup(&object, &Value::Int(IntValue::U64(2))).is_err());
    }

    #[test]
    fn dictionary_vector_lookup_preserves_integer_class_and_query_shape() {
        let dictionary = dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&[
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![10, 20]), vec![1, 2]).unwrap(),
            ),
        ]))
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        let query =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![2, 1]), vec![2, 1]).unwrap());
        let Value::Tensor(found) = dictionary_lookup(&object, &query).unwrap() else {
            panic!("expected configured integer tensor output");
        };
        assert_eq!(found.shape(), [2, 1]);
        assert_eq!(
            found.integer_storage(),
            Some(&IntegerStorage::I16(vec![20, 10]))
        );
    }

    #[test]
    fn dictionary_vector_assignment_requires_matching_non_scalar_shape() {
        let dictionary = dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&[
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).unwrap()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![10, 20]), vec![1, 2]).unwrap(),
            ),
        ]))
        .unwrap();
        let Value::Object(object) = dictionary else {
            panic!("expected dictionary");
        };
        let keys =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap());
        let values = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![30, 40]), vec![1, 2]).unwrap(),
        );
        assert!(dictionary_assign(object, &keys, values).is_err());
    }
    #[test]
    fn dictionary_cell_keys_and_cell_results_follow_full_column_major_geometry() {
        use crate::call::FunctionArgumentView;
        let keys = Value::Cell(
            CellArray::new_with_shape(
                ["a", "b", "c", "d", "e", "f", "g", "h"]
                    .into_iter()
                    .map(Value::from)
                    .collect(),
                vec![2, 2, 2],
            )
            .unwrap(),
        );
        let values =
            Value::Tensor(Tensor::new((1..=8).map(f64::from).collect(), vec![2, 2, 2]).unwrap());
        let args = [keys.clone(), values];
        let Value::Object(dictionary) =
            dictionary_from_args(FunctionArgumentView::from_slice(&args)).unwrap()
        else {
            panic!("dictionary")
        };
        let Value::Tensor(result) = dictionary_lookup(&dictionary, &keys).unwrap() else {
            panic!("numeric lookup")
        };
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(
            result.materialize_f64(),
            vec![1., 2., 3., 4., 5., 6., 7., 8.]
        );
        // Independent keys expose the Cell row-major/page-major storage
        // permutation; a self-query alone can mask paired input mistakes.
        for (key, expected) in [
            ("a", 1.),
            ("b", 3.),
            ("c", 2.),
            ("d", 4.),
            ("e", 5.),
            ("f", 7.),
            ("g", 6.),
            ("h", 8.),
        ] {
            assert_eq!(
                dictionary_lookup(&dictionary, &Value::from(key)).unwrap(),
                Value::Num(expected)
            );
        }
        let values = vec![
            Value::Num(1.),
            Value::from("two"),
            Value::Bool(true),
            Value::Int(IntValue::U64(u64::MAX)),
        ];
        let Value::Cell(result) = dictionary_lookup_output(values.clone(), vec![2, 2]).unwrap()
        else {
            panic!("mixed cell lookup")
        };
        assert_eq!(
            result.iter_column_major().cloned().collect::<Vec<_>>(),
            values
        );
        assert_eq!(
            result.data(),
            &[
                values[0].clone(),
                values[2].clone(),
                values[1].clone(),
                values[3].clone()
            ]
        );
        let values = values.into_iter().cycle().take(8).collect::<Vec<_>>();
        let (runtime, ledger, _) = managed_dictionary_runtime(65_536);
        let output = lookup::output(
            Some(&runtime),
            values.len(),
            &[2, 2, 2],
            |index| &values[index],
            || Ok(()),
        )
        .unwrap();
        let output = crate::execution::RootedValue::in_context(output, &runtime).unwrap();
        let Value::Cell(cell) = output.value() else {
            panic!("managed mixed Cell")
        };
        assert_eq!(cell.shape(), &[2, 2, 2]);
        assert_eq!(
            cell.iter_column_major().cloned().collect::<Vec<_>>(),
            values
        );
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    fn singles(data: Vec<f32>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::from_f32(data, shape).unwrap())
    }
    fn single_dictionary(keys: Vec<f32>, values: Vec<f32>) -> ObjectInstance {
        let count = keys.len();
        let args = [
            singles(keys, vec![1, count]),
            singles(values, vec![1, count]),
        ];
        let Value::Object(object) =
            dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&args)).unwrap()
        else {
            panic!("dictionary")
        };
        object
    }
    fn single_bits(value: &Value) -> Vec<u32> {
        let Value::Tensor(tensor) = value else {
            panic!("native Single scalar/array")
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        tensor
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }
    fn typed(value: Value) -> crate::object::indexing::ObjectIndexSelector {
        crate::object::indexing::ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                crate::object::indexing::ObjectIndexComponent::Value(value),
            ]),
        }
    }
    #[test]
    fn dictionary_single_constructor_lookup_bits_duplicates_and_scalar_alias() {
        let payload = f32::from_bits(0x7fa0_0042);
        let object = single_dictionary(vec![1., 2., 1.], vec![payload, -0., f32::from_bits(1)]);
        let keys = dictionary_cells(&object, "Keys").unwrap();
        assert_eq!(keys.data().len(), 2);
        assert_eq!(single_bits(&keys.data()[0]), [1f32.to_bits()]);
        let values = dictionary_cells(&object, "Values").unwrap();
        assert_eq!(single_bits(&values.data()[0]), [1]);
        assert_eq!(single_bits(&values.data()[1]), [(-0f32).to_bits()]);
        let query =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![2, 1]), vec![2, 1]).unwrap());
        let selected = dictionary_lookup(&object, &query).unwrap();
        assert_eq!(single_bits(&selected), [(-0f32).to_bits(), 1]);
        let Value::Tensor(selected) = selected else {
            unreachable!()
        };
        assert_eq!(selected.shape(), [2, 1]);
        let scalar = singles(vec![payload], vec![1, 1]);
        let args = [
            Value::Tensor(Tensor::new(vec![1., 2.], vec![1, 2]).unwrap()),
            scalar.clone(),
        ];
        let Value::Object(object) =
            dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&args)).unwrap()
        else {
            unreachable!()
        };
        let found = dictionary_lookup(&object, &Value::Num(2.)).unwrap();
        assert_eq!(single_bits(&found), [payload.to_bits()]);
        let (Value::Tensor(original), Value::Tensor(found)) = (scalar, found) else {
            unreachable!()
        };
        assert_eq!(
            original.as_f32_slice().unwrap().as_ptr(),
            found.as_f32_slice().unwrap().as_ptr()
        );
    }
    #[test]
    fn dictionary_single_assignment_deduplicates_pending_keys_and_casts_in_native_domain() {
        let wide = (1u64 << 63) + (1u64 << 39) + 1;
        assert_ne!(wide as f32, (wide as f64) as f32, "rounding witness");
        let object = single_dictionary(vec![1.], vec![0.]);
        let query = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![2, 2, wide]), vec![1, 3]).unwrap(),
        );
        let rhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![10, 20, wide]), vec![1, 3]).unwrap(),
        );
        let Value::Object(updated) = dictionary_assign(object.clone(), &query, rhs).unwrap() else {
            unreachable!()
        };
        assert_eq!(dictionary_cells(&updated, "Keys").unwrap().data().len(), 3);
        assert_eq!(
            single_bits(&dictionary_lookup(&updated, &Value::Int(IntValue::U64(2))).unwrap()),
            [20f32.to_bits()]
        );
        assert_eq!(
            single_bits(&dictionary_lookup(&updated, &Value::Int(IntValue::U64(wide))).unwrap()),
            [(wide as f32).to_bits()]
        );
        assert!(dictionary_lookup(&object, &Value::Num(2.)).is_err());
        let empty = Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
        let Value::Object(removed) = dictionary_assign(updated, &Value::Num(2.), empty).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(dictionary_cells(&removed, "Keys").unwrap().data().len(), 2);
        assert!(dictionary_lookup(&removed, &Value::Num(2.)).is_err());
    }

    struct DictionaryParallel(crate::context::host_effect::ManagedHostEffectLedger);
    impl crate::context::RuntimeParallelService for DictionaryParallel {
        fn supports(&self, _: crate::context::ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            crate::context::RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..crate::context::RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(
            &self,
        ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    fn managed_dictionary_runtime(
        capacity: u64,
    ) -> (
        crate::context::RuntimeContext,
        crate::context::host_effect::ManagedHostEffectLedger,
        runmat_execution::cancellation::CancellationSource,
    ) {
        use std::rc::Rc;
        let ledger =
            crate::context::host_effect::ManagedHostEffectLedger::new_with_epoch(capacity, 1);
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        )
        .with_service_ports(
            crate::context::RuntimeServicePorts::default()
                .with_parallel(Rc::new(DictionaryParallel(ledger.clone()))),
        );
        (runtime, ledger, cancellation)
    }
    #[test]
    fn dictionary_single_managed_lookup_and_retained_scalar_claims_outlive_call() {
        use crate::context::host_allocation::HostAllocationPlan;
        let (runtime, ledger, _) = managed_dictionary_runtime(1 << 24);
        let object = single_dictionary(vec![1., 2.], vec![f32::from_bits(0x7fa0_0042), -0.]);
        let selector = typed(Value::Tensor(
            Tensor::new(vec![2., 1.], vec![2, 1]).unwrap(),
        ));
        let selected =
            dictionary_lookup_selector_in_context(Some(&runtime), &object, &selector).unwrap();
        assert_eq!(single_bits(&selected), [(-0f32).to_bits(), 0x7fa0_0042]);
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            HostAllocationPlan::retained_tensor_bytes(NumericDType::F32, &[2, 1]).unwrap()
        );
        drop(selected);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let selector = typed(Value::Tensor(
            Tensor::new(vec![3., 3.], vec![1, 2]).unwrap(),
        ));
        let rhs = singles(vec![30., 31.], vec![1, 2]);
        let updated_root =
            dictionary_assign_selector_in_context(Some(&runtime), object, &selector, rhs).unwrap();
        let Value::Object(updated) = updated_root.value() else {
            unreachable!()
        };
        assert_eq!(dictionary_cells(updated, "Keys").unwrap().data().len(), 3);
        let key_alias = dictionary_cells(updated, "Keys").unwrap().data()[2].clone();
        let value_alias = dictionary_cells(updated, "Values").unwrap().data()[2].clone();
        assert_eq!(single_bits(&value_alias), [31f32.to_bits()]);
        let Value::Tensor(key) = &key_alias else {
            unreachable!()
        };
        assert!(key.shared_shape().is_admitted());
        drop(updated_root);
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            2 * HostAllocationPlan::retained_tensor_bytes(NumericDType::F32, &[1, 1]).unwrap()
        );
        drop(key_alias);
        drop(value_alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
    #[test]
    fn dictionary_managed_denial_and_cancellation_leave_original_unchanged() {
        let object = single_dictionary(vec![1., 2.], vec![10., 20.]);
        let selector = typed(Value::Tensor(
            Tensor::new(vec![2., 3.], vec![1, 2]).unwrap(),
        ));
        let (runtime, ledger, _) = managed_dictionary_runtime(1);
        assert!(dictionary_lookup_selector_in_context(Some(&runtime), &object, &selector).is_err());
        assert!(dictionary_assign_selector_in_context(
            Some(&runtime),
            object.clone(),
            &selector,
            singles(vec![99.], vec![1, 1])
        )
        .is_err());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(
            single_bits(&dictionary_lookup(&object, &Value::Num(2.)).unwrap()),
            [20f32.to_bits()]
        );
        assert_eq!(dictionary_cells(&object, "Keys").unwrap().data().len(), 2);
        let (runtime, ledger, cancellation) = managed_dictionary_runtime(1 << 24);
        assert!(dictionary_protocol_ingress(Some(&runtime))
            .unwrap_err()
            .message()
            .contains("shared standard-substruct"));
        cancellation.cancel(runmat_execution::CancellationReason::User);
        assert!(dictionary_lookup_selector_in_context(Some(&runtime), &object, &selector).is_err());
        assert!(dictionary_assign_selector_in_context(
            Some(&runtime),
            object.clone(),
            &selector,
            Value::Num(99.)
        )
        .is_err());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(
            single_bits(&dictionary_lookup(&object, &Value::Num(2.)).unwrap()),
            [20f32.to_bits()]
        );
    }

    #[test]
    fn dictionary_uniform_output_operational_failures_never_become_cell_fallback() {
        use crate::builtins::common::uniform_scalar_output::UniformScalarCollector;
        use std::cell::Cell;
        let values = [singles(vec![1.], vec![1, 1]), singles(vec![2.], vec![1, 1])];
        let (runtime, ledger, _) = managed_dictionary_runtime(1 << 24);
        let collector = UniformScalarCollector::in_context(&runtime, 2).unwrap();
        let scratch_only = ledger.snapshot().live_reserved_bytes;
        drop(collector);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let (denied, ledger, _) = managed_dictionary_runtime(scratch_only);
        assert!(
            lookup::output(Some(&denied), 2, &[1, 2], |index| &values[index], || Ok(())).is_err()
        );
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let (runtime, ledger, cancellation) = managed_dictionary_runtime(1 << 24);
        let checks = Cell::new(0);
        let result = lookup::output(
            Some(&runtime),
            2,
            &[1, 2],
            |index| &values[index],
            || {
                checks.set(checks.get() + 1);
                if checks.get() == 4 {
                    cancellation.cancel(runmat_execution::CancellationReason::User);
                }
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn dictionary_configured_object_classes_use_borrowed_identity_owners() {
        use runmat_types::StaticClassIdentity;
        let a = StaticClassIdentity::new("DictionaryObjectA");
        let b = StaticClassIdentity::new("DictionaryObjectB");
        let first = Value::Object(runmat_value::UnboundedObjectBuilder::new(a).finish());
        let same = Value::Object(runmat_value::UnboundedObjectBuilder::new(a).finish());
        let other = Value::Object(runmat_value::UnboundedObjectBuilder::new(b).finish());
        assert!(Element::Borrowed(&same)
            .normalize(Some(Element::Borrowed(&first)))
            .is_ok());
        assert!(Element::Borrowed(&other)
            .normalize(Some(Element::Borrowed(&first)))
            .is_err());
        let first_array = Value::ObjectArray(
            runmat_value::ObjectArray::new(a, vec![first.clone()], vec![1, 1]).unwrap(),
        );
        let other_array = Value::ObjectArray(
            runmat_value::ObjectArray::new(b, vec![other.clone()], vec![1, 1]).unwrap(),
        );
        assert!(Element::Borrowed(&other_array)
            .normalize(Some(Element::Borrowed(&first_array)))
            .is_err());
        runmat_gc::gc_test_context(|| {
            let first_target = runmat_gc::gc_allocate_rooted(first.clone()).unwrap();
            let other_target = runmat_gc::gc_allocate_rooted(other.clone()).unwrap();
            let first_handle = Value::HandleObject(runmat_value::HandleRef {
                class_name: a.into(),
                target: first_target.handle(),
                valid: true,
            });
            // Metadata alone must not hide the actual target's other class.
            let other_handle = Value::HandleObject(runmat_value::HandleRef {
                class_name: a.into(),
                target: other_target.handle(),
                valid: true,
            });
            assert!(Element::Borrowed(&first_handle)
                .normalize(Some(Element::Borrowed(&first_handle)))
                .is_ok());
            assert!(Element::Borrowed(&other_handle)
                .normalize(Some(Element::Borrowed(&first_handle)))
                .is_err());
        });
        // Cell construction intentionally preserves heterogeneous rich values;
        // lookup must still return a Cell rather than normalize/promote them.
        let args = [
            Value::Tensor(Tensor::new(vec![1., 2.], vec![1, 2]).unwrap()),
            Value::Cell(CellArray::new(vec![first, other], 1, 2).unwrap()),
        ];
        let Value::Object(object) =
            dictionary_from_args(crate::call::FunctionArgumentView::from_slice(&args)).unwrap()
        else {
            unreachable!()
        };
        let query = Value::Tensor(Tensor::new(vec![2., 1.], vec![1, 2]).unwrap());
        let Value::Cell(output) = dictionary_lookup(&object, &query).unwrap() else {
            panic!("heterogeneous Cell output")
        };
        assert_eq!(output.shape(), [1, 2]);
        let (Value::Object(other), Value::Object(first)) = (&output.data()[0], &output.data()[1])
        else {
            unreachable!()
        };
        assert!(other.class_name().is(b));
        assert!(first.class_name().is(a));
    }
}
