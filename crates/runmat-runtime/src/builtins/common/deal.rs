use crate::build_runtime_error;
pub use runmat_builtins::{
    DEAL_DESCRIPTOR, DEAL_EXTENSIONS, DEAL_INTEGER_CAPABILITIES, DEAL_RESIDENT_INPUT_EXTENSION,
};
use runmat_value::Value;
#[runmat_macros::runtime_builtin(
    name = "deal",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::deal"
)]
pub async fn deal_builtin_registered(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count().unwrap_or(1);
    let valid_count = rest.len() == 1 || rest.len() == out_count;
    if !valid_count {
        return Err(build_runtime_error(
            "deal: the number of outputs must match the number of inputs unless there is exactly one input",
        )
        .with_builtin("deal")
        .with_identifier("RunMat:deal:InputOutputCountMismatch")
        .build());
    }
    if rest.iter().any(crate::value_contains_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DEAL_RESIDENT_INPUT_EXTENSION,
            "deal",
        )?;
    }
    if out_count == 0 {
        return Ok(Value::OutputList(Vec::new()));
    }
    if rest.len() == 1 {
        let value = rest.into_iter().next().expect("one deal input");
        return if out_count == 1 {
            Ok(value)
        } else {
            Ok(Value::OutputList(vec![value; out_count]))
        };
    }
    Ok(Value::OutputList(rest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn nested_resident_aliases_pass_without_owner_access_and_counts_fail_first() {
        runmat_gc::gc_test_context(|| {
            let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap();
            let handle = allocation.handle;
            let nested = Value::Cell(
                runmat_value::CellArray::new(vec![Value::GpuTensor(handle)], 1, 1).unwrap(),
            );
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = call(vec![nested.clone(), Value::Num(1.0)], 1).unwrap_err();
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:deal:InputOutputCountMismatch")
                );
                let error = call(vec![nested.clone()], 0).unwrap_err();
                assert_eq!(
                    error.identifier(),
                    DEAL_RESIDENT_INPUT_EXTENSION.error_identifier
                );
            }
            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            assert_eq!(
                call(vec![nested.clone()], 2).unwrap(),
                Value::OutputList(vec![nested.clone(), nested])
            );
            runmat_gc::gc_collect_major().unwrap();
            assert_eq!(allocation.probe.retirements(), 1);
        });
    }

    fn call(inputs: Vec<Value>, outputs: usize) -> crate::BuiltinResult<Value> {
        let _outputs = crate::output_count::push_output_count(Some(outputs));
        block_on(deal_builtin_registered(inputs))
    }

    #[test]
    fn deal_distributes_matching_inputs_and_replicates_one_input() {
        assert_eq!(
            call(vec![Value::Num(3.0)], 3).unwrap(),
            Value::OutputList(vec![Value::Num(3.0); 3])
        );
        assert_eq!(
            call(vec![Value::Num(2.0), Value::from("x")], 2).unwrap(),
            Value::OutputList(vec![Value::Num(2.0), Value::from("x")])
        );
        assert_eq!(call(vec![Value::Num(4.0)], 1).unwrap(), Value::Num(4.0));
    }

    #[test]
    fn deal_enforces_public_input_output_count_rules() {
        for (inputs, outputs) in [
            (vec![Value::Num(1.0), Value::Num(2.0)], 1),
            (vec![Value::Num(1.0), Value::Num(2.0)], 3),
            (Vec::new(), 1),
        ] {
            let error = call(inputs, outputs).expect_err("count mismatch");
            assert_eq!(
                error.identifier(),
                Some("RunMat:deal:InputOutputCountMismatch")
            );
        }
        assert_eq!(call(Vec::new(), 0).unwrap(), Value::OutputList(Vec::new()));
        assert_eq!(
            call(vec![Value::Int(runmat_value::IntValue::U64(u64::MAX))], 0).unwrap(),
            Value::OutputList(Vec::new())
        );
    }

    #[test]
    fn deal_preserves_all_integer_storage_classes_exactly() {
        let storages = [
            IntegerStorage::I8(vec![-1, 2]),
            IntegerStorage::I16(vec![-1, 2]),
            IntegerStorage::I32(vec![-1, 2]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, 2]),
            IntegerStorage::U16(vec![0, 2]),
            IntegerStorage::U32(vec![0, 2]),
            IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap();
            let output = call(vec![Value::Tensor(tensor)], 1).unwrap();
            let Value::Tensor(output) = output else {
                panic!("integer tensor output");
            };
            assert_eq!(output.integer_storage(), Some(&storage));
            assert_eq!(output.shape, vec![1, 2]);
        }
    }

    #[test]
    fn deal_resident_pass_through_is_gated_before_provider_access() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            for storage in [
                IntegerStorage::I8(vec![-1, 2]),
                IntegerStorage::I16(vec![-1, 2]),
                IntegerStorage::I32(vec![-1, 2]),
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::U8(vec![0, 2]),
                IntegerStorage::U16(vec![0, 2]),
                IntegerStorage::U32(vec![0, 2]),
                IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993]),
            ] {
                let tensor = Tensor::new_integer(storage, vec![1, 2]).unwrap();
                let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                    .expect("upload exact resident integer tensor");
                let integer_type = runmat_accelerate_api::handle_integer_type(&handle)
                    .expect("resident integer class metadata");
                assert!(std::ptr::eq(
                    runmat_accelerate_api::provider_for_handle(&handle).expect("handle owner"),
                    provider
                ));
                {
                    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                    let error =
                        call(vec![Value::GpuTensor(handle.clone())], 2).expect_err("resident gate");
                    assert_eq!(
                        error.identifier(),
                        DEAL_RESIDENT_INPUT_EXTENSION.error_identifier
                    );
                }
                let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
                let result = call(vec![Value::GpuTensor(handle.clone())], 2).unwrap();
                let Value::OutputList(outputs) = result else {
                    panic!("resident replicated outputs");
                };
                assert_eq!(outputs.len(), 2);
                for output in outputs {
                    let Value::GpuTensor(output) = output else {
                        panic!("resident output");
                    };
                    assert_eq!(output, handle);
                    assert_eq!(
                        runmat_accelerate_api::handle_integer_type(&output),
                        Some(integer_type)
                    );
                    assert!(std::ptr::eq(
                        runmat_accelerate_api::provider_for_handle(&output).expect("output owner"),
                        provider
                    ));
                }
                let physical = handle.allocation_lease().downgrade();
                assert!(block_on(provider.download_integer(&handle)).is_ok());
                handle.release().expect("release resident integer tensor");
                assert!(physical.acquire().is_none());
            }
        });
    }
}
