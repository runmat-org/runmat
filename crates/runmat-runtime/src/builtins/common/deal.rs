use crate::build_runtime_error;
pub use runmat_builtins::{
    DEAL_DESCRIPTOR, DEAL_EXTENSIONS, DEAL_INTEGER_CAPABILITIES, DEAL_RESIDENT_INPUT_EXTENSION,
};
use runmat_value::Value;
#[runmat_macros::runtime_builtin(
    name = "deal",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::common::deal"
)]
pub async fn deal_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
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
        return Ok(Value::output_list(Vec::new()));
    }
    if rest.len() == 1 && out_count == 1 {
        return Ok(rest.into_iter().next().expect("one deal input"));
    }
    let arguments = rest.into_function_arguments();
    let outputs = crate::execution::RootedValues::transform_arguments(
        &arguments,
        out_count,
        None,
        runtime.as_ref(),
        |index| {
            Ok(arguments
                .get(if arguments.len() == 1 { 0 } else { index })
                .expect("validated deal output arity")
                .clone())
        },
    )?;
    Ok(outputs.with_owned_output_storage(Value::OutputList))
}

#[cfg(all(test, feature = "wgpu"))]
#[path = "deal/wgpu_tests.rs"]
mod wgpu_tests;

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
                Value::output_list(vec![nested.clone(), nested])
            );
            runmat_gc::gc_collect_major().unwrap();
            assert_eq!(allocation.probe.retirements(), 1);
        });
    }

    #[test]
    fn rejected_nested_resident_input_retains_no_hidden_allocation_owner() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, HostNumericDataOwned,
            HostNumericTensorOwned, IntegerElementType,
        };

        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::U64(vec![9_007_199_254_740_993]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
            .expect("own exact U64 fixture payload");
        let handle = allocation.handle;
        let weak_view = handle.downgrade();
        let nested = Value::Cell(
            runmat_value::CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1)
                .expect("nested resident fixture"),
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            Some(IntegerElementType::U64)
        );
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let mismatch =
                call(vec![nested.clone(), Value::Num(1.0)], 1).expect_err("count fails first");
            assert_eq!(
                mismatch.identifier(),
                Some("RunMat:deal:InputOutputCountMismatch")
            );
            let gated = call(vec![nested.clone()], 2).expect_err("resident input is gated");
            assert_eq!(
                gated.identifier(),
                DEAL_RESIDENT_INPUT_EXTENSION.error_identifier
            );
        }
        assert_eq!(allocation.probe.retirements(), 0);
        assert!(
            weak_view.acquire().is_ok(),
            "call errors must not retire the input"
        );
        drop(nested);
        drop(handle);
        assert_eq!(
            allocation.probe.retirements(),
            1,
            "rejected calls must not retain hidden output aliases"
        );
        assert!(weak_view.acquire().is_err());
    }

    #[test]
    fn replicated_resident_outputs_independently_retain_the_native_allocation() {
        use runmat_accelerate_api::{
            allocation::AllocationReleaseOutcome, test_support::TestAllocationOwner,
            GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned, IntegerElementType,
        };

        let allocation = TestAllocationOwner::new()
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::U64(vec![9_007_199_254_740_993]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
            .expect("own exact U64 fixture payload");
        let handle = allocation.handle;
        let weak_view = handle.downgrade();
        let input = Value::Cell(
            runmat_value::CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1)
                .expect("nested resident input"),
        );
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let Value::OutputList(outputs) = call(vec![input.clone()], 2).expect("replicate") else {
            panic!("two resident outputs");
        };
        let mut outputs = outputs
            .try_into_owned()
            .expect("unique unbounded fixture outputs")
            .into_iter();
        let first = outputs.next().expect("first replicated output");
        let second = outputs.next().expect("second replicated output");
        drop(outputs);
        drop(input);
        assert_eq!(
            handle.release().expect("release original input view"),
            AllocationReleaseOutcome::ReferenceReleased
        );
        assert_eq!(allocation.probe.retirements(), 0);
        drop(first);
        let retained = weak_view.acquire().expect("second output retains the view");
        assert_eq!(retained.shape(), &[1, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&retained),
            Some(IntegerElementType::U64)
        );
        drop(retained);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(second);
        assert_eq!(allocation.probe.retirements(), 1);
        assert!(weak_view.acquire().is_err());
    }

    #[test]
    fn replicated_managed_aliases_survive_gc_until_output_roots_retire() {
        runmat_gc::gc_test_context(|| {
            let allocation = runmat_gc::gc_allocate_rooted(Value::String("deal payload".into()))
                .expect("managed payload");
            let target = allocation.handle();
            let nested = Value::Cell(
                runmat_value::CellArray::new(
                    vec![Value::HandleObject(runmat_value::HandleRef {
                        class_name: "DealPayload".into(),
                        target,
                        valid: true,
                    })],
                    1,
                    1,
                )
                .expect("nested managed input"),
            );
            let result = call(vec![nested.clone()], 3).expect("replicate managed input");
            assert_eq!(result, Value::output_list(vec![nested.clone(); 3]));
            let output_roots =
                crate::execution::RootedValueHandles::for_values(std::iter::once(&result))
                    .expect("root returned aliases");
            allocation.unroot().expect("release input allocation root");
            drop(nested);
            runmat_gc::gc_collect_major().expect("collect while outputs are retained");
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_ok());
            drop(output_roots);
            drop(result);
            runmat_gc::gc_collect_major().expect("collect after output retirement");
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn replicated_nested_foreign_references_share_one_release_lease() {
        use runmat_value::{
            ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignRef, ForeignResourceKey,
            ForeignResourceRelease, ForeignTypeIdentity,
        };
        use std::sync::{Arc, Mutex};

        #[derive(Debug, Default)]
        struct Releases(Mutex<Vec<ForeignResourceKey>>);

        impl ForeignResourceRelease for Releases {
            fn release(&self, key: &ForeignResourceKey) {
                self.0
                    .lock()
                    .expect("release observations")
                    .push(key.clone());
            }
        }

        let releases = Arc::new(Releases::default());
        let key = ForeignResourceKey {
            host_identity: "deal-test-host".into(),
            handle: 17,
            generation: 3,
        };
        let reference = ForeignRef::managed(
            key.clone(),
            ForeignTypeIdentity {
                family: "native-ffi".into(),
                name: "opaque-resource".into(),
                version: 1,
            },
            ForeignOwnership::Shared,
            ForeignAffinity::OriginProcess,
            ForeignLifetime::Session,
            releases.clone(),
        );
        let weak = reference.downgrade().expect("managed foreign lease");
        let input = Value::Cell(
            runmat_value::CellArray::new(vec![Value::Foreign(reference)], 1, 1)
                .expect("nested foreign input"),
        );
        let Value::OutputList(outputs) = call(vec![input], 3).expect("foreign replication") else {
            panic!("expected replicated outputs");
        };
        assert_eq!(outputs.len(), 3);
        {
            let original = weak.upgrade().expect("outputs retain foreign lease");
            for output in &outputs {
                let Value::Cell(cell) = output else {
                    panic!("expected nested cell");
                };
                let Value::Foreign(alias) = &cell.data()[0] else {
                    panic!("expected foreign reference");
                };
                assert_eq!(alias, &original);
                assert!(alias.shares_managed_lease(&original));
            }
        }
        let mut outputs = outputs
            .try_into_owned()
            .expect("unique unbounded fixture outputs")
            .into_iter();
        drop(outputs.next_back());
        drop(outputs.next_back());
        assert!(weak.upgrade().is_some());
        assert!(releases.0.lock().expect("release observations").is_empty());
        drop(outputs);
        assert!(weak.upgrade().is_none());
        assert_eq!(*releases.0.lock().expect("release observations"), vec![key]);
    }

    fn call(inputs: Vec<Value>, outputs: usize) -> crate::BuiltinResult<Value> {
        let _outputs =
            crate::output_count::push_output_count(Some(outputs)).expect("output-count scope");
        let inputs = crate::execution::RootedValues::new(inputs)
            .expect("ordinary deal fixture arguments")
            .into();
        block_on(deal_builtin_registered(None, inputs))
    }

    #[test]
    fn deal_distributes_matching_inputs_and_replicates_one_input() {
        assert_eq!(
            call(vec![Value::Num(3.0)], 3).unwrap(),
            Value::output_list(vec![Value::Num(3.0); 3])
        );
        assert_eq!(
            call(vec![Value::Num(2.0), Value::from("x")], 2).unwrap(),
            Value::output_list(vec![Value::Num(2.0), Value::from("x")])
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
        assert_eq!(call(Vec::new(), 0).unwrap(), Value::output_list(Vec::new()));
        assert_eq!(
            call(vec![Value::Int(runmat_value::IntValue::U64(u64::MAX))], 0).unwrap(),
            Value::output_list(Vec::new())
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
            assert_eq!(output.shape(), &[1, 2]);
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
                assert!(runmat_accelerate_api::same_provider_instance(
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
                for output in outputs
                    .try_into_owned()
                    .expect("unique unbounded fixture outputs")
                {
                    let Value::GpuTensor(output) = output else {
                        panic!("resident output");
                    };
                    assert_eq!(output, handle);
                    assert_eq!(
                        runmat_accelerate_api::handle_integer_type(&output),
                        Some(integer_type)
                    );
                    assert!(runmat_accelerate_api::same_provider_instance(
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
