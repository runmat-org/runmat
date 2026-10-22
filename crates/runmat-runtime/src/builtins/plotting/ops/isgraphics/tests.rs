use super::*;
use runmat_builtins::ISGRAPHICS_DESCRIPTOR;

#[test]
fn isgraphics_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = ISGRAPHICS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"tf = isgraphics(h)"));
    assert!(labels.contains(&"tf = isgraphics(h, type)"));
}

#[test]
fn isgraphics_vectorizes_numeric_handle_arrays() {
    let handles = runmat_value::Tensor::new(vec![f64::NAN, -1.0], vec![2, 1]).unwrap();
    let result = invoke(vec![Value::Tensor(handles)]).unwrap();
    match result {
        Value::LogicalArray(logical) => {
            assert_eq!(logical.shape, vec![2, 1]);
            assert_eq!(logical.data, vec![0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn isgraphics_numeric_tensor_masks_keep_low_rank_shapes() {
    for (data, shape) in [(vec![-1.0], Vec::new()), (vec![-1.0, -2.0], vec![2])] {
        let input = runmat_value::Tensor::new(data.clone(), shape.clone()).unwrap();
        let Value::LogicalArray(output) = invoke(vec![Value::Tensor(input)]).unwrap() else {
            panic!("numeric tensor input must return a logical array")
        };
        assert_eq!(output.shape, shape);
        assert_eq!(output.data, vec![0; data.len()]);
    }
}

#[test]
fn isgraphics_vectorizes_typed_handles_without_a_floating_mirror() {
    let handles = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::I16(vec![-1, -2]),
        vec![2, 1],
    )
    .unwrap();
    let result = invoke(vec![Value::Tensor(handles)]).unwrap();
    match result {
        Value::LogicalArray(logical) => assert_eq!(logical.data, vec![0, 0]),
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[test]
fn compatibility_mode_treats_integer_arrays_as_nonobjects() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let handles = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX]),
        vec![1, 1],
    )
    .unwrap();
    let Value::LogicalArray(result) = invoke(vec![Value::Tensor(handles)]).unwrap() else {
        panic!("expected logical array");
    };
    assert_eq!(result.data, vec![0]);
}

#[test]
fn isgraphics_validates_arity_and_type_selector() {
    assert!(invoke(Vec::new()).is_err());
    assert!(invoke(vec![Value::Num(0.0), Value::from("root"), Value::Num(1.0)]).is_err());
    assert_eq!(
        invoke(vec![Value::Num(0.0), Value::from("root")]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        invoke(vec![Value::Num(0.0), Value::from("figure")]).unwrap(),
        Value::Bool(false)
    );
    let handles = runmat_value::Tensor::new(vec![0.0, f64::NAN], vec![2, 1]).unwrap();
    let Value::LogicalArray(result) =
        invoke(vec![Value::Tensor(handles), Value::from("root")]).unwrap()
    else {
        panic!("expected logical array");
    };
    assert_eq!(result.data, vec![1, 0]);
}
fn invoke(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    isgraphics_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

type Predicate = fn(
    Option<crate::context::RuntimeContext>,
    crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value>;
fn predicates() -> [Predicate; 2] {
    [
        isgraphics_builtin,
        crate::builtins::plotting::ishandle::ishandle_builtin,
    ]
}

#[test]
fn family_full_empty_shapes_native_values_and_resident_metadata_do_not_gather_or_retire_inputs() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };
    for predicate in predicates() {
        for shape in [vec![0, 5], vec![2, 0, 3], vec![0], vec![]] {
            let count: usize = shape.iter().product();
            let input = runmat_value::Tensor::from_f32(vec![-1.; count], shape.clone()).unwrap();
            let Value::LogicalArray(output) = predicate(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                    input,
                )]),
            )
            .unwrap() else {
                panic!("numeric tensor produces a mask");
            };
            assert_eq!(output.shape, shape);
            assert_eq!(output.data, vec![0; count]);
        }
        for element in [
            NumericElementType::F32,
            NumericElementType::F64,
            NumericElementType::U64,
        ] {
            let owner = TestAllocationOwner::new();
            let input = owner
                .allocate_zeroed(vec![2, 0, 3], element, GpuTensorStorage::Real)
                .unwrap();
            let values = [Value::GpuTensor(input.handle.clone())];
            let args = crate::call::BuiltinArguments::from_borrowed(&values, None).unwrap();
            let Value::LogicalArray(output) = predicate(None, args).unwrap() else {
                panic!("empty resident mask");
            };
            assert_eq!(output.shape, vec![2, 0, 3]);
            assert!(output.data.is_empty());
            assert_eq!(input.probe.retirements(), 0);
            input.handle.allocation_lease().validate_live().unwrap();
        }
        for value in [
            Value::Bool(false),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
            Value::Complex(0., 0.),
        ] {
            assert_eq!(
                predicate(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![value])
                )
                .unwrap(),
                Value::Bool(false)
            );
        }
    }
}

#[test]
fn exact_graphics_receiver_objects_remain_rooted_and_retire_without_numeric_retargeting() {
    use crate::builtins::plotting::state::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let figure = FigureHandle::from(37);
    select_figure(figure);
    let receiver = figure_graphics_receiver(figure).unwrap().unwrap();
    let value = graphics_receiver_object(receiver).unwrap().unwrap();
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    for predicate in predicates() {
        let values = [value.clone()];
        let args = crate::call::BuiltinArguments::from_borrowed(&values, Some(&runtime)).unwrap();
        assert_eq!(
            predicate(Some(runtime.clone()), args).unwrap(),
            Value::Bool(true)
        );
    }
    assert_eq!(
        invoke(vec![value.clone(), Value::from(" FIGURE ")]).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        invoke(vec![value.clone(), Value::from("line")]).unwrap(),
        Value::Bool(false)
    );
    let mut forged = value.clone();
    if let Value::HandleObject(handle) = &mut forged {
        handle.class_name = runmat_types::standard::GRAPHICS_LINE.into();
    }
    assert_eq!(invoke(vec![forged]).unwrap(), Value::Bool(false));
    let child = register_line_handle(figure, 0, 0);
    let child_value =
        graphics_receiver_object(plot_child_graphics_receiver(child as u64).unwrap().unwrap())
            .unwrap()
            .unwrap();
    assert_eq!(
        invoke(vec![child_value.clone(), Value::from("line")]).unwrap(),
        Value::Bool(true)
    );
    close_figure(Some(figure)).unwrap();
    select_figure(figure);
    for predicate in predicates() {
        for stale in [value.clone(), child_value.clone()] {
            assert_eq!(
                predicate(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![stale])
                )
                .unwrap(),
                Value::Bool(false)
            );
        }
    }
    assert_eq!(invoke(vec![Value::Num(37.)]).unwrap(), Value::Bool(true));
    reset_plot_state();
}

#[test]
fn family_output_preflight_and_selected_cancellation_are_terminal() {
    for predicate in predicates() {
        {
            let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
            let error = predicate(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            )
            .unwrap_err();
            assert!(error.identifier().unwrap().ends_with(":TooManyOutputs"));
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        }
        let source = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_cancellation_token(source.token());
        let values = [Value::Num(0.)];
        let args = crate::call::BuiltinArguments::from_borrowed(&values, Some(&runtime)).unwrap();
        source.cancel(runmat_execution::CancellationReason::User);
        let error = predicate(Some(runtime), args).unwrap_err();
        assert!(error.identifier().unwrap().contains("Cancelled"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
}

#[test]
fn family_managed_masks_refuse_unowned_work_without_spending_ledger() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
        RuntimeParallelService, RuntimeServicePorts,
    };
    struct Finite(ManagedHostEffectLedger);
    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let state = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(state.free_bytes()),
                epoch: state.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(4096);
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(std::rc::Rc::new(Finite(ledger.clone()))),
    );
    for predicate in predicates() {
        // Unbounded test ingress isolates the operation's own admission.
        let args = crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.)]);
        let error = predicate(Some(runtime.clone()), args).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn selector_forms_unknown_names_and_invalid_values_follow_public_grammar() {
    for text in [
        Value::String(" ROOT ".into()),
        Value::StringArray(
            runmat_value::StringArray::new(vec!["ROOT".into()], vec![1, 1]).unwrap(),
        ),
        Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars("root".chars().collect(), 1, 4).unwrap(),
        ),
    ] {
        assert_eq!(
            invoke(vec![Value::Num(0.), text]).unwrap(),
            Value::Bool(true)
        );
    }
    for text in ["", "unknown-type"] {
        assert_eq!(
            invoke(vec![Value::Num(0.), Value::from(text)]).unwrap(),
            Value::Bool(false)
        );
    }
    let error = invoke(vec![Value::Num(0.), Value::Num(1.)]).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:isgraphics:InvalidArgument")
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

#[test]
fn family_integer_scalar_aliases_cover_all_native_classes_without_rounded_aliases() {
    use runmat_value::IntValue;
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for predicate in predicates() {
        for zero in [
            IntValue::I8(0),
            IntValue::U8(0),
            IntValue::I16(0),
            IntValue::U16(0),
            IntValue::I32(0),
            IntValue::U32(0),
            IntValue::I64(0),
            IntValue::U64(0),
        ] {
            assert_eq!(
                predicate(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                        zero
                    )])
                )
                .unwrap(),
                Value::Bool(true)
            );
        }
        let unrepresentable = IntValue::U64(9_007_199_254_740_993);
        assert_eq!(
            predicate(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                    unrepresentable
                )])
            )
            .unwrap(),
            Value::Bool(false)
        );
    }
}
