use super::*;

#[test]
fn failures_use_the_declared_boundary_identifiers() {
    let size = futures::executor::block_on(int8_empty_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.5)]),
    ))
    .unwrap_err();
    assert_eq!(size.identifier(), Some("RunMat:empty:InvalidSize"));
    let unsupported =
        futures::executor::block_on(build_empty(EmptyClass::Struct, vec![0, 0])).unwrap_err();
    assert_eq!(
        unsupported.identifier(),
        Some("RunMat:empty:UnsupportedClass")
    );
    // Invalid internal assembly input exercises the storage-construction boundary.
    let assembly =
        futures::executor::block_on(build_empty(EmptyClass::Double, vec![2, 2])).unwrap_err();
    assert_eq!(assembly.identifier(), Some("RunMat:empty:Internal"));
    assert_eq!(
        parse_numeric_dimension(usize::MAX as f64).is_ok(),
        usize::BITS < 64
    );
}

#[test]
fn explicit_gpu_empty_keeps_shape_and_device_intent() {
    crate::builtins::common::test_support::with_test_provider(|_| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = futures::executor::block_on(empty_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(3.0),
                Value::from("gpuArray"),
            ]),
        ))
        .unwrap();
        let Value::GpuTensor(handle) = result else {
            panic!("expected device empty");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&handle));
        assert_eq!(handle.shape(), vec![0, 3]);
        let host = crate::builtins::common::test_support::gather(Value::GpuTensor(handle)).unwrap();
        assert!(host.is_empty());
        assert_eq!(host.shape(), vec![0, 3]);
    });
}

#[test]
fn resident_integer_size_controls_preserve_host_receiver_class() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Tensor::new_integer(IntegerStorage::U64(vec![0, 7]), vec![1, 2]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input).unwrap();
        let result = futures::executor::block_on(int16_empty_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle,
            )]),
        ))
        .unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected host empty");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::I16);
        assert_eq!(result.shape(), vec![0, 7]);
    });
}

#[test]
fn requested_outputs_reject_before_size_or_provider_access() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle,
    );
    let error = futures::executor::block_on(uint64_empty_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![resident]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:empty:TooManyOutputs"));
}

#[test]
fn discarded_output_still_validates_empty_shape() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert!(futures::executor::block_on(int8_empty_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![])
    ))
    .is_ok());
    assert!(futures::executor::block_on(int8_empty_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)])
    ))
    .is_err());
}

#[test]
fn global_compatibility_rejects_before_resident_controls() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle,
    );
    let error = futures::executor::block_on(empty_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![resident]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:EmptyGlobalCallExtension")
    );
}

fn run_static(class: EmptyClass, args: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(async {
        let args = crate::builtins::common::test_support::builtin_arguments(args);
        let shape = parse_shape(&args).await?;
        build_empty(class, shape).await
    })
}

#[test]
fn all_integer_receivers_preserve_exact_empty_storage() {
    for (class, dtype) in [
        (EmptyClass::Int8, NumericDType::I8),
        (EmptyClass::Int16, NumericDType::I16),
        (EmptyClass::Int32, NumericDType::I32),
        (EmptyClass::Int64, NumericDType::I64),
        (EmptyClass::Uint8, NumericDType::U8),
        (EmptyClass::Uint16, NumericDType::U16),
        (EmptyClass::Uint32, NumericDType::U32),
        (EmptyClass::Uint64, NumericDType::U64),
    ] {
        let Value::Tensor(value) = run_static(
            class,
            vec![Value::Int(IntValue::U64(0)), Value::Int(IntValue::U64(7))],
        )
        .unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(value.shape(), vec![0, 7]);
        assert_eq!(value.numeric_dtype(), dtype);
        assert!(value.integer_storage().is_some());
        assert!(value.is_empty());
    }
}

#[test]
fn signed_negative_dimensions_clamp_and_trailing_ones_trim() {
    let Value::Tensor(value) = run_static(
        EmptyClass::Int64,
        vec![
            Value::Int(IntValue::I64(-9)),
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U8(1)),
        ],
    )
    .unwrap() else {
        panic!("expected tensor");
    };
    assert_eq!(value.shape(), vec![0, usize::MAX]);
    assert_eq!(value.numeric_dtype(), NumericDType::I64);
}

#[test]
fn row_size_vectors_accept_exact_integer_storage_and_columns_reject() {
    let row = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX, 1]), vec![1, 3]).unwrap();
    let Value::Tensor(value) = run_static(EmptyClass::Uint64, vec![Value::Tensor(row)]).unwrap()
    else {
        panic!("expected tensor");
    };
    assert_eq!(value.shape(), vec![0, usize::MAX]);

    let column = Tensor::new_integer(IntegerStorage::U8(vec![0, 3]), vec![2, 1]).unwrap();
    let err = run_static(EmptyClass::Uint8, vec![Value::Tensor(column)]).unwrap_err();
    assert!(err.to_string().contains("row vector"));
}

#[test]
fn nonempty_shapes_reject_without_multiplication() {
    let err = run_static(
        EmptyClass::Uint64,
        vec![
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Int(IntValue::U64(u64::MAX)),
        ],
    )
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:empty:NonEmptyShape"));
}

#[test]
fn global_shorthand_is_gated_in_matlab_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = futures::executor::block_on(empty_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(IntValue::U8(0)),
            Value::from("uint8"),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:EmptyGlobalCallExtension")
    );
}

#[test]
fn resident_size_control_rejects_before_provider_access_in_matlab_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle,
    );
    let arguments = crate::builtins::common::test_support::builtin_arguments(vec![resident]);
    let err = futures::executor::block_on(parse_shape(&arguments)).unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:EmptyResidentSizeExtension")
    );
}

#[test]
fn integer_static_names_are_registered() {
    for name in [
        "int8.empty",
        "int16.empty",
        "int32.empty",
        "int64.empty",
        "uint8.empty",
        "uint16.empty",
        "uint32.empty",
        "uint64.empty",
    ] {
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(builtin.integer_capabilities.len(), 1);
        assert!(
            runmat_builtins::builtin_function_by_name(name).is_none(),
            "legacy authority retained for {name}"
        );
        assert_eq!(builtin.bindings.len(), 1);
        assert_eq!(builtin.bindings[0].variant, "default");
        let bindings = crate::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(bindings.len(), 1, "expected one implementation for {name}");
        assert_eq!(
            bindings[0].identity,
            builtin.binding_identity(&builtin.bindings[0])
        );
        let result = futures::executor::block_on(crate::call_builtin_async(
            name,
            &[Value::Num(0.0), Value::Num(7.0)],
        ))
        .unwrap_or_else(|error| panic!("joined dispatch of {name} failed: {error}"));
        let Value::Tensor(result) = result.value() else {
            panic!("expected typed empty for {name}");
        };
        assert_eq!(result.shape(), vec![0, 7]);
        assert_eq!(
            result.numeric_dtype().class_name(),
            name.strip_suffix(".empty").unwrap()
        );
    }
}

#[test]
fn dotted_integer_static_dispatch_preserves_wide_shape_and_class() {
    let value = futures::executor::block_on(crate::call_builtin_async(
        "uint64.empty",
        &[
            Value::Int(IntValue::I64(-1)),
            Value::Int(IntValue::U64(u64::MAX)),
        ],
    ))
    .unwrap();
    let Value::Tensor(value) = value.value() else {
        panic!("expected uint64 tensor");
    };
    assert_eq!(value.shape(), vec![0, usize::MAX]);
    assert_eq!(value.numeric_dtype(), NumericDType::U64);
    assert!(value.integer_storage().is_some());
}
