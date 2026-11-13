use super::*;

#[test]
fn decomposition_nonfloating_input_follows_compatibility_mode() {
    let matrix = || typed_int_tensor(IntegerStorage::I16(vec![2, 0, 0, 4]), vec![2, 2]);
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call_constructor(vec![matrix()])
            .expect_err("MATLAB mode rejects integer decomposition coefficients");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DecompositionNonfloatingInputExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(matches!(
            call_constructor(vec![matrix()]).expect("RunMat mode accepts integer coefficients"),
            Value::Object(_)
        ));
    }

    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 2],
            runmat_accelerate_api::NumericElementType::I16,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap()
        .handle;
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call_constructor(vec![Value::GpuTensor(handle.clone())])
            .expect_err("MATLAB mode rejects resident integer coefficients before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DecompositionGpuInputExtension")
        );
    }
}

#[test]
fn integer_check_condition_has_an_independent_strict_mode_gate() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let args = || {
        vec![
            a.clone(),
            Value::String("CheckCondition".to_string().into()),
            Value::Int(IntValue::U8(1)),
        ]
    };
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call_constructor(args()).expect_err("strict mode must reject extension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DecompositionIntegerCheckConditionExtension")
        );
    }
    {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(call_constructor(args()).is_ok());
    }
}

#[test]
fn nonfloating_object_methods_recheck_the_compatibility_gate() {
    let d = {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        call_constructor(vec![typed_int_tensor(
            IntegerStorage::I32(vec![2, 0, 0, 4]),
            vec![2, 2],
        )])
        .expect("integer decomposition")
    };
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let errors = vec![
        block_on(decomposition_subsref(
            d.clone(),
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(TYPE_FIELD.to_string().into()),
        ))
        .expect_err("property access must recheck gate"),
        block_on(decomposition_uplus(d.clone())).expect_err("uplus must recheck gate"),
        block_on(decomposition_uminus(d.clone())).expect_err("uminus must recheck gate"),
        block_on(decomposition_times(d.clone(), Value::Num(2.0)))
            .expect_err("times must recheck gate"),
        block_on(decomposition_mtimes(Value::Num(2.0), d.clone()))
            .expect_err("mtimes must recheck gate"),
        call_rdivide(d.clone(), Value::Num(2.0)).expect_err("rdivide must recheck gate"),
        call_ctranspose(d.clone()).expect_err("ctranspose must recheck gate"),
        call_mldivide(d.clone(), tensor(&[8.0, 12.0], 2, 1))
            .expect_err("mldivide must recheck gate"),
        call_mrdivide(tensor(&[8.0, 12.0], 1, 2), d).expect_err("mrdivide must recheck gate"),
    ];
    for error in errors {
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DecompositionNonfloatingInputExtension")
        );
    }
}

#[test]
fn every_nonfloating_object_entry_point_binds_canonical_extension_metadata() {
    for name in [
        "decomposition.subsref",
        "decomposition.uplus",
        "decomposition.uminus",
        "decomposition.ctranspose",
        "decomposition.times",
        "decomposition.mtimes",
        "decomposition.rdivide",
        "decomposition.mrdivide",
        "decomposition.mldivide",
    ] {
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("missing canonical catalog entry for {name}"));
        assert!(
            runmat_builtins::builtin_function_by_name(name).is_none(),
            "legacy metadata authority retained for {name}"
        );
        assert_eq!(builtin.bindings.len(), 1, "canonical bindings for {name}");
        assert_eq!(builtin.bindings[0].variant, "default");
        let bindings = crate::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(bindings.len(), 1, "runtime implementations for {name}");
        assert_eq!(
            bindings[0].identity,
            builtin.binding_identity(&builtin.bindings[0]),
            "runtime binding must join canonical metadata for {name}"
        );
        assert!(
            builtin
                .extensions
                .iter()
                .any(|extension| extension.id == "decomposition-nonfloating-input"),
            "{name} extension"
        );
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} integer capability disposition"
        );
    }
}

#[test]
fn gpu_object_operands_are_gated_before_provider_access() {
    let d = call_constructor(vec![tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2)])
        .expect("floating decomposition");
    let gpu_scalar = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for error in [
        block_on(decomposition_times(d.clone(), gpu_scalar.clone())).expect_err("times gpu gate"),
        block_on(decomposition_mtimes(gpu_scalar.clone(), d.clone())).expect_err("mtimes gpu gate"),
        call_rdivide(d.clone(), gpu_scalar.clone()).expect_err("rdivide gpu gate"),
        call_mldivide(d.clone(), gpu_scalar.clone()).expect_err("mldivide gpu gate"),
        call_mrdivide(gpu_scalar.clone(), d).expect_err("mrdivide gpu gate"),
    ] {
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DecompositionGpuInputExtension")
        );
    }
}

#[test]
fn scale_and_rank_integer_controls_gate_and_reject_inexact_binary64_values() {
    let d = call_constructor(vec![tensor(&[2.0], 1, 1)]).expect("decomposition");
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(decomposition_times(d.clone(), Value::Int(IntValue::U64(2))))
            .expect_err("integer scale gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DecompositionIntegerScaleFactorExtension")
        );
        let complex_factor = mirrorless_complex_int_tensor(
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I16(vec![1]),
            vec![1, 1],
        );
        let error = block_on(decomposition_times(d.clone(), complex_factor))
            .expect_err("complex integer scale gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DecompositionComplexIntegerScaleFactorExtension")
        );
        let error = call_constructor(vec![
            tensor(&[2.0], 1, 1),
            Value::String("RankTolerance".to_string().into()),
            Value::Int(IntValue::I64(1)),
        ])
        .expect_err("integer rank gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DecompositionIntegerRankToleranceExtension")
        );
    }

    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    for factor in [
        Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        Value::Int(IntValue::I64(-((1_i64 << 53) + 1))),
        mirrorless_complex_int_tensor(
            IntegerStorage::U64(vec![1]),
            IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        ),
    ] {
        let error = block_on(decomposition_times(d.clone(), factor))
            .expect_err("inexact integer scale boundary");
        assert!(error.message().contains("exactly representable as double"));
    }
    let rank_error = call_constructor(vec![
        tensor(&[2.0], 1, 1),
        Value::String("RankTolerance".to_string().into()),
        Value::Int(IntValue::U64((1_u64 << 53) + 1)),
    ])
    .expect_err("inexact rank boundary");
    assert!(rank_error
        .message()
        .contains("exactly representable as double"));

    block_on(decomposition_times(
        d,
        Value::Int(IntValue::U64(1_u64 << 53)),
    ))
    .expect("exact endpoint scale");
}
