use futures::executor::block_on;
use runmat_builtins::{
    BuiltinIntegerAuditKind, IntValue, IntegerStorage, NumericStorage, Tensor, Value,
};

fn integer_tensor(storage: IntegerStorage, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape.to_vec()).expect("integer tensor"))
}

fn call(name: &str, args: &[Value]) -> Result<Value, runmat_runtime::RuntimeError> {
    block_on(runmat_runtime::call_builtin_async(name, args))
}

#[test]
fn rounding_row_reduction_and_scatter_packet_has_settled_metadata() {
    for (name, forms) in [
        ("roots", 1),
        ("rot90", 1),
        ("round", 2),
        ("rref", 2),
        ("runtests", 1),
        ("scatter3", 2),
        ("scatterhist", 3),
        ("scatterplot", 2),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(builtin.integer_capabilities.iter().all(|capability| {
            !capability.inputs.is_empty()
                && capability
                    .inputs
                    .iter()
                    .all(|input| !input.classes.is_empty())
        }));
    }
    for name in ["rowfilter", "saveas", "savefig", "saveobj"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert_eq!(
            builtin.integer_audit.expect("integer audit").kind,
            BuiltinIntegerAuditKind::NotApplicable,
            "{name}"
        );
    }
}

#[test]
fn matlab_mode_rejects_each_runmat_only_integer_form() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let cases = [
        (
            "roots",
            vec![integer_tensor(IntegerStorage::U8(vec![1, 0, 1]), &[1, 3])],
            "RunMat:compatibility:RootsIntegerCoefficientsExtension",
        ),
        (
            "rref",
            vec![integer_tensor(
                IntegerStorage::U8(vec![1, 0, 0, 1]),
                &[2, 2],
            )],
            "RunMat:compatibility:RrefIntegerMatrixExtension",
        ),
        (
            "round",
            vec![Value::Num(1.25), Value::Int(IntValue::U8(1))],
            "RunMat:compatibility:RoundTypedIntegerDigitsExtension",
        ),
        (
            "scatterhist",
            vec![
                integer_tensor(IntegerStorage::U8(vec![1, 2]), &[1, 2]),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            ],
            "RunMat:compatibility:ScatterhistIntegerDataExtension",
        ),
        (
            "scatterplot",
            vec![integer_tensor(IntegerStorage::U8(vec![1, 2]), &[1, 2])],
            "RunMat:compatibility:ScatterplotIntegerDataExtension",
        ),
    ];
    for (name, args, identifier) in cases {
        let error = call(name, &args).expect_err("strict mode must reject extension");
        assert_eq!(error.identifier(), Some(identifier), "{name}");
    }
    let error = call(
        "rot90",
        &[
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::String("clockwise".into()),
        ],
    )
    .expect_err("direction token is a RunMat extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Rot90DirectionTokenExtension")
    );
}

#[test]
fn checked_binary64_boundaries_reject_wide_integer_rounding() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let wide = 9_007_199_254_740_993_u64;
    for (name, args) in [
        (
            "roots",
            vec![integer_tensor(IntegerStorage::U64(vec![1, wide]), &[1, 2])],
        ),
        (
            "rref",
            vec![integer_tensor(IntegerStorage::U64(vec![1, wide]), &[1, 2])],
        ),
        (
            "scatterhist",
            vec![
                integer_tensor(IntegerStorage::U64(vec![1, wide]), &[1, 2]),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            ],
        ),
        (
            "scatterplot",
            vec![integer_tensor(IntegerStorage::U64(vec![1, wide]), &[1, 2])],
        ),
    ] {
        let error = call(name, &args).expect_err("inexact boundary must reject");
        assert!(error.message().contains("exactly representable"), "{name}");
    }
}

#[test]
fn documented_integer_round_and_rotation_preserve_wide_native_storage() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let wide = 9_007_199_254_740_993_u64;
    let input = integer_tensor(IntegerStorage::U64(vec![1, wide, 2, 3]), &[2, 2]);
    let rounded = call("round", std::slice::from_ref(&input)).expect("integer round");
    let Value::Tensor(rounded) = rounded else {
        panic!("expected integer tensor")
    };
    assert_eq!(
        rounded.into_numeric_storage().expect("numeric storage"),
        NumericStorage::U64(vec![1, wide, 2, 3])
    );

    let rotated = call("rot90", &[input, Value::Int(IntValue::I64(-1))]).expect("integer rot90");
    let Value::Tensor(rotated) = rotated else {
        panic!("expected rotated integer tensor")
    };
    assert_eq!(rotated.numeric_dtype(), runmat_builtins::NumericDType::U64);
    assert!(rotated
        .integer_storage()
        .is_some_and(|storage| storage.exact_values().contains(&IntValue::U64(wide))));
}

#[test]
fn scatter3_properties_reconstruct_authoritative_integer_coordinates() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let wide = 9_007_199_254_740_993_u64;
    let x = integer_tensor(IntegerStorage::U64(vec![wide, wide + 2]), &[1, 2]);
    let y = integer_tensor(IntegerStorage::I16(vec![-2, 3]), &[1, 2]);
    let z = integer_tensor(IntegerStorage::U8(vec![4, 5]), &[1, 2]);
    let handle = call("scatter3", &[x, y, z]).expect("scatter3 handle");
    let xdata = call("get", &[handle, Value::String("XData".into())]).expect("XData");
    let Value::Tensor(xdata) = xdata else {
        panic!("expected XData tensor")
    };
    assert_eq!(
        xdata.integer_storage(),
        Some(&IntegerStorage::U64(vec![wide, wide + 2]))
    );
}

#[test]
fn object_and_persistence_boundaries_do_not_treat_integers_as_handles() {
    for (name, args) in [
        ("rowfilter", vec![Value::Int(IntValue::U8(1))]),
        (
            "saveas",
            vec![Value::Int(IntValue::U8(1)), Value::String("x.fig".into())],
        ),
        (
            "savefig",
            vec![Value::Int(IntValue::U8(1)), Value::String("x.fig".into())],
        ),
        ("saveobj", vec![Value::Int(IntValue::U8(1))]),
    ] {
        call(name, &args).expect_err("typed integer is not an object or graphics handle");
    }
}
