#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

use runmat_builtins::{IntValue, IntegerStorage, NumericDType, Value};

fn logical_truth(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Num(value) => *value != 0.0,
        other => panic!("expected logical value, got {other:?}"),
    }
}

fn sparse_scalar(value: &Value) -> f64 {
    match value {
        Value::SparseTensor(sparse) if sparse.shape() == vec![1, 1] => {
            sparse.get(0, 0).unwrap_or(0.0)
        }
        other => panic!("expected sparse scalar value, got {other:?}"),
    }
}

#[test]
fn logical_operators_and_short_circuit() {
    let vars =
        execute_source("a = 0 && (1/0); b = 1 || (1/0); c = 0 & 5; d = 0 | 5; e = ~0; f = ~5;")
            .unwrap();
    assert!(!logical_truth(&vars[0]));
    assert!(logical_truth(&vars[1]));
    assert!(!logical_truth(&vars[2]));
    assert!(logical_truth(&vars[3]));
    assert!(logical_truth(&vars[4]));
    assert!(!logical_truth(&vars[5]));
}

#[test]
fn elementwise_and_accepts_all_integer_classes_through_vm_dispatch() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a = {constructor}([0 1]); b = {constructor}([2; 0]); via_operator = a & b; via_function = and(a, b);"
        );
        let vars = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor}: compiled and failed: {error}"));
        for index in [2, 3] {
            assert!(
                matches!(
                    &vars[index],
                    Value::LogicalArray(array)
                        if array.shape == vec![2, 2] && array.data == vec![0, 0, 1, 0]
                ),
                "{constructor}: unexpected result at {index}: {:?}",
                vars[index]
            );
        }
    }

    let vars = execute_source(
        "wide = uint64([0 9007199254740993 18446744073709551615]); mask = wide & int8([1 1 0]);",
    )
    .expect("wide integer and");
    assert!(matches!(
        &vars[1],
        Value::LogicalArray(array) if array.data == vec![0, 1, 0]
    ));
}

#[test]
fn angle_rejects_all_real_and_componentwise_complex_integer_classes_through_vm_dispatch() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        for source in [
            format!("out = angle({constructor}(1));"),
            format!("out = angle({constructor}([1 2]));"),
            format!("z = complex({constructor}([1 2]), {constructor}([3 4])); out = angle(z);"),
        ] {
            let error = execute_source(&source).expect_err("angle must reject integer input");
            assert_eq!(
                error.identifier(),
                Some("RunMat:angle:InvalidInput"),
                "{constructor}: {source}"
            );
        }
    }
}

#[test]
fn append_rejects_all_integer_classes_through_vm_dispatch() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        for source in [
            format!("out = append({constructor}(1), 'suffix');"),
            format!("out = append({constructor}([1 2]), 'suffix');"),
        ] {
            let error = execute_source(&source).expect_err("append must reject integer input");
            assert!(
                error.to_string().contains(
                    "expected string, character vector, or cell array of character vectors"
                ),
                "{constructor}: {source}: {error}"
            );
        }
    }
}

#[test]
fn append_preserves_text_output_precedence_through_vm_dispatch() {
    let vars = execute_source(
        "char_out = append('Hello ', 'World'); cell_out = append({'alpha','beta'}, ' '); string_out = append([\"A\";\"B\"], {'x','y'});",
    )
    .expect("append text outputs");

    assert!(matches!(
        &vars[0],
        Value::CharArray(array)
            if array.rows == 1
                && array.cols == 11
                && array.data.iter().collect::<String>() == "Hello World"
    ));
    assert!(matches!(
        &vars[1],
        Value::Cell(cell)
            if cell.shape == vec![1, 2]
                && matches!(&cell.data[0], Value::CharArray(array) if array.data.iter().collect::<String>() == "alpha ")
                && matches!(&cell.data[1], Value::CharArray(array) if array.data.iter().collect::<String>() == "beta ")
    ));
    assert!(matches!(
        &vars[2],
        Value::StringArray(array)
            if array.shape == vec![2, 2]
                && array.data == vec!["Ax", "Bx", "Ay", "By"]
    ));
}

#[test]
fn short_circuit_or_accepts_boolean_lhs_without_numeric_coercion() {
    let vars = execute_source(
        "tau = []; flight_duration = 10; guard = isempty(tau) || tau(end) < flight_duration;",
    )
    .unwrap();
    assert!(logical_truth(&vars[2]));
}

#[test]
fn integer_scalar_arithmetic_keeps_int64_and_uint64_exact_through_vm_dispatch() {
    let vars = execute_source(
        "u = uint64(9223372036854775808); up = u + 1; down = uint64(18446744073709551615) - 1; lo = int64(-9223372036854775808) + 1; reverse = 1 - int64(-9223372036854775808); same = up .* 1; row = uint64([9223372036854775808 18446744073709551615]); rowdown = row - 1; col = uint64([9223372036854775808; 18446744073709551615]); row2 = uint64([1 2 0]); plusgrid = col + row2; timesgrid = col .* row2; signedcol = int64([-9223372036854775808; 9223372036854775807]); signedrow = int64([1 -7 -9223372036854775808]); minusgrid = signedcol - signedrow; remgrid = rem(int64([-7; 7]), int64([4 -4 0])); modgrid = mod(int64([-7; 7]), int64([4 -4 0]));",
    )
    .expect("integer arithmetic should execute");

    assert_eq!(vars[0], Value::Int(IntValue::U64(1_u64 << 63)));
    assert_eq!(vars[1], Value::Int(IntValue::U64((1_u64 << 63) + 1)));
    assert_eq!(vars[2], Value::Int(IntValue::U64(u64::MAX - 1)));
    assert_eq!(vars[3], Value::Int(IntValue::I64(i64::MIN + 1)));
    assert_eq!(vars[4], Value::Int(IntValue::I64(i64::MAX)));
    assert_eq!(vars[5], Value::Int(IntValue::U64((1_u64 << 63) + 1)));
    assert!(matches!(
        &vars[7],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![(1_u64 << 63) - 1, u64::MAX - 1]))
    ));
    assert!(matches!(
        &vars[10],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![
                        (1_u64 << 63) + 1,
                        u64::MAX,
                        (1_u64 << 63) + 2,
                        u64::MAX,
                        1_u64 << 63,
                        u64::MAX
                    ]))
    ));
    assert!(matches!(
        &vars[11],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![
                        1_u64 << 63,
                        u64::MAX,
                        u64::MAX,
                        u64::MAX,
                        0,
                        0
                    ]))
    ));
    assert!(matches!(
        &vars[14],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::I64(vec![
                        i64::MIN,
                        i64::MAX - 1,
                        i64::MIN + 7,
                        i64::MAX,
                        0,
                        i64::MAX
                    ]))
    ));
    assert!(matches!(
        &vars[15],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::I64(vec![-3, 3, -3, 3, 0, 0]))
    ));
    assert!(matches!(
        &vars[16],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::I64(vec![1, 3, -3, -1, -7, 7]))
    ));
}

#[test]
fn every_integer_class_uses_native_arithmetic_and_comparison_dispatch() {
    let classes = [
        ("int8", IntValue::I8(5)),
        ("int16", IntValue::I16(5)),
        ("int32", IntValue::I32(5)),
        ("int64", IntValue::I64(5)),
        ("uint8", IntValue::U8(5)),
        ("uint16", IntValue::U16(5)),
        ("uint32", IntValue::U32(5)),
        ("uint64", IntValue::U64(5)),
    ];
    let source = classes
        .iter()
        .enumerate()
        .map(|(index, (class, _))| {
            format!("a{index} = {class}(2) + {class}(3); c{index} = {class}(5) > {class}(2);")
        })
        .collect::<String>();
    let vars = execute_source(&source).expect("integer VM dispatch should execute");

    for (index, (_, expected)) in classes.iter().enumerate() {
        assert_eq!(vars[index * 2], Value::Int(expected.clone()));
        assert!(logical_truth(&vars[index * 2 + 1]));
    }
}

#[test]
fn complex_integer_values_preserve_exact_components_through_vm_dispatch() {
    let vars = execute_source(
        "r = uint64([9223372036854775808 18446744073709551615]); z = complex(r, 1); zr = real(z); zi = imag(z); scalar = complex(int64(-9223372036854775808), int64(7)); sr = real(scalar); si = imag(scalar); tf = isreal(z); high = uint64(9223372036854775808) + 1; highz = complex(high, 1); highr = real(highz); picked = z([2 1]); pickedreal = real(picked); reshaped = reshape(z, 2, 1); reshapedreal = real(reshaped); scalarreshape = reshape(high, 1, 1, 1);",
    )
    .expect("integer complex construction should execute");

    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]),
                        &IntegerStorage::U64(vec![1, 1]),
                    ))
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![1, 1]))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I64(vec![i64::MIN]),
                    &IntegerStorage::I64(vec![7]),
                ))
    ));
    assert_eq!(vars[5], Value::Int(IntValue::I64(i64::MIN)));
    assert_eq!(vars[6], Value::Int(IntValue::I64(7)));
    assert!(!logical_truth(&vars[7]));
    assert_eq!(
        vars[10],
        Value::Int(IntValue::U64(9_223_372_036_854_775_809))
    );
    assert!(matches!(
        &vars[11],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]),
                    &IntegerStorage::U64(vec![1, 1]),
                ))
    ));
    assert!(matches!(
        &vars[12],
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(
        &vars[13],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![2, 1]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]),
                        &IntegerStorage::U64(vec![1, 1]),
                    ))
    ));
    assert!(matches!(
        &vars[14],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 1]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]))
    ));
    assert!(matches!(
        &vars[15],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 1, 1]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_809]))
    ));
}

#[test]
fn typed_complex_integer_arithmetic_is_rejected_before_f64_coercion() {
    let operators = [
        ("plus", "z + z"),
        ("minus", "z - z"),
        ("times", "z .* z"),
        ("rdivide", "z ./ z"),
        ("ldivide", "z .\\ z"),
        ("power", "z .^ z"),
        ("mtimes", "z * z"),
        ("mrdivide", "z / z"),
        ("mldivide", "z \\ z"),
        ("mpower", "z ^ z"),
        ("uminus", "-z"),
        ("uplus", "+z"),
        ("plus like", "plus(z, z, 'like', z)"),
    ];

    for (name, operation) in operators {
        let source =
            format!("z = complex(uint64(9223372036854775808), uint64(1)); out = {operation};");
        let err = execute_source(&source).expect_err(name);
        assert!(
            err.to_string()
                .contains("complex integer arithmetic is not supported"),
            "{name} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_kron_is_rejected_before_f64_coercion() {
    for operation in ["kron(z, z)", "kron(z, 1)", "kron(1, z)"] {
        let source =
            format!("z = complex(uint64(9223372036854775808), uint64(1)); out = {operation};");
        let err = execute_source(&source).expect_err("kron must reject typed complex integers");
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "unexpected kron error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_trace_is_rejected_before_f64_coercion() {
    let err = execute_source(
        "z = complex(uint64([9223372036854775808 2; 3 4]), uint64([1 2; 3 4])); out = trace(z);",
    )
    .expect_err("trace must reject typed complex integers");
    assert!(
        err.to_string()
            .contains("operations involving complex numbers with integer types are not supported"),
        "unexpected trace error: {err}"
    );
}

#[test]
fn typed_complex_integer_dot_is_rejected_before_f64_coercion() {
    for operation in ["dot(z, z)", "dot(z, [1 2])", "dot([1 2], z)"] {
        let source = format!(
            "z = complex(uint64([9223372036854775808 2]), uint64([1 2])); out = {operation};"
        );
        let err = execute_source(&source).expect_err("dot must reject typed complex integers");
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "unexpected dot error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_cross_uses_runmat_double_extension() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("a = complex(int16([1 0 0]), int16([1 0 0])); b = complex(uint16([0 1 0]), uint16([0 1 0])); c = cross(a,b); if ~strcmp(class(c),'double') || ~isequal(size(c),[1 3]) || isreal(c) || real(c(3)) ~= 0 || imag(c(3)) ~= 2; error('typed complex cross mismatch'); end;").expect("typed complex cross");
}

#[test]
fn typed_complex_integer_pagemtimes_is_rejected_before_f64_coercion() {
    for operation in ["pagemtimes(z, z)", "pagemtimes(z, 1)", "pagemtimes(1, z)"] {
        let source = format!(
            "z = complex(uint64([9223372036854775808 2; 3 4]), uint64([1 2; 3 4])); out = {operation};"
        );
        let err =
            execute_source(&source).expect_err("pagemtimes must reject typed complex integers");
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "unexpected pagemtimes error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_norm_is_rejected_before_f64_coercion() {
    for operation in ["norm(z)", "norm(z, 1)", "norm(z, 'fro')"] {
        let source = format!(
            "z = complex(uint64([9223372036854775808 2]), uint64([1 2])); out = {operation};"
        );
        let err = execute_source(&source).expect_err("norm must reject typed complex integers");
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "unexpected norm error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_det_is_rejected_before_f64_coercion() {
    let err = execute_source(
        "z = complex(uint64([9223372036854775808 2; 3 4]), uint64([1 2; 3 4])); out = det(z);",
    )
    .expect_err("det must reject typed complex integers");
    assert!(
        err.to_string()
            .contains("operations involving complex numbers with integer types are not supported"),
        "unexpected det error: {err}"
    );
}

#[test]
fn typed_complex_integer_factorization_and_solve_operations_are_rejected_before_f64_coercion() {
    let operations = [
        ("chol", "chol(z)"),
        ("lu", "lu(z)"),
        ("qr", "qr(z)"),
        ("svd", "svd(z)"),
        ("eig", "eig(z)"),
        ("generalized eig", "eig(z, z)"),
        ("inv", "inv(z)"),
        ("cond", "cond(z)"),
        ("rcond", "rcond(z)"),
        ("pinv", "pinv(z)"),
        ("rank", "rank(z)"),
        ("rref", "rref(z)"),
        ("null", "null(z)"),
        ("linsolve matrix", "linsolve(z, [1; 2])"),
        ("linsolve rhs", "linsolve(eye(2), z)"),
    ];

    for (name, operation) in operations {
        let source = format!(
            "z = complex(uint64([9223372036854775808 2; 3 4]), uint64([1 2; 3 4])); out = {operation};"
        );
        let err = execute_source(&source).expect_err(name);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "{name} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_polynomial_and_convolution_operations_are_rejected_before_f64_coercion() {
    let operations = [
        ("roots", "roots(z(1, :))"),
        ("polyder coefficients", "polyder(z(1, :))"),
        ("polyder product", "polyder([1 2], z(1, :))"),
        ("polyint coefficients", "polyint(z(1, :))"),
        ("polyint constant", "polyint([1 2], z)"),
        (
            "polyint scalar constant",
            "polyint([1 2], complex(uint64(9223372036854775808), uint64(1)))",
        ),
        ("polyint indexed scalar constant", "polyint([1 2], z(1, 1))"),
        ("polyval coefficients", "polyval(z(1, :), [1 2])"),
        ("polyval points", "polyval([1 2], z(1, :))"),
        ("deconv numerator", "deconv(z(1, :), [1 2])"),
        ("deconv denominator", "deconv([1 2], z(1, :))"),
    ];

    for (name, operation) in operations {
        let source = format!(
            "z = complex(uint64([9223372036854775808 2; 3 4]), uint64([1 2; 3 4])); out = {operation};"
        );
        let err = execute_source(&source).expect_err(name);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ) || err.to_string().contains("input must be single or double"),
            "{name} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn typed_integer_conv_and_conv2_cross_to_documented_double_outputs() {
    execute_source("a = uint64([9007199254740993 2]); b = int16([1 2]); c = conv(a,b); if ~strcmp(class(c),'double') || ~isequal(size(c),[1 3]); error('integer conv mismatch'); end; m = uint32([1 2;3 4]); k = int8([1 0;0 -1]); d = conv2(m,k,'same'); if ~strcmp(class(d),'double') || ~isequal(size(d),[2 2]); error('integer conv2 mismatch'); end; z = complex(int16([1 2]),int16([1 -1])); q = conv(z,int8(2)); if ~strcmp(class(q),'double') || real(q(1)) ~= 2 || imag(q(1)) ~= 2; error('complex integer conv mismatch'); end;").expect("typed integer convolution script");
}

#[test]
fn typed_complex_integer_signal_operations_are_rejected_before_f64_coercion() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let operations = [
        ("filter numerator", "filter(z(1, :), [1], [1 2])"),
        ("filter denominator", "filter([1], z(1, :), [1 2])"),
        ("filter signal", "filter([1], [1], z(1, :))"),
        ("filter initial state", "filter([1], [1], [1 2], z(1, :))"),
        ("filtfilt numerator", "filtfilt(z(1, :), [1], [1 2 3 4])"),
        ("filtfilt denominator", "filtfilt([1], z(1, :), [1 2 3 4])"),
        ("filtfilt signal", "filtfilt([1], [1], z(1, :))"),
        ("freqz numerator", "freqz(z(1, :), [1])"),
        ("freqz denominator", "freqz([1], z(1, :))"),
        ("freqz count", "freqz([1], [1], z(1, 1))"),
        ("pwelch signal", "pwelch(z(1, :))"),
        ("periodogram signal", "periodogram(z(1, :))"),
        ("spectrogram signal", "spectrogram(z(1, :))"),
        ("sinc", "sinc(z(1, :))"),
        ("fir1 cutoff", "fir1(4, z(1, 1))"),
        ("buttord passband edge", "buttord(z(1, 1), 0.5, 3, 40)"),
        ("zplane coefficients", "zplane(z(1, :), [1])"),
        ("zplane SOS", "zplane(z)"),
    ];

    for (name, operation) in operations {
        let source = format!(
            "z = complex(uint64([9223372036854775808 2; 3 4]), uint64([1 2; 3 4])); out = {operation};"
        );
        let err = execute_source(&source).expect_err(name);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "{name} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_rounding_operations_are_rejected_before_f64_coercion() {
    let operations = [
        ("ceil", "ceil(z)"),
        ("floor", "floor(z)"),
        ("fix", "fix(z)"),
        ("round", "round(z)"),
        ("round decimal places", "round(z, 1)"),
        ("rem left", "rem(z, 2)"),
        ("rem right", "rem(2, z)"),
        ("mod left", "mod(z, 2)"),
        ("mod right", "mod(2, z)"),
    ];

    for (name, operation) in operations {
        let source = format!(
            "z = complex(uint64([9223372036854775808 18446744073709551615]), uint64([1 2])); out = {operation};"
        );
        let err = execute_source(&source).expect_err(name);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ) || err.to_string().contains("inputs must be real"),
            "{name} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn ceil_preserves_all_integer_classes_through_compiled_dispatch() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let values = if constructor.starts_with('u') {
            "[0 3 4]"
        } else {
            "[-3 0 4]"
        };
        let source = format!("x = {constructor}({values}); y = ceil(x);");
        let vars = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor}: compiled ceil failed: {error}"));
        assert_eq!(
            vars[1], vars[0],
            "{constructor} ceil must preserve class and bits"
        );
    }
}

#[test]
fn compiled_fix_floor_and_flip_family_preserve_exact_wide_integer_storage() {
    let base = 9_007_199_254_740_992_u64;
    let vars = execute_source(
        "base = uint64(9007199254740992); top = intmax('uint64'); a = reshape([base+uint64(1), base+uint64(2), top-uint64(1), top], 2, 2); fixed = fix(a); floored = floor(a); default_flip = flip(a); dim_flip = flip(a, 2); left_right = fliplr(a); up_down = flipud(a);",
    )
    .expect("compiled rounding and flip family");

    let expected = [
        IntegerStorage::U64(vec![base + 1, base + 2, u64::MAX - 1, u64::MAX]),
        IntegerStorage::U64(vec![base + 2, base + 1, u64::MAX, u64::MAX - 1]),
        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, base + 1, base + 2]),
    ];
    assert!(
        vars.iter()
            .filter(|value| matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&expected[0])))
            .count()
            >= 3,
        "source, fix, and floor must retain exact storage: {vars:?}"
    );
    assert!(
        vars.iter()
            .filter(|value| matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&expected[1])))
            .count()
            >= 2,
        "default flip and flipud must retain exact reordered storage: {vars:?}"
    );
    assert!(
        vars.iter()
            .filter(|value| matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(&expected[2])))
            .count()
            >= 2,
        "dimension flip and fliplr must retain exact reordered storage: {vars:?}"
    );
}

#[test]
fn compiled_flip_and_gamma_family_extensions_obey_compatibility_mode() {
    {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        let flip_error = execute_source("a = uint64([1 2]); out = flip(a, uint8(2));")
            .expect_err("typed flip dimension must be gated");
        assert_eq!(
            flip_error.identifier(),
            Some("RunMat:compatibility:FlipTypedDimensionExtension")
        );
        let gamma_error = execute_source("out = gamma(uint16(5));")
            .expect_err("gamma integer input remains unsupported");
        assert_eq!(gamma_error.identifier(), Some("RunMat:gamma:InvalidInput"));
        let gammaln_error = execute_source("out = gammaln(uint16(5));")
            .expect_err("gammaln integer extension must be gated");
        assert_eq!(
            gammaln_error.identifier(),
            Some("RunMat:compatibility:GammalnIntegerInputExtension")
        );
    }
    {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
        let vars = execute_source(
            "a = uint64([1 2]); out = flip(a, uint8(2)); values = gammaln(uint16([1 2 5]));",
        )
        .expect("RunMat flip and gammaln extensions");
        assert!(vars.iter().any(|value| matches!(
            value,
            Value::Tensor(tensor)
                if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![2, 1]))
        )));
        assert!(vars.iter().any(|value| matches!(
            value,
            Value::Tensor(tensor)
                if tensor.numeric_dtype() == NumericDType::F64
                    && tensor.shape == vec![1, 3]
                    && tensor.integer_storage().is_none()
                    && tensor.materialize_f64().iter().zip([0.0, 0.0, 24.0_f64.ln()]).all(|(actual, expected)| (actual - expected).abs() < 1.0e-10)
        )));
        let gamma_error = execute_source("out = gamma(uint16(5));")
            .expect_err("gamma must not be broadened by RunMat mode");
        assert_eq!(gamma_error.identifier(), Some("RunMat:gamma:InvalidInput"));
        let wide_error =
            execute_source("wide = uint64(9007199254740992) + uint64(1); out = gammaln(wide);")
                .expect_err("gammaln must reject inexact binary64 conversion");
        assert_eq!(wide_error.identifier(), Some("RunMat:gammaln:InvalidInput"));
    }
}

#[test]
fn cell_accepts_all_integer_size_classes_through_compiled_dispatch() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!("dims = {constructor}([2 3]); c = cell(dims);");
        let vars = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor}: compiled cell failed: {error}"));
        let Value::Cell(cell) = &vars[1] else {
            panic!("{constructor}: expected cell output, got {:?}", vars[1]);
        };
        assert_eq!(cell.shape, vec![2, 3], "{constructor}: shape");
        assert_eq!(cell.data.len(), 6, "{constructor}: element count");
        assert!(cell.data.iter().all(|value| matches!(
            value,
            Value::Tensor(tensor)
                if tensor.shape == vec![0, 0]
                    && tensor.numeric_dtype() == NumericDType::F64
                    && tensor.is_empty()
        )));
    }

    let vars = execute_source("dims = int64([-2 3 1 1]); c = cell(dims);")
        .expect("compiled cell must clamp signed negative sizes and trim trailing singletons");
    assert!(matches!(&vars[1], Value::Cell(cell) if cell.shape == vec![0, 3]));
}

#[test]
fn typed_complex_integer_gpuarray_is_rejected_before_provider_dispatch() {
    let err = execute_source(
        "z = complex(uint64([9223372036854775808 18446744073709551615]), uint64([1 2])); g = gpuArray(z);",
    )
    .expect_err("typed complex integer gpuArray input must be rejected");
    assert!(
        err.to_string()
            .contains("typed complex integer arrays are not supported"),
        "unexpected error: {err}"
    );
}

#[test]
fn typed_complex_integer_analytic_operations_are_rejected_before_f64_coercion() {
    for builtin in [
        "abs", "angle", "exp", "expm1", "gamma", "log", "log10", "log1p", "log2", "sign", "sqrt",
        "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosd", "cosh", "cospi",
        "deg2rad", "rad2deg", "sin", "sind", "sinh", "sinpi", "tan", "tand", "tanh",
    ] {
        let source =
            format!("z = complex(uint64(9223372036854775808), uint64(1)); out = {builtin}(z);");
        let err = execute_source(&source).expect_err(builtin);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ) || err
                .to_string()
                .contains("expected real single or double input")
                || err.to_string().contains("expected single or double input"),
            "{builtin} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn find_preserves_typed_complex_integer_values_through_vm_dispatch() {
    let vars = execute_source(
        "z = complex(uint64([0 9223372036854775808 18446744073709551615]), uint64([0 1 2])); [row, col, values] = find(z);",
    )
    .expect("find should preserve selected typed complex integer values");
    assert!(matches!(
        &vars[3],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![2, 1]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                        &IntegerStorage::U64(vec![1, 2]),
                    ))
    ));
}

#[test]
fn isequal_preserves_typed_complex_integer_precision_through_vm_dispatch() {
    let vars = execute_source(
        "base = uint64(9223372036854775808); next = base + 1; a = complex(base, uint64(1)); b = complex(next, uint64(1)); same = isequal(a, a); different = isequal(a, b); different_n = isequaln(a, b);",
    )
    .expect("isequal should compare typed complex integer storage exactly");
    assert_eq!(vars[4], Value::Bool(true));
    assert_eq!(vars[5], Value::Bool(false));
    assert_eq!(vars[6], Value::Bool(false));
}

#[test]
fn typed_complex_integer_relational_equality_preserves_exact_storage() {
    let vars = execute_source(
        "base = uint64(9223372036854775808); next = base + 1; z = complex(base, uint64(1)); w = complex(next, uint64(1)); same = z == z; different = z == w; builtin_same = eq(z, z); builtin_different = eq(z, w); ne_same = z ~= z; ne_different = ne(z, w);",
    )
    .expect("typed complex integer equality should execute exactly");

    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.0] && tensor.shape == vec![1, 1]
    ));
    assert!(matches!(
        &vars[5],
        Value::Tensor(tensor) if tensor.materialize_f64() == vec![0.0] && tensor.shape == vec![1, 1]
    ));
    assert_eq!(vars[6], Value::Bool(true));
    assert_eq!(vars[7], Value::Bool(false));
    assert!(matches!(
        &vars[8],
        Value::Tensor(tensor) if tensor.materialize_f64() == vec![0.0] && tensor.shape == vec![1, 1]
    ));
    assert_eq!(vars[9], Value::Bool(true));
}

#[test]
fn typed_complex_integer_arithmetic_reductions_are_rejected_before_f64_coercion() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for operation in ["sum(z)", "cumsum(z)", "cumprod(z)", "diff([z z])"] {
        let source =
            format!("z = complex(uint64(9223372036854775808), uint64(1)); out = {operation};");
        let err = execute_source(&source).expect_err(operation);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "{operation} returned an unexpected error: {err}"
        );
    }

    let vars = execute_source(
        "z = complex(uint64(9223372036854775808), uint64(1)); unchanged = diff(z, 0);",
    )
    .expect("diff with order zero should preserve its input");
    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![1_u64 << 63]),
                    &IntegerStorage::U64(vec![1]),
                ))
    ));
}

#[test]
fn typed_complex_integer_extrema_and_mean_are_rejected_before_f64_coercion() {
    for operation in ["mean(z)", "min(z)", "max(z)", "cummin(z)", "cummax(z)"] {
        let source =
            format!("z = complex(uint64(9223372036854775808), uint64(1)); out = {operation};");
        let err = execute_source(&source).expect_err(operation);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "{operation} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_ordering_operations_are_rejected_before_f64_coercion() {
    for operation in ["sort(z)", "argsort(z)", "issorted(z)", "sortrows(z)"] {
        let source = format!(
            "z = complex(uint64([9223372036854775808 1]), uint64([1 0])); out = {operation};"
        );
        let err = execute_source(&source).expect_err(operation);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "{operation} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_set_operations_are_rejected_before_f64_coercion() {
    for operation in [
        "unique(z)",
        "union(z, z)",
        "intersect(z, z)",
        "setdiff(z, z)",
        "setxor(z, z)",
        "ismember(z, z)",
    ] {
        let source = format!(
            "z = complex(uint64([9223372036854775808 1]), uint64([1 0])); out = {operation};"
        );
        let err = execute_source(&source).expect_err(operation);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "{operation} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn unique_preserves_every_integer_row_class_through_vm_dispatch() {
    let vars = execute_source(
        "a = unique(int8([3 1 3])); b = unique(int16([3 1 3])); c = unique(int32([3 1 3])); d = unique(int64([3 1 3])); e = unique(uint8([3 1 3])); f = unique(uint16([3 1 3])); g = unique(uint32([3 1 3])); h = unique(uint64([18446744073709551615 0 18446744073709551615]));",
    )
    .expect("compiled integer unique rows");
    let expected = [
        IntegerStorage::I8(vec![1, 3]),
        IntegerStorage::I16(vec![1, 3]),
        IntegerStorage::I32(vec![1, 3]),
        IntegerStorage::I64(vec![1, 3]),
        IntegerStorage::U8(vec![1, 3]),
        IntegerStorage::U16(vec![1, 3]),
        IntegerStorage::U32(vec![1, 3]),
        IntegerStorage::U64(vec![0, u64::MAX]),
    ];
    for (value, expected_storage) in vars.iter().zip(expected.iter()) {
        assert!(matches!(
            value,
            Value::Tensor(tensor)
                if tensor.shape == vec![1, 2]
                    && tensor.integer_storage() == Some(expected_storage)
        ));
    }
}

#[test]
fn typed_complex_integer_fft_operations_are_rejected_before_f64_coercion() {
    for operation in [
        "fft(z)", "ifft(z)", "fft2(z)", "ifft2(z)", "fftn(z)", "ifftn(z)",
    ] {
        let source = format!(
            "z = complex(uint64([9223372036854775808 1]), uint64([1 0])); out = {operation};"
        );
        let err = execute_source(&source).expect_err(operation);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "{operation} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_fft_shifts_preserve_exact_components() {
    let vars = execute_source(
        "z = complex(uint64([18446744073709551615 9223372036854775808 7]), uint64([5 6 7])); shifted = fftshift(z); restored = ifftshift(shifted); shifted_real = real(shifted); shifted_imag = imag(shifted); restored_real = real(restored); restored_imag = imag(restored);",
    )
    .expect("FFT shifts should preserve typed complex integer storage");

    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&IntegerStorage::U64(vec![7, u64::MAX, 1_u64 << 63]))
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![7, 5, 6]))
    ));
    assert!(matches!(
        &vars[5],
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63, 7]))
    ));
    assert!(matches!(
        &vars[6],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![5, 6, 7]))
    ));
}

#[test]
fn typed_complex_integer_blkdiag_preserves_exact_components() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "a = complex(uint64([18446744073709551615 9223372036854775808]), uint64([5 6])); b = complex(uint64([7; 8]), uint64([9; 10])); out = blkdiag(a, b); real_out = real(out); imag_out = imag(out);",
    )
    .expect("blkdiag should preserve typed complex integer storage");

    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![u64::MAX, 0, 0, 1_u64 << 63, 0, 0, 0, 7, 8]))
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![5, 0, 0, 6, 0, 0, 0, 9, 10]))
    ));
}

#[test]
fn typed_complex_integer_blkdiag_rejects_mixed_representations_before_f64_coercion() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "a = complex(uint64(9223372036854775808), uint64(1)); b = complex(uint32(2), uint32(3)); out = blkdiag(a, b);",
        "a = complex(uint64(9223372036854775808), uint64(1)); out = blkdiag(a, 2);",
    ] {
        let err = execute_source(source).expect_err("mixed blkdiag inputs must not coerce to f64");
        assert!(
            err.to_string()
                .contains("typed complex integer blkdiag inputs must all use the same integer class"),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_ndgrid_preserves_exact_components() {
    let vars = execute_source(
        "axis = complex(uint64([18446744073709551615 9223372036854775808]), uint64([5 6])); [x, y] = ndgrid(axis, [7 8]); real_x = real(x); imag_x = imag(x);",
    )
    .expect("ndgrid should preserve typed complex integer storage");

    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63, u64::MAX, 1_u64 << 63]))
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![5, 6, 5, 6]))
    ));
}

#[test]
fn typed_complex_integer_meshgrid_preserves_exact_components() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "axis = complex(uint64([18446744073709551615 9223372036854775808]), uint64([5 6])); [x, y] = meshgrid(axis, [7 8]); real_x = real(x); imag_x = imag(x);",
    )
    .expect("meshgrid should preserve typed complex integer storage");

    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX, 1_u64 << 63, 1_u64 << 63]))
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![5, 5, 6, 6]))
    ));
}

#[test]
fn typed_complex_integer_perms_preserves_exact_components() {
    let vars = execute_source(
        "z = complex(uint64([18446744073709551615 9223372036854775808 7]), uint64([5 6 7])); out = perms(z); real_out = real(out); imag_out = imag(out);",
    )
    .expect("perms should preserve typed complex integer storage");

    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.shape == vec![6, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![
                        7, 7, 1_u64 << 63, 1_u64 << 63, u64::MAX, u64::MAX,
                        1_u64 << 63, u64::MAX, 7, u64::MAX, 7, 1_u64 << 63,
                        u64::MAX, 1_u64 << 63, u64::MAX, 7, 1_u64 << 63, 7,
                    ]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.shape == vec![6, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![
                        7, 7, 6, 6, 5, 5, 6, 5, 7, 5, 7, 6, 5, 6, 5, 7, 6, 7,
                    ]))
    ));
}

#[test]
fn typed_complex_integer_toeplitz_preserves_exact_components() {
    let vars = execute_source(
        "c = complex(uint64([18446744073709551615 7]), uint64([5 6])); r = complex(uint64([18446744073709551615 9223372036854775808 9]), uint64([5 8 10])); out = toeplitz(c, r); real_out = real(out); imag_out = imag(out);",
    )
    .expect("two-vector toeplitz should preserve typed complex integer storage");

    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 3]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![u64::MAX, 7, 1_u64 << 63, u64::MAX, 9, 1_u64 << 63]))
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 3]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![5, 6, 8, 5, 10, 8]))
    ));

    let vars = execute_source(
        "z = complex(int64([1 2]), int64([5 6])); out = toeplitz(z); real_out = real(out); imag_out = imag(out);",
    )
    .expect("one-vector toeplitz should use typed complex conjugation");
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::I64(vec![1, 2, 2, 1]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::I64(vec![5, -6, 6, 5]))
    ));
}

#[test]
fn typed_complex_integer_num2cell_preserves_exact_components() {
    let vars = execute_source(
        "z = complex(uint64([18446744073709551615 9223372036854775808]), uint64([5 6])); c = num2cell(z); item = c{2}; item_real = real(item); item_imag = imag(item); grouped = num2cell(z, 2); grouped_item = grouped{1}; grouped_real = real(grouped_item); grouped_imag = imag(grouped_item);",
    )
    .expect("num2cell should preserve typed complex integer storage");

    assert!(matches!(
        &vars[2],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 1]
                && tensor.integer_storage().as_ref().is_some_and(|storage|
                    storage.real == IntegerStorage::U64(vec![1_u64 << 63])
                        && storage.imag == IntegerStorage::U64(vec![6]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Int(IntValue::U64(value)) if *value == 1_u64 << 63
    ));
    assert!(matches!(&vars[4], Value::Int(IntValue::U64(6))));
    assert!(matches!(
        &vars[7],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
    ));
    assert!(matches!(
        &vars[6],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage().as_ref().is_some_and(|storage|
                    storage.real == IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63])
                        && storage.imag == IntegerStorage::U64(vec![5, 6]))
    ));
    assert!(matches!(
        &vars[8],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![5, 6]))
    ));
}

#[test]
fn typed_complex_integer_cell2mat_preserves_exact_components() {
    let vars = execute_source(
        "z = complex(uint64([18446744073709551615 9223372036854775808]), uint64([5 6])); c = num2cell(z); out = cell2mat(c); real_out = real(out); imag_out = imag(out);",
    )
    .expect("cell2mat should preserve typed complex integer storage");

    assert!(matches!(
        &vars[2],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage().as_ref().is_some_and(|storage|
                    storage.real == IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63])
                        && storage.imag == IntegerStorage::U64(vec![5, 6]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![5, 6]))
    ));
}

#[test]
fn typed_complex_integer_mat2cell_preserves_exact_components() {
    let vars = execute_source(
        "z = complex(uint64([18446744073709551615 9223372036854775808]), uint64([5 6])); c = mat2cell(z, [1], [1 1]); item = c{1,2}; item_real = real(item); item_imag = imag(item); out = cell2mat(c); out_real = real(out); out_imag = imag(out);",
    )
    .expect("mat2cell should preserve typed complex integer storage");

    assert!(matches!(
        &vars[2],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 1]
                && tensor.integer_storage().as_ref().is_some_and(|storage|
                    storage.real == IntegerStorage::U64(vec![1_u64 << 63])
                        && storage.imag == IntegerStorage::U64(vec![6]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Int(IntValue::U64(value)) if *value == 1_u64 << 63
    ));
    assert!(matches!(&vars[4], Value::Int(IntValue::U64(6))));
    assert!(matches!(
        &vars[6],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
    ));
    assert!(matches!(
        &vars[7],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![5, 6]))
    ));
}

#[test]
fn typed_complex_integer_numerical_integration_is_rejected_before_f64_coercion() {
    for operation in [
        "gradient(z)",
        "trapz(z)",
        "trapz(0, z)",
        "trapz(z, 1)",
        "trapz(0, z, 1)",
        "cumtrapz(z)",
        "cumtrapz(0, z)",
        "cumtrapz(z, 1)",
        "cumtrapz(0, z, 1)",
    ] {
        let source = format!(
            "z = complex(uint64([9223372036854775808 9223372036854775809]), uint64([1 1])); out = {operation};"
        );
        let err = execute_source(&source).expect_err(operation);
        assert!(
            err.to_string().contains(
                "operations involving complex numbers with integer types are not supported"
            ),
            "{operation} returned an unexpected error: {err}"
        );
    }
}

#[test]
fn typed_complex_integer_truthiness_uses_exact_paired_storage() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "z = complex(uint64([0 9223372036854775808 0]), uint64([0 0 1])); l = logical(z); n = ~z; a = all(z, 'all'); b = any(z, 'all'); c = nnz(z); [r, col, values] = find(z); p = z & z; q = z | 0; x = xor(z, 0);",
    )
    .expect("typed complex integer truthiness should execute");

    for index in [1, 9, 10, 11] {
        assert!(
            matches!(
                &vars[index],
                Value::LogicalArray(array) if array.data == vec![0, 1, 1]
            ),
            "unexpected value at {index}: {:?}",
            vars[index]
        );
    }
    assert!(matches!(
        &vars[2],
        Value::LogicalArray(array) if array.data == vec![1, 0, 0]
    ));
    assert!(!logical_truth(&vars[3]));
    assert!(logical_truth(&vars[4]));
    assert_eq!(vars[5], Value::Num(2.0));
    assert!(matches!(
        &vars[6],
        Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.0, 1.0]
    ));
    assert!(matches!(
        &vars[7],
        Value::Tensor(tensor) if tensor.materialize_f64() == vec![2.0, 3.0]
    ));
    assert!(matches!(
        &vars[8],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![9_223_372_036_854_775_808, 0]),
                    &IntegerStorage::U64(vec![0, 1]),
                ))
    ));
}

#[test]
fn complex_integer_slice_assignment_preserves_exact_components_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(uint64([1 2; 3 4]), uint64([10 20; 30 40])); rhs = complex(uint64([18446744073709551615 9223372036854775808; 18446744073709551615 9223372036854775808]), uint64([7 8; 7 8])); a(:, :) = rhs; ar = real(a); ai = imag(a); b = complex(uint64([1 2; 3 4]), uint64([10 20; 30 40])); b(1:end, :) = rhs;",
    )
    .expect("typed complex slice assignment should execute");

    assert!(matches!(
        &vars[0],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![u64::MAX, u64::MAX, 9_223_372_036_854_775_808, 9_223_372_036_854_775_808]),
                    &IntegerStorage::U64(vec![7, 7, 8, 8]),
                ))
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX, 9_223_372_036_854_775_808, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![7, 7, 8, 8]))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![u64::MAX, u64::MAX, 9_223_372_036_854_775_808, 9_223_372_036_854_775_808]),
                    &IntegerStorage::U64(vec![7, 7, 8, 8]),
                ))
    ));
}

#[test]
fn complex_integer_shape_transforms_preserve_exact_components_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(uint64(reshape([9223372036854775808 18446744073709551615 3 4], 2, 2)), uint64(reshape([7 8 9 10], 2, 2))); p = permute(a, [2 1]); q = ipermute(p, [2 1]); r = repmat(a, 2, 2); s = squeeze(reshape(a, 1, 2, 2, 1)); f = flip(a); t = rot90(a); h = circshift(a, [1 1]); e = repelem(a, [1 2], 1); d = diag(a); m = diag(d); u = triu(a); l = tril(a); qr = real(q); rr = real(r); sr = real(s);",
    )
    .expect("typed complex integer shape transforms should execute");

    assert!(matches!(
        &vars[2],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, 3, 4]),
                    &IntegerStorage::U64(vec![7, 8, 9, 10]),
                ))
    ));
    assert!(matches!(
        &vars[3],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![4, 4]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![
                            9_223_372_036_854_775_808,
                            u64::MAX,
                            9_223_372_036_854_775_808,
                            u64::MAX,
                            3,
                            4,
                            3,
                            4,
                            9_223_372_036_854_775_808,
                            u64::MAX,
                            9_223_372_036_854_775_808,
                            u64::MAX,
                            3,
                            4,
                            3,
                            4,
                        ]),
                        &IntegerStorage::U64(vec![7, 8, 7, 8, 9, 10, 9, 10, 7, 8, 7, 8, 9, 10, 9, 10]),
                    ))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, 3, 4]),
                        &IntegerStorage::U64(vec![7, 8, 9, 10]),
                    ))
    ));
    assert!(matches!(
        &vars[5],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808, 4, 3]),
                    &IntegerStorage::U64(vec![8, 7, 10, 9]),
                ))
    ));
    assert!(matches!(
        &vars[6],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![3, 9_223_372_036_854_775_808, 4, u64::MAX]),
                    &IntegerStorage::U64(vec![9, 7, 10, 8]),
                ))
    ));
    assert!(matches!(
        &vars[7],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![4, 3, u64::MAX, 9_223_372_036_854_775_808]),
                    &IntegerStorage::U64(vec![10, 9, 8, 7]),
                ))
    ));
    assert!(matches!(
        &vars[8],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, u64::MAX, 3, 4, 4]),
                        &IntegerStorage::U64(vec![7, 8, 8, 9, 10, 10]),
                    ))
    ));
    for (index, real, imag, shape) in [
        (
            9,
            vec![9_223_372_036_854_775_808, 4],
            vec![7, 10],
            vec![2, 1],
        ),
        (
            10,
            vec![9_223_372_036_854_775_808, 0, 0, 4],
            vec![7, 0, 0, 10],
            vec![2, 2],
        ),
        (
            11,
            vec![9_223_372_036_854_775_808, 0, 3, 4],
            vec![7, 0, 9, 10],
            vec![2, 2],
        ),
        (
            12,
            vec![9_223_372_036_854_775_808, u64::MAX, 0, 4],
            vec![7, 8, 0, 10],
            vec![2, 2],
        ),
    ] {
        assert!(matches!(
            &vars[index],
            Value::ComplexTensor(tensor)
                if tensor.shape == shape
                    && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                        == Some((&IntegerStorage::U64(real), &IntegerStorage::U64(imag)))
        ));
    }
}

#[test]
fn complex_single_shape_transforms_preserve_native_storage_through_vm_dispatch() {
    let expressions = [
        ("reshape", "reshape(a, 1, 4)"),
        ("permute", "permute(a, [2 1])"),
        ("ipermute", "ipermute(permute(a, [2 1]), [2 1])"),
        ("transpose", "a.'"),
        ("ctranspose", "a'"),
        ("squeeze", "squeeze(reshape(a, 1, 2, 2, 1))"),
        ("shiftdim", "shiftdim(a, 1)"),
        ("circshift", "circshift(a, [1 1])"),
        ("flip", "flip(a)"),
        ("repmat", "repmat(a, 2, 1)"),
        ("repelem", "repelem(a, [1 2], 1)"),
        ("rot90", "rot90(a)"),
        ("diag", "diag(a)"),
        ("tril", "tril(a)"),
        ("triu", "triu(a)"),
    ];
    for (name, expression) in expressions {
        let source = format!(
            "a = complex(single(reshape([1.25 -2.5 3.75 -4.5], 2, 2)), single(reshape([5.5 6.25 -7.75 8.5], 2, 2))); result = {expression};"
        );
        let vars = execute_source(&source)
            .unwrap_or_else(|error| panic!("native complex single {name} should execute: {error}"));
        let mut complex_results = 0;
        for value in vars {
            if let Value::ComplexTensor(tensor) = value {
                complex_results += 1;
                assert_eq!(
                    tensor.numeric_dtype(),
                    NumericDType::F32,
                    "{name} widened native complex single storage"
                );
            }
        }
        assert!(
            complex_results >= 2,
            "{name} did not return a complex tensor"
        );
    }
}

#[test]
fn every_real_integer_class_survives_the_full_shape_matrix_through_vm_dispatch() {
    let classes = [
        ("int8", NumericDType::I8),
        ("int16", NumericDType::I16),
        ("int32", NumericDType::I32),
        ("int64", NumericDType::I64),
        ("uint8", NumericDType::U8),
        ("uint16", NumericDType::U16),
        ("uint32", NumericDType::U32),
        ("uint64", NumericDType::U64),
    ];
    for (class, expected_dtype) in classes {
        let source = format!(
            "a = {class}(reshape([1 2 3 4], 2, 2)); reshaped = reshape(a, 1, 4); permuted = permute(a, [2 1]); ipermuted = ipermute(permuted, [2 1]); transposed = a.'; conjugated = a'; squeezed = squeeze(reshape(a, 1, 2, 2, 1)); shifted = shiftdim(a, 1); circulated = circshift(a, [1 1]); flipped = flip(a); tiled = repmat(a, 2, 1); repeated = repelem(a, [1 2], 1); rotated = rot90(a); diagonalized = diag(a); lower = tril(a); upper = triu(a);"
        );
        let vars = execute_source(&source)
            .unwrap_or_else(|error| panic!("{class} shape matrix should execute: {error}"));
        let mut integer_results = 0;
        for value in vars {
            if let Value::Tensor(tensor) = value {
                if tensor.integer_storage().is_some() {
                    integer_results += 1;
                    assert_eq!(
                        tensor.numeric_dtype(),
                        expected_dtype,
                        "{class} shape operation changed class"
                    );
                }
            }
        }
        assert_eq!(
            integer_results, 17,
            "{class} shape matrix did not return every typed result"
        );
    }
}

#[test]
fn empty_complex_single_shape_transforms_preserve_native_storage_through_vm_dispatch() {
    let expressions = [
        ("reshape", "reshape(a, 0, 1)"),
        ("permute", "permute(a, [2 1])"),
        ("ipermute", "ipermute(permute(a, [2 1]), [2 1])"),
        ("transpose", "a.'"),
        ("squeeze", "squeeze(reshape(a, 1, 0, 3, 1))"),
        ("shiftdim", "shiftdim(a, 1)"),
        ("circshift", "circshift(a, [1 1])"),
        ("flip", "flip(a)"),
        ("repmat", "repmat(a, 2, 1)"),
        ("repelem", "repelem(a, 2, 1)"),
        ("rot90", "rot90(a)"),
        ("diag", "diag(a)"),
        ("tril", "tril(a)"),
        ("triu", "triu(a)"),
    ];
    for (name, expression) in expressions {
        let source = format!(
            "a = complex(zeros(0, 3, 'single'), zeros(0, 3, 'single')); result = {expression};"
        );
        let vars = execute_source(&source)
            .unwrap_or_else(|error| panic!("empty complex single {name} should execute: {error}"));
        let mut complex_results = 0;
        for value in vars {
            if let Value::ComplexTensor(tensor) = value {
                complex_results += 1;
                assert_eq!(
                    tensor.numeric_dtype(),
                    NumericDType::F32,
                    "{name} widened empty native complex single storage"
                );
                assert!(tensor.is_empty(), "{name} synthesized empty elements");
            }
        }
        assert!(
            complex_results >= 2,
            "{name} did not return an empty complex tensor"
        );
    }
}

#[test]
fn complex_integer_nonconjugate_transpose_preserves_exact_components_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(int8([1 -128; 3 4]), int8([-128 2; 3 4])); t = a.'; z = complex(uint64([9223372036854775808 18446744073709551615]), uint64([0 0])); zt = z.';",
    )
    .expect("typed complex integer nonconjugate transpose should execute");

    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I8(vec![1, -128, 3, 4]),
                    &IntegerStorage::I8(vec![-128, 2, 3, 4]),
                ))
    ));
    assert!(matches!(
        &vars[3],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]),
                    &IntegerStorage::U64(vec![0, 0]),
            ))
    ));
}

#[test]
fn complex_integer_ctranspose_is_rejected_through_vm_dispatch() {
    for source in [
        "z = complex(int8([1 2]), int8([3 4])); out = z';",
        "z = complex(uint64([9223372036854775808 18446744073709551615]), uint64([0 1])); out = ctranspose(z);",
    ] {
        let error =
            execute_source(source).expect_err("complex integer ctranspose must be rejected");
        assert_eq!(
            error.identifier(),
            Some("RunMat:ctranspose:InvalidInput"),
            "{source}: unexpected error: {error}"
        );
        assert!(error.to_string().contains("complex integer"), "{source}");
    }
}

#[test]
fn transpose_operators_reject_nd_arrays_but_accept_trailing_singletons() {
    for source in [
        "x = reshape(int8(1:8), 2, 2, 2); out = x.';",
        "x = reshape(int8(1:8), 2, 2, 2); out = x';",
    ] {
        let error = execute_source(source).expect_err("N-D transpose must be rejected");
        assert!(
            error.to_string().contains("use permute"),
            "{source}: {error}"
        );
    }

    let vars = execute_source("x = reshape(int8([1 2; 3 4]), 2, 2, 1); t = x.'; c = x';")
        .expect("explicit trailing singleton dimensions remain a matrix");
    for value in [&vars[1], &vars[2]] {
        assert!(matches!(
            value,
            Value::Tensor(tensor)
                if tensor.shape == vec![2, 2, 1]
                    && tensor.integer_storage() == Some(&IntegerStorage::I8(vec![1, 2, 3, 4]))
        ));
    }
}

#[test]
fn complex_integer_concatenation_preserves_exact_components_through_vm_dispatch() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "a = complex(uint64([9223372036854775808 18446744073709551615]), uint64([7 8])); b = complex(uint64([1 2]), uint64([3 4])); h = horzcat(a, b); v = vertcat(a, b); c = cat(2, a, uint64([9 10])); q = [a b];",
    )
    .expect("typed complex integer concatenation should execute");

    assert!(matches!(
        &vars[2],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (storage.real.clone(), storage.imag.clone()))
                == Some((
                    IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, 1, 2]),
                    IntegerStorage::U64(vec![7, 8, 3, 4]),
                ))
    ));
    assert!(matches!(
        &vars[3],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_storage().as_ref().map(|storage| (storage.real.clone(), storage.imag.clone()))
                    == Some((
                        IntegerStorage::U64(vec![9_223_372_036_854_775_808, 1, u64::MAX, 2]),
                        IntegerStorage::U64(vec![7, 3, 8, 4]),
                    ))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (storage.real.clone(), storage.imag.clone()))
                == Some((
                    IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, 9, 10]),
                    IntegerStorage::U64(vec![7, 8, 0, 0]),
                ))
    ));
    assert!(matches!(
        &vars[5],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (storage.real.clone(), storage.imag.clone()))
                == Some((
                    IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, 1, 2]),
                    IntegerStorage::U64(vec![7, 8, 3, 4]),
                ))
    ));
}

#[test]
fn mixed_concatenation_precedence_and_nd_shapes_flow_through_vm_dispatch() {
    let vars = execute_source(
        "m = uint16([1 2; 3 4]); c = {'head'}; wrapped = [c m]; text = [\"id\" uint64(9223372036854775808) true]; chars = cat(3, 'ab', 'cd'); cells = cat(3, {1}, {2});",
    )
    .expect("mixed concatenation should execute");

    assert!(matches!(
        &vars[2],
        Value::Cell(cell)
            if cell.shape == vec![1, 2]
                && cell.data[0] == Value::CharArray(runmat_builtins::CharArray::new_row("head"))
                && cell.data[1] == vars[0]
    ));
    assert!(matches!(
        &vars[3],
        Value::StringArray(array)
            if array.shape == vec![1, 3]
                && array.data == vec!["id", "9223372036854775808", "1"]
    ));
    assert!(matches!(
        &vars[4],
        Value::CharArray(array)
            if array.shape == vec![1, 2, 2]
                && array.data.iter().collect::<String>() == "abcd"
    ));
    assert!(matches!(
        &vars[5],
        Value::Cell(cell)
            if cell.shape == vec![1, 1, 2]
                && cell.data == vec![Value::Num(1.0), Value::Num(2.0)]
    ));

    let error =
        execute_source("bad = ['a' true];").expect_err("char plus logical must be rejected");
    assert!(error.to_string().contains("char"), "{error}");
}

#[test]
fn integer_casts_preserve_complex_storage_for_every_integer_class_through_vm_dispatch() {
    let vars = execute_source(
        "z = complex([1.5 -2.5], [0.49 -1.5]); a = int8(z); b = int16(z); c = int32(z); d = int64(z); e = uint8(z); f = uint16(z); g = uint32(z); h = uint64(z); flags = [isreal(a) isreal(b) isreal(c) isreal(d) isreal(e) isreal(f) isreal(g) isreal(h)]; q = complex(uint64([9223372036854775808 18446744073709551615]), uint64([1 2])); q64 = int64(q);",
    )
    .expect("complex integer casts should execute");

    let expected = vec![
        (
            IntegerStorage::I8(vec![2, -3]),
            IntegerStorage::I8(vec![0, -2]),
        ),
        (
            IntegerStorage::I16(vec![2, -3]),
            IntegerStorage::I16(vec![0, -2]),
        ),
        (
            IntegerStorage::I32(vec![2, -3]),
            IntegerStorage::I32(vec![0, -2]),
        ),
        (
            IntegerStorage::I64(vec![2, -3]),
            IntegerStorage::I64(vec![0, -2]),
        ),
        (
            IntegerStorage::U8(vec![2, 0]),
            IntegerStorage::U8(vec![0, 0]),
        ),
        (
            IntegerStorage::U16(vec![2, 0]),
            IntegerStorage::U16(vec![0, 0]),
        ),
        (
            IntegerStorage::U32(vec![2, 0]),
            IntegerStorage::U32(vec![0, 0]),
        ),
        (
            IntegerStorage::U64(vec![2, 0]),
            IntegerStorage::U64(vec![0, 0]),
        ),
    ];
    for (value, (real, imag)) in vars[1..9].iter().zip(expected) {
        let Value::ComplexTensor(tensor) = value else {
            panic!("integer cast must preserve complex tensor storage: {value:?}");
        };
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(
            tensor
                .integer_storage()
                .as_ref()
                .map(|storage| (&storage.real, &storage.imag)),
            Some((&real, &imag))
        );
    }
    assert!(matches!(
        &vars[9],
        Value::LogicalArray(flags) if flags.data == vec![0; 8]
    ));
    assert!(matches!(
        &vars[11],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I64(vec![i64::MAX, i64::MAX]),
                    &IntegerStorage::I64(vec![1, 2]),
                ))
    ));
}

#[test]
fn conj_preserves_typed_complex_storage_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(int8([1 -2]), int8([3 -128])); b = conj(a); u = complex(uint64([9223372036854775808 18446744073709551615]), uint64([1 2])); v = conj(u); z = conj(complex(uint16(7), uint16(0))); tf = [isreal(b) isreal(v) isreal(z)]; w = conj(complex(7)); tw = isreal(w);",
    )
    .expect("typed complex conjugates should execute");

    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I8(vec![1, -2]),
                    &IntegerStorage::I8(vec![-3, i8::MAX]),
                ))
    ));
    assert!(matches!(
        &vars[3],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                    &IntegerStorage::U64(vec![0, 0]),
                ))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U16(vec![7]),
                    &IntegerStorage::U16(vec![0]),
                ))
    ));
    assert!(matches!(
        &vars[5],
        Value::LogicalArray(flags) if flags.data == vec![0; 3]
    ));
    assert!(matches!(&vars[6], Value::Complex(re, im) if *re == 7.0 && *im == 0.0));
    assert!(!logical_truth(&vars[7]));
}

#[test]
fn typed_complex_integer_deletion_preserves_paired_exact_storage_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(uint64([9223372036854775808 2 18446744073709551615]), uint64([7 8 9])); a(2) = []; b = complex(int16([1 2 3 4]), int16([-1 -2 -3 -4])); b([4 2]) = [];",
    )
    .expect("typed complex integer deletion should execute");

    assert!(matches!(
        &vars[0],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                        &IntegerStorage::U64(vec![7, 9]),
                    ))
    ));
    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::I16(vec![1, 3]),
                        &IntegerStorage::I16(vec![-1, -3]),
                    ))
    ));
}

#[test]
fn typed_complex_integer_scalar_assignment_preserves_paired_exact_storage_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(uint64([9223372036854775808 2]), uint64([7 8])); a(2) = complex(uint64(18446744073709551615), uint64(3)); a(3) = complex(4, 5); b = complex(int8([1 2; 3 4]), int8([-1 -2; -3 -4])); b(2, 1) = complex(int8(-128), int8(127));",
    )
    .expect("typed complex integer scalar assignment should execute");

    assert!(matches!(
        &vars[0],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 3]
                && tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX, 4]),
                        &IntegerStorage::U64(vec![7, 3, 5]),
                    ))
    ));
    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I8(vec![1, i8::MIN, 2, 4]),
                    &IntegerStorage::I8(vec![-1, i8::MAX, -2, -4]),
                ))
    ));
}

#[test]
fn indexed_assignment_does_not_demote_last_zeroed_complex_integer_component() {
    let vars = execute_source(
        "a = complex(uint16([1 2]), uint16([3 0])); a(1) = uint16(9); still_complex = ~isreal(a); ar = real(a); ai = imag(a);",
    )
    .expect("real indexed assignment should retain complex integer storage");

    assert!(matches!(
        &vars[0],
        Value::ComplexTensor(tensor)
            if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U16(vec![9, 2]),
                    &IntegerStorage::U16(vec![0, 0]),
                ))
    ));
    assert!(logical_truth(&vars[1]));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U16(vec![9, 2]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U16(vec![0, 0]))
    ));
}

#[test]
fn issparse_reports_sparse_storage_through_vm_dispatch() {
    let vars = execute_source(
        "s = sparse([1 2], [1 2], [10 20], 2, 2); a = issparse(s); b = issparse([10 0; 0 20]); c = issparse(42);",
    )
    .unwrap();
    assert!(logical_truth(&vars[1]));
    assert!(!logical_truth(&vars[2]));
    assert!(!logical_truth(&vars[3]));
}

#[test]
fn full_densifies_sparse_storage_through_vm_dispatch() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 2], [10 30 20], 3, 2); a = full(s); b = full([1 0; 0 2]); c = issparse(a);",
    )
    .unwrap();
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64() == vec![10.0, 0.0, 30.0, 0.0, 20.0, 0.0]
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.materialize_f64() == vec![1.0, 0.0, 0.0, 2.0]
    ));
    assert!(!logical_truth(&vars[3]));
}

#[test]
fn sparse_indexing_reads_stored_unstored_and_column_major_values() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 3], [10 30 23], 3, 3); a = s(1,1); b = s(2,1); c = s(8); d = s(end,end); tf = [issparse(a), issparse(b), issparse(c), issparse(d)]; e = s([1],[1]);",
    )
    .unwrap();
    assert_eq!(sparse_scalar(&vars[1]), 10.0);
    assert_eq!(sparse_scalar(&vars[2]), 0.0);
    assert_eq!(sparse_scalar(&vars[3]), 23.0);
    assert_eq!(sparse_scalar(&vars[4]), 0.0);
    assert!(matches!(
        &vars[5],
        Value::LogicalArray(logical)
            if logical.shape == vec![1, 4] && logical.data == vec![1, 1, 1, 1]
    ));
    assert_eq!(sparse_scalar(&vars[6]), 10.0);
}

#[test]
fn sparse_slice_indexing_preserves_sparse_outputs() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 3], [10 30 23], 3, 3); c = full(s(:,1)); r = full(s(2,:)); sub = s([1 2], [1 3]); d = full(sub); tf = issparse(sub); lin = full(s(:)); lin_tf = issparse(s(:)); pick = full(s([1 8])); pick_tf = issparse(s([1 8])); rev = full(s(3:-1:1,1)); full_range = full(s(1:end)); full_range_tf = issparse(s(1:end));",
    )
    .unwrap();
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor) if tensor.shape == vec![3, 1] && tensor.materialize_f64() == vec![10.0, 0.0, 30.0]
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor) if tensor.shape == vec![1, 3] && tensor.materialize_f64() == vec![0.0, 0.0, 23.0]
    ));
    assert!(matches!(
        &vars[3],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![2, 2]
                && sparse.get(0, 0) == Some(10.0)
                && sparse.get(1, 0).unwrap_or(0.0) == 0.0
                && sparse.get(0, 1).unwrap_or(0.0) == 0.0
                && sparse.get(1, 1) == Some(23.0)
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.materialize_f64() == vec![10.0, 0.0, 0.0, 23.0]
    ));
    assert!(logical_truth(&vars[5]));
    assert!(matches!(
        &vars[6],
        Value::Tensor(tensor)
            if tensor.shape == vec![9, 1]
                && tensor.materialize_f64() == vec![10.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 23.0, 0.0]
    ));
    assert!(logical_truth(&vars[7]));
    assert!(matches!(
        &vars[8],
        Value::Tensor(tensor) if tensor.shape == vec![1, 2] && tensor.materialize_f64() == vec![10.0, 23.0]
    ));
    assert!(logical_truth(&vars[9]));
    assert!(matches!(
        &vars[10],
        Value::Tensor(tensor) if tensor.shape == vec![3, 1] && tensor.materialize_f64() == vec![30.0, 0.0, 10.0]
    ));
    assert!(matches!(
        &vars[11],
        Value::Tensor(tensor)
            if tensor.shape == vec![9, 1]
                && tensor.materialize_f64() == vec![10.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 23.0, 0.0]
    ));
    assert!(logical_truth(&vars[12]));
}

#[test]
fn sparse_integer_script_surface_is_runmat_mode_only() {
    {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        for source in [
            "s = sparse(uint64([1 2]));",
            "s = sparse(uint64([1 2])); t = s(:);",
            "s = sparse(uint64([1 2])); s(1) = uint64(3);",
        ] {
            let error = execute_source(source).expect_err("MATLAB mode must reject sparse integer");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:SparseIntegerExtension")
            );
        }
    }
    {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
        let vars =
            execute_source("s = sparse(uint64([1 2])); s(1) = uint64(3); t = s(:); f = full(t);")
                .expect("RunMat mode sparse integer extension");
        assert!(matches!(
            &vars[0],
            Value::SparseTensor(sparse)
                if sparse.integer_storage() == Some(&IntegerStorage::U64(vec![3, 2]))
        ));
        assert!(matches!(
            &vars[2],
            Value::Tensor(tensor)
                if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![3, 2]))
        ));
    }
}

#[test]
fn typed_sparse_slice_indexing_preserves_uint64_storage() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 3], uint64([1 9223372036854775808 4]), 3, 3); a = s(3,1); z = s(2,1); lin = s(:); pick = s([1 3 8]); sub = s([3 1], [1 3]); empty = s([],1);",
    )
    .unwrap();

    let expected_scalar = IntegerStorage::U64(vec![9_223_372_036_854_775_808]);
    assert!(matches!(
        &vars[1],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![1, 1]
                && sparse.integer_storage() == Some(&expected_scalar)
    ));
    assert!(matches!(
        &vars[2],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![1, 1]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![]))
    ));
    assert!(matches!(
        &vars[3],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![9, 1]
                && sparse.row_indices == vec![0, 2, 7]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808, 4]))
    ));
    let Value::SparseTensor(pick) = &vars[4] else {
        panic!("expected typed sparse linear selection, got {:?}", vars[4]);
    };
    assert_eq!(pick.shape(), vec![1, 3]);
    assert_eq!(pick.col_ptrs, vec![0, 1, 2, 3]);
    assert_eq!(pick.row_indices, vec![0, 0, 0]);
    assert_eq!(
        pick.integer_storage(),
        Some(&IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808, 4]))
    );
    let Value::SparseTensor(sub) = &vars[5] else {
        panic!("expected typed sparse matrix selection, got {:?}", vars[5]);
    };
    assert_eq!(sub.shape(), vec![2, 2]);
    assert_eq!(sub.col_ptrs, vec![0, 2, 2]);
    assert_eq!(sub.row_indices, vec![0, 1]);
    assert_eq!(
        sub.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808, 1]))
    );
    assert!(matches!(
        &vars[6],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![0, 1]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![]))
    ));
}

#[test]
fn typed_sparse_find_preserves_exact_values_and_directional_order() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "s = sparse(uint64([0 9223372036854775808;18446744073709551615 0])); [i,j,v] = find(s); [il,jl,vl] = find(s,1,'last');",
    )
    .expect("execute typed sparse find");
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor) if tensor.shape == vec![2, 1] && tensor.materialize_f64() == vec![2.0, 1.0]
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor) if tensor.shape == vec![2, 1] && tensor.materialize_f64() == vec![1.0, 2.0]
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(&vars[4], Value::Num(value) if *value == 1.0));
    assert!(matches!(&vars[5], Value::Num(value) if *value == 2.0));
    assert_eq!(
        vars[6],
        Value::Int(IntValue::U64(9_223_372_036_854_775_808))
    );
}

#[test]
fn sparse_assignment_updates_scalar_and_selector_entries() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let vars = execute_source(
        "s = sparse([1], [1], [5], 2, 2); s(2,2) = 7; s(1) = 0; s(:,1) = [1;2]; s(1:2,2) = [3;4]; s([3 3]) = [5 6]; f = full(s); n = nnz(s); t = sparse(uint64([0 0;0 0])); t(:,1) = uint64([1;9223372036854775808]);",
    )
    .expect("execute sparse assignment");
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.0, 2.0, 6.0, 4.0]
    ));
    assert!(matches!(&vars[2], Value::Num(value) if *value == 4.0));
    assert!(matches!(
        &vars[3],
        Value::SparseTensor(sparse)
            if sparse.col_ptrs == vec![0, 2, 2]
                && sparse.row_indices == vec![0, 1]
                && sparse.integer_storage()
                    == Some(&IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808]))
    ));

    let deleted = execute_source(
        "s = sparse([1 3 2], [1 1 2], [1 3 2], 3, 2); s(:,1) = []; a = full(s); s([1 3],:) = []; b = full(s); t = sparse(uint64([1 0 9223372036854775808])); t(1,2) = []; c = full(t); u = sparse([1; 0; 3]); u(2) = []; d = full(u);",
    )
    .expect("execute sparse structural deletion");
    assert!(matches!(
        &deleted[1],
        Value::Tensor(tensor) if tensor.shape == vec![3, 1] && tensor.materialize_f64() == vec![0.0, 2.0, 0.0]
    ));
    assert!(matches!(
        &deleted[2],
        Value::Tensor(tensor) if tensor.shape == vec![1, 1] && tensor.materialize_f64() == vec![2.0]
    ));
    assert!(matches!(
        &deleted[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(
        &deleted[6],
        Value::Tensor(tensor) if tensor.shape == vec![2, 1] && tensor.materialize_f64() == vec![1.0, 3.0]
    ));

    let deletion_err = execute_source("s = sparse([1], [1], [5], 2, 2); s(1,1) = [];").unwrap_err();
    assert_eq!(
        deletion_err.identifier(),
        Some("RunMat:UnsupportedDeletion")
    );

    let expression_deleted = execute_source(
        "s = sparse([1 3 2], [1 1 2], [1 3 2], 3, 2); s(:,1) = []; rows = [1 3]; s(rows,:) = []; f = full(s);",
    )
    .expect("execute expression-backed sparse row deletion");
    assert!(matches!(
        &expression_deleted[2],
        Value::Tensor(tensor) if tensor.shape == vec![1, 1] && tensor.materialize_f64() == vec![2.0]
    ));

    let all_deleted =
        execute_source("s = sparse(uint64([1 0; 0 9223372036854775808])); s(:,:) = [];")
            .expect("delete all sparse entries structurally");
    assert!(matches!(
        &all_deleted[0],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![0, 0]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![]))
    ));

    let grown = execute_source(
        "s = sparse(uint64([1 0])); s(1,4) = uint64(9223372036854775808); a = full(s); s(3,6) = uint64(7); b = full(s); z = sparse(uint64([])); z(5) = uint64(9); c = full(z); q = sparse([1]); q(1,4) = 0; d = full(q);",
    )
    .expect("execute sparse scalar growth");
    assert!(matches!(
        &grown[1],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 4]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![1, 0, 0, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(
        &grown[2],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 6]
                && tensor.integer_storage().is_some()
                && tensor.integer_storage().and_then(|storage| storage.value_at(17)) == Some(IntValue::U64(7))
    ));
    assert!(matches!(
        &grown[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 5]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![0, 0, 0, 0, 9]))
    ));
    assert!(matches!(
        &grown[6],
        Value::Tensor(tensor) if tensor.shape == vec![1, 4] && tensor.materialize_f64() == vec![1.0, 0.0, 0.0, 0.0]
    ));

    let selector_grown = execute_source(
        "s = sparse(uint64([1 2;3 4])); s([3 4],[4 5]) = uint64([5 9223372036854775808;7 8]); a = full(s); r = 5:6; c = [6 8]; s(r,c) = uint64([9 10;11 12]); b = full(s); t = sparse(uint64([1;2])); t(:,4) = uint64([9223372036854775808;6]); q = full(t); u = sparse(uint64([1 0;0 2])); u([4],[5]) = uint64(0); ue = full(u); un = nnz(u);",
    )
    .expect("execute sparse selector growth");
    assert!(matches!(
        &selector_grown[1],
        Value::Tensor(tensor)
            if tensor.shape == vec![4, 5]
                && tensor.integer_storage().and_then(|storage| storage.value_at(18))
                    == Some(IntValue::U64(9_223_372_036_854_775_808))
                && tensor.integer_storage().and_then(|storage| storage.value_at(0))
                    == Some(IntValue::U64(1))
                && tensor.integer_storage().and_then(|storage| storage.value_at(4))
                    == Some(IntValue::U64(2))
    ));
    assert!(matches!(
        &selector_grown[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![6, 8]
                && tensor.integer_storage().and_then(|storage| storage.value_at(34))
                    == Some(IntValue::U64(9))
                && tensor.integer_storage().and_then(|storage| storage.value_at(47))
                    == Some(IntValue::U64(12))
                && tensor.integer_storage().and_then(|storage| storage.value_at(1))
                    == Some(IntValue::U64(3))
    ));
    assert!(matches!(
        &selector_grown[6],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 4]
                && tensor.integer_storage().and_then(|storage| storage.value_at(6))
                    == Some(IntValue::U64(9_223_372_036_854_775_808))
                && tensor.integer_storage().and_then(|storage| storage.value_at(7))
                    == Some(IntValue::U64(6))
    ));
    assert!(matches!(
        &selector_grown[7],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![4, 5]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![1, 2]))
    ));
    assert!(matches!(
        &selector_grown[8],
        Value::Tensor(tensor)
            if tensor.shape == vec![4, 5]
                && tensor.integer_storage().and_then(|storage| storage.value_at(0))
                    == Some(IntValue::U64(1))
                && tensor.integer_storage().and_then(|storage| storage.value_at(5))
                    == Some(IntValue::U64(2))
    ));
    assert!(matches!(&selector_grown[9], Value::Num(value) if *value == 2.0));

    let invalid_slice_err =
        execute_source("s = sparse([1], [1], [5], 2, 2); s([0]) = 0;").unwrap_err();
    assert_eq!(
        invalid_slice_err.identifier(),
        Some("RunMat:IndexOutOfBounds")
    );
}

#[test]
fn sparse_arithmetic_interops_with_dense_and_scalar_values() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 2], [10 30 20], 3, 2); t = sparse([3 1 2], [1 2 2], [5 7 -20], 3, 2); a = s + t; af = full(a); atf = issparse(a); b = s + 2; c = [1 2; 3 4; 5 6] - s; d = s .* [2 2; 3 3; 4 4]; df = full(d); dtf = issparse(d); e = 3 .* s; ef = full(e); etf = issparse(e);",
    )
    .unwrap();
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64() == vec![10.0, 0.0, 35.0, 7.0, 0.0, 0.0]
    ));
    assert!(logical_truth(&vars[4]));
    assert!(matches!(
        &vars[5],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64() == vec![12.0, 2.0, 32.0, 2.0, 22.0, 2.0]
    ));
    assert!(matches!(
        &vars[6],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64() == vec![-9.0, 3.0, -25.0, 2.0, -16.0, 6.0]
    ));
    assert!(matches!(
        &vars[8],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64() == vec![20.0, 0.0, 120.0, 0.0, 60.0, 0.0]
    ));
    assert!(logical_truth(&vars[9]));
    assert!(matches!(
        &vars[11],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64() == vec![30.0, 0.0, 90.0, 0.0, 60.0, 0.0]
    ));
    assert!(logical_truth(&vars[12]));
}

#[test]
fn sparse_arithmetic_handles_sparse_scalar_and_complex_interop() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 2], [10 30 20], 3, 2); cs = s + complex(1, 2); ct = complex(1, -1) .* s; sf = sparse(1, 1, 2, 1, 1) + s; sff = full(sf); sft = issparse(sf);",
    )
    .unwrap();
    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64()[0] == (11.0, 2.0)
                && tensor.materialize_f64()[1] == (1.0, 2.0)
                && tensor.materialize_f64()[2] == (31.0, 2.0)
    ));
    assert!(matches!(
        &vars[2],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64()[0] == (10.0, -10.0)
                && tensor.materialize_f64()[1] == (0.0, -0.0)
                && tensor.materialize_f64()[2] == (30.0, -30.0)
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.materialize_f64() == vec![12.0, 2.0, 32.0, 2.0, 22.0, 2.0]
    ));
    assert!(logical_truth(&vars[5]));
}

#[test]
fn bit_position_and_count_functions_require_scalar_or_exact_sizes_when_compiled() {
    let vars = execute_source(
        "a = uint16([1; 2]); \
         shifted = bitshift(a, [1; -1]); \
         got = bitget(a, [1; 2]); \
         set = bitset(a, [2; 1], [1; 0]);",
    )
    .expect("compiled bit position operations");
    for expected in [
        IntegerStorage::U16(vec![2, 1]),
        IntegerStorage::U16(vec![1, 1]),
        IntegerStorage::U16(vec![3, 2]),
    ] {
        assert!(vars.iter().any(|value| matches!(
            value,
            Value::Tensor(tensor) if tensor.integer_storage() == Some(&expected)
        )));
    }

    for source in [
        "out = bitshift(uint16([1; 2]), [1 2]);",
        "out = bitget(uint16([1; 2]), [1 2]);",
        "out = bitset(uint16([1; 2]), 1, [1 0]);",
        "out = bitset(uint16(1), [1; 2], [1 0]);",
    ] {
        let error = execute_source(source).expect_err("singleton expansion must reject");
        assert_eq!(
            error.identifier(),
            Some("RunMat:bitwise:SizeMismatch"),
            "{source}: unexpected error: {error}"
        );
    }
}

#[test]
fn bitand_and_bitor_integer_and_logical_contracts_execute_through_compiled_dispatch() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a = {constructor}([6 3]); b = {constructor}([3 1]); c = bitand(a, b); d = bitor(a, b);"
        );
        let vars = execute_source(&source).expect("compiled same-class bitand and bitor");
        let expected_and = match constructor {
            "int8" => IntegerStorage::I8(vec![2, 1]),
            "int16" => IntegerStorage::I16(vec![2, 1]),
            "int32" => IntegerStorage::I32(vec![2, 1]),
            "int64" => IntegerStorage::I64(vec![2, 1]),
            "uint8" => IntegerStorage::U8(vec![2, 1]),
            "uint16" => IntegerStorage::U16(vec![2, 1]),
            "uint32" => IntegerStorage::U32(vec![2, 1]),
            "uint64" => IntegerStorage::U64(vec![2, 1]),
            _ => unreachable!(),
        };
        let expected_or = match constructor {
            "int8" => IntegerStorage::I8(vec![7, 3]),
            "int16" => IntegerStorage::I16(vec![7, 3]),
            "int32" => IntegerStorage::I32(vec![7, 3]),
            "int64" => IntegerStorage::I64(vec![7, 3]),
            "uint8" => IntegerStorage::U8(vec![7, 3]),
            "uint16" => IntegerStorage::U16(vec![7, 3]),
            "uint32" => IntegerStorage::U32(vec![7, 3]),
            "uint64" => IntegerStorage::U64(vec![7, 3]),
            _ => unreachable!(),
        };
        assert!(matches!(
            &vars[2],
            Value::Tensor(tensor)
                if tensor.integer_storage() == Some(&expected_and)
        ));
        assert!(matches!(
            &vars[3],
            Value::Tensor(tensor)
                if tensor.integer_storage() == Some(&expected_or)
        ));
    }

    let vars = execute_source(
        "a = logical([1 1 0 0]); b = logical([1 0 1 0]); c = bitand(a, b); d = bitor(a, b); okc = islogical(c); okd = islogical(d);",
    )
    .expect("compiled logical bitand and bitor");
    assert!(matches!(
        &vars[2],
        Value::LogicalArray(array) if array.shape == vec![1, 4] && array.data == vec![1, 0, 0, 0]
    ));
    assert!(matches!(
        &vars[3],
        Value::LogicalArray(array) if array.shape == vec![1, 4] && array.data == vec![1, 1, 1, 0]
    ));
    assert!(logical_truth(&vars[4]));
    assert!(logical_truth(&vars[5]));
}

#[test]
fn bitshift_all_integer_value_and_count_classes_execute_through_compiled_dispatch() {
    for value_constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        for count_constructor in [
            "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        ] {
            let source = format!(
                "a = {value_constructor}([3 4]); k = {count_constructor}([1 2]); out = bitshift(a, k);"
            );
            let vars = execute_source(&source).unwrap_or_else(|error| {
                panic!("{value_constructor}/{count_constructor}: compiled bitshift failed: {error}")
            });
            let expected = match value_constructor {
                "int8" => IntegerStorage::I8(vec![6, 16]),
                "int16" => IntegerStorage::I16(vec![6, 16]),
                "int32" => IntegerStorage::I32(vec![6, 16]),
                "int64" => IntegerStorage::I64(vec![6, 16]),
                "uint8" => IntegerStorage::U8(vec![6, 16]),
                "uint16" => IntegerStorage::U16(vec![6, 16]),
                "uint32" => IntegerStorage::U32(vec![6, 16]),
                "uint64" => IntegerStorage::U64(vec![6, 16]),
                _ => unreachable!(),
            };
            assert!(
                matches!(
                    &vars[2],
                    Value::Tensor(tensor)
                        if tensor.integer_storage() == Some(&expected)
                ),
                "{value_constructor}/{count_constructor}: unexpected output {:?}",
                vars[2]
            );
        }
    }

    let vars = execute_source("out = bitshift(-4, -1, 'int8');")
        .expect("compiled assumedtype arithmetic shift");
    assert!(matches!(vars[0], Value::Num(value) if value == -2.0));
}

#[test]
fn blackman_all_integer_length_classes_execute_through_compiled_dispatch() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "n = {constructor}(5); a = blackman(n); b = blackman(n, 'periodic', 'single');"
        );
        let vars = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor}: compiled blackman failed: {error}"));
        assert!(matches!(
            &vars[1],
            Value::Tensor(tensor)
                if tensor.shape == vec![5, 1]
                    && tensor.numeric_dtype() == runmat_builtins::NumericDType::F64
        ));
        assert!(matches!(
            &vars[2],
            Value::Tensor(tensor)
                if tensor.shape == vec![5, 1]
                    && tensor.numeric_dtype() == runmat_builtins::NumericDType::F32
        ));
    }
}

#[test]
fn blanks_all_integer_length_classes_execute_through_compiled_dispatch() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!("n = {constructor}(3); a = blanks(n);");
        let vars = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor}: compiled blanks failed: {error}"));
        assert!(matches!(
            &vars[1],
            Value::CharArray(chars)
                if chars.shape == vec![1, 3] && chars.data == vec![' ', ' ', ' ']
        ));
    }
    let vars = execute_source("a = blanks(int16(-3));").expect("negative integer length");
    assert!(matches!(
        &vars[0],
        Value::CharArray(chars) if chars.shape == vec![1, 0] && chars.data.is_empty()
    ));
}

#[test]
fn blkdiag_mixed_dense_inputs_use_first_integer_class_in_compiled_dispatch() {
    let vars = execute_source(
        "a = blkdiag(300, single(-200.5), int8(5), uint16(500), true); b = blkdiag(uint16(5), int8(-3));",
    )
    .expect("compiled mixed integer blkdiag");
    assert!(matches!(
        &vars[0],
        Value::Tensor(tensor)
            if tensor.numeric_dtype() == runmat_builtins::NumericDType::I8
                && tensor.integer_storage()
                    == Some(&runmat_builtins::IntegerStorage::I8(vec![
                        127, 0, 0, 0, 0, 0, -128, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 127, 0, 0,
                        0, 0, 0, 1,
                    ]))
    ));
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor)
            if tensor.numeric_dtype() == runmat_builtins::NumericDType::U16
                && tensor.integer_storage()
                    == Some(&runmat_builtins::IntegerStorage::U16(vec![5, 0, 0, 0]))
    ));
}

#[test]
fn shiftdim_is_registered_and_preserves_exact_integer_shapes_through_vm_dispatch() {
    let vars = execute_source(
        "a = reshape(uint64([9223372036854775808 18446744073709551615 3 4]), 1, 2, 2); p = shiftdim(a, 1); base = reshape(a, 1, 1, 2, 2); [d, m] = shiftdim(base); n = shiftdim(a, -2);",
    )
    .expect("compiled shiftdim calls should execute");
    let expected = IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, 3, 4]);
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_storage() == Some(&expected)
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_storage() == Some(&expected)
    ));
    assert_eq!(vars[4], Value::Num(2.0));
    assert!(matches!(
        &vars[5],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 1, 1, 2, 2]
                && tensor.integer_storage() == Some(&expected)
    ));
}

#[test]
fn shiftdim_preserves_all_integer_classes_through_vm_dispatch() {
    let classes = [
        ("int8", "int8"),
        ("int16", "int16"),
        ("int32", "int32"),
        ("int64", "int64"),
        ("uint8", "uint8"),
        ("uint16", "uint16"),
        ("uint32", "uint32"),
        ("uint64", "uint64"),
    ];
    for (constructor, expected_class) in classes {
        let source = format!(
            "a = reshape({constructor}([1 2 3 4]), 1, 2, 2); b = shiftdim(a, 1); c = class(b);"
        );
        let vars = execute_source(&source).expect("compiled integer shiftdim call should execute");
        assert!(matches!(
            &vars[1],
            Value::Tensor(tensor)
                if tensor.shape == vec![2, 2]
                    && tensor.integer_storage().is_some_and(|storage| storage.class_name() == expected_class)
        ));
        assert_eq!(vars[2], Value::String(expected_class.into()));
    }
}
