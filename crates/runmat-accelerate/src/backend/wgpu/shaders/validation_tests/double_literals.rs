//! Check the actual binary64 literals in production FFT and random shaders.

use crate::backend::wgpu::shaders::{creation, fft};

fn parsed(source: &str) -> naga::Module {
    let source = crate::backend::wgpu::pipelines::substitute_tokens(source, 64, 8, None).unwrap();
    naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)))
}

fn named<'a>(function: &'a naga::Function, name: &str) -> &'a naga::Expression {
    let handle = function
        .named_expressions
        .iter()
        .find_map(|(handle, label)| (label == name).then_some(*handle))
        .unwrap_or_else(|| panic!("missing named expression {name}"));
    &function.expressions[handle]
}

fn binary64_leaf(
    expressions: &naga::Arena<naga::Expression>,
    expression: &naga::Expression,
) -> f64 {
    match expression {
        naga::Expression::Literal(naga::Literal::F64(value))
        | naga::Expression::Literal(naga::Literal::AbstractFloat(value)) => *value,
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr,
        } => -binary64_leaf(expressions, &expressions[*expr]),
        naga::Expression::As {
            expr,
            kind: naga::ScalarKind::Float,
            convert: Some(8),
        } => binary64_leaf(expressions, &expressions[*expr]),
        other => panic!("expected unrounded binary64 literal, got {other:?}"),
    }
}

fn assert_named_binary64(function: &naga::Function, name: &str, expected: f64) {
    let expression = named(function, name);
    let actual = binary64_leaf(&function.expressions, expression);
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{name}: wrong binary64 literal"
    );
}

fn assert_named_concrete_binary64(function: &naga::Function, name: &str, expected: f64) {
    let expression = named(function, name);
    let concrete = match expression {
        naga::Expression::Literal(naga::Literal::F64(_)) => true,
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr,
        } => matches!(
            &function.expressions[*expr],
            naga::Expression::Literal(naga::Literal::F64(_))
        ),
        _ => false,
    };
    assert!(
        concrete,
        "{name}: FFT constant is not a concrete F64 literal"
    );
    assert_named_binary64(function, name, expected);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fft_radix_constants_keep_binary64_precision() {
    let stage3 = parsed(fft::FFT_STAGE3_SHADER_F64);
    let stage3_main = &stage3.entry_points[0].function;
    assert_named_concrete_binary64(stage3_main, "sqrt3_over_2", 0.8660254037844386_f64);

    let stage5 = parsed(fft::FFT_STAGE5_SHADER_F64);
    let stage5_main = &stage5.entry_points[0].function;
    for (name, expected) in [
        ("c1", 0.30901699437494745_f64),
        ("s1", 0.9510565162951535_f64),
        ("s2", 0.5877852522924732_f64),
    ] {
        assert_named_concrete_binary64(stage5_main, name, expected);
    }
    // Naga 24 cannot negate a concrete F64 constant. Its F64 constructor
    // must still receive the unrounded abstract-float value, never F32.
    assert_named_binary64(stage5_main, "c2", -0.8090169943749473_f64);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn random_normal_clamp_is_below_one_in_binary64() {
    for (name, source) in [
        ("random-normal", creation::RANDOM_NORMAL_SHADER_F64),
        ("normrnd", creation::RANDOM_NORMRND_SHADER_F64),
    ] {
        let module = parsed(source);
        let function = module
            .functions
            .iter()
            .find_map(|(_, function)| {
                (function.name.as_deref() == Some("uniform_pair")).then_some(function)
            })
            .unwrap_or_else(|| panic!("{name}: missing uniform_pair"));
        let naga::Expression::Math {
            fun: naga::MathFunction::Min,
            arg1: Some(bound),
            ..
        } = named(function, "u2")
        else {
            panic!("{name}: u2 is not the production min clamp");
        };
        let value = binary64_leaf(&function.expressions, &function.expressions[*bound]);
        assert_eq!(
            value.to_bits(),
            0.9999999999999999_f64.to_bits(),
            "{name}: random clamp rounded to one"
        );
    }
}
