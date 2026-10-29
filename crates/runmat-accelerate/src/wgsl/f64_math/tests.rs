use super::{prepare, PreparationError};

#[test]
fn typed_calls_preserve_comments_and_single_precision() {
    let source = "alias Double = f64; fn f(x: Double) -> Double { return exp /* call */ (log(x)); } fn single(x: f32) -> f32 { return exp(x); }";
    let prepared = prepare(source).unwrap();
    assert!(prepared
        .source()
        .contains("runmat_math64_exp /* call */ (runmat_math64_log(x))"));
    assert!(prepared.source().contains("return exp(x);"));
    assert!(prepared.requires_f64_math());
    assert_eq!(
        prepare(prepared.source()).unwrap().source(),
        prepared.source()
    );
    assert!(prepare(prepared.source()).unwrap().requires_f64_math());
}

#[test]
fn typed_vector_wrappers_cover_every_width() {
    for width in 2..=4 {
        let source =
            format!("fn f(x: vec{width}<f64>) -> vec{width}<f64> {{ return exp(log(x)); }}");
        let prepared = prepare(&source).unwrap();
        assert!(prepared.source().contains(&format!(
            "runmat_math64_exp_vec{width}(runmat_math64_log_vec{width}(x))"
        )));
        assert!(prepared
            .source()
            .contains(&format!("runmat_math64_log(a[{}])", width - 1)));
    }
}

#[test]
fn canonical_ieee_helpers_are_shared_by_parsed_declaration() {
    let source = crate::wgsl::ieee_words::inject("fn f(x: f64) -> f64 { return exp(x); }");
    let prepared = prepare(&source).unwrap();
    assert_eq!(prepared.source().matches("fn runmat_f64_high(").count(), 1);
    assert_eq!(prepared.source().matches("fn runmat_f64_low(").count(), 1);
    assert_eq!(
        prepared
            .source()
            .matches("fn runmat_f64_from_words(")
            .count(),
        1
    );
}

#[test]
fn incompatible_reserved_declaration_is_a_structured_error() {
    let source =
        "fn runmat_f64_low(x: f64) -> u32 { return 0u; } fn f(x: f64) -> f64 { return exp(x); }";
    assert!(matches!(
        prepare(source),
        Err(PreparationError::ReservedHelperCollision(_))
    ));
}

#[test]
fn permitted_intrinsics_and_single_precision_source_are_unchanged() {
    for source in [
        "fn f(x: f64) -> f64 { return sqrt(x) + floor(x); }",
        "fn f(x: f32) -> f32 { return exp(log(x)); }",
    ] {
        assert_eq!(prepare(source).unwrap().source(), source);
    }
}

#[test]
fn malformed_shader_is_not_deferred_to_device_creation() {
    assert!(matches!(
        prepare("not WGSL"),
        Err(PreparationError::Parse(_))
    ));
}

#[test]
fn every_restricted_intrinsic_and_shape_is_legalized() {
    use super::intrinsic::{Intrinsic, Shape};
    let operations = [
        Intrinsic::Radians,
        Intrinsic::Degrees,
        Intrinsic::Sin,
        Intrinsic::Cos,
        Intrinsic::Tan,
        Intrinsic::Asin,
        Intrinsic::Acos,
        Intrinsic::Atan,
        Intrinsic::Atan2,
        Intrinsic::Sinh,
        Intrinsic::Cosh,
        Intrinsic::Tanh,
        Intrinsic::Asinh,
        Intrinsic::Acosh,
        Intrinsic::Atanh,
        Intrinsic::Exp,
        Intrinsic::Exp2,
        Intrinsic::Log,
        Intrinsic::Log2,
        Intrinsic::Pow,
    ];
    let mut failures = Vec::new();
    for intrinsic in operations {
        for shape in [Shape::Scalar, Shape::Vec2, Shape::Vec3, Shape::Vec4] {
            let ty = shape
                .vector_width()
                .map_or_else(|| "f64".to_owned(), |width| format!("vec{width}<f64>"));
            let arguments = if intrinsic.arguments() == 2 {
                "x, x"
            } else {
                "x"
            };
            let source = format!(
                "fn f(x: {ty}) -> {ty} {{ return {}({arguments}); }}",
                intrinsic.spelling()
            );
            let prepared = match prepare(&source) {
                Ok(prepared) => prepared,
                Err(error) => {
                    failures.push(format!("{intrinsic:?}/{shape:?}: {error}"));
                    continue;
                }
            };
            assert!(prepared
                .source()
                .contains(&format!("return {}({arguments});", intrinsic.helper(shape))));
            assert_eq!(
                prepare(prepared.source()).unwrap().source(),
                prepared.source()
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
