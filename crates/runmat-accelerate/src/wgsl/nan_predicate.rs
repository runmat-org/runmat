//! One precision-selected IEEE NaN predicate for generated floating WGSL.
//! The F64 test checks both mantissa words: a NaN may have only low-word bits.
//! Callers retain their distinct include/omit/propagate policy; only the
//! classification primitive is shared.

use super::NumericPrecision;

#[cfg(all(test, feature = "wgpu"))]
#[path = "../backend/wgpu/shaders/nan_predicate/validation_tests.rs"]
mod validation_tests;

const FLOAT_MARKER: &str = "@IEEE_NAN_PREDICATE@";
const WORD_MARKER: &str = "@IEEE_NAN_WORD_TEST@";
#[cfg(feature = "wgpu")]
const FINITE_MARKER: &str = "@IEEE_FINITE_PREDICATE@";

// The raw-word and floating-value kernels inject these same expressions.
const WORD_TEST_F32: &str = "(bits & 0x7fffffffu) > 0x7f800000u";
const WORD_TEST_F64: &str =
    "((high & 0x7fffffffu) > 0x7ff00000u) || (((high & 0x7fffffffu) == 0x7ff00000u) && low != 0u)";

fn word_test(precision: NumericPrecision) -> &'static str {
    match precision {
        NumericPrecision::F32 => WORD_TEST_F32,
        NumericPrecision::F64 => WORD_TEST_F64,
    }
}

/// Write the canonical predicate without allocating intermediate replacements.
pub(crate) fn write_body(
    writer: &mut dyn std::fmt::Write,
    precision: NumericPrecision,
) -> std::fmt::Result {
    let prefix = match precision {
        NumericPrecision::F32 => "let bits = bitcast<u32>(value); return ",
        NumericPrecision::F64 => {
            "let low = runmat_f64_low(value); let high = runmat_f64_high(value); return "
        }
    };
    writer.write_str(prefix)?;
    writer.write_str(word_test(precision))?;
    writer.write_str(";")
}

pub(crate) fn inject(source: &str, precision: NumericPrecision) -> String {
    assert_eq!(
        source.matches(FLOAT_MARKER).count(),
        1,
        "WGSL NaN predicate marker"
    );
    let body = match precision {
        NumericPrecision::F32 => "let bits = bitcast<u32>(value); return @IEEE_NAN_WORD_TEST@;",
        NumericPrecision::F64 => "let low = runmat_f64_low(value); let high = runmat_f64_high(value); return @IEEE_NAN_WORD_TEST@;",
    };
    let source = source
        .replace(FLOAT_MARKER, body)
        .replace(WORD_MARKER, word_test(precision));
    match precision {
        NumericPrecision::F32 => source,
        NumericPrecision::F64 => super::ieee_words::inject(&source),
    }
}

#[cfg(any(feature = "wgpu", test))]
pub(crate) fn inject_words(source: &str, precision: NumericPrecision) -> String {
    assert_eq!(
        source.matches(WORD_MARKER).count(),
        1,
        "WGSL raw-word NaN marker"
    );
    source.replace(WORD_MARKER, word_test(precision))
}

/// Classifies finite IEEE words before an owning kernel's separate numeric
/// range constraint. This does not replace a caller's max-magnitude policy.
#[cfg(feature = "wgpu")]
pub(crate) fn inject_finite(source: &str, precision: NumericPrecision) -> String {
    assert_eq!(
        source.matches(FINITE_MARKER).count(),
        1,
        "WGSL finite predicate marker"
    );
    let body = match precision {
        NumericPrecision::F32 => {
            "let bits = bitcast<u32>(value); return (bits & 0x7f800000u) != 0x7f800000u;"
        }
        NumericPrecision::F64 => {
            "let high = runmat_f64_high(value); return (high & 0x7ff00000u) != 0x7ff00000u;"
        }
    };
    let source = source.replace(FINITE_MARKER, body);
    match precision {
        NumericPrecision::F32 => source,
        NumericPrecision::F64 => super::ieee_words::inject(&source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_checks_f64_low_mantissa_word_and_f32_payload() {
        let f32 = inject(FLOAT_MARKER, NumericPrecision::F32);
        let f64 = inject(FLOAT_MARKER, NumericPrecision::F64);
        assert!(f32.contains(WORD_TEST_F32));
        assert!(f64.contains(WORD_TEST_F64));
        assert!(f64.contains("low != 0u"));
        assert_eq!(
            inject_words(WORD_MARKER, NumericPrecision::F32),
            WORD_TEST_F32
        );
        assert_eq!(
            inject_words(WORD_MARKER, NumericPrecision::F64),
            WORD_TEST_F64
        );
        assert!(!f32.contains("value != value"));
        assert!(!f64.contains("value != value"));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn reduction_and_scan_sources_bind_exactly_one_typed_predicate() {
        use crate::backend::wgpu::shaders::{reduction, scan};

        let f32_sources = [
            scan::CUMSUM_SHADER_F32,
            scan::CUMPROD_SHADER_F32,
            scan::CUMMIN_SHADER_F32,
            scan::CUMMAX_SHADER_F32,
            reduction::REDUCE_DIM_SHADER_F32,
        ];
        let f64_sources = [
            scan::CUMSUM_SHADER_F64,
            scan::CUMPROD_SHADER_F64,
            scan::CUMMIN_SHADER_F64,
            scan::CUMMAX_SHADER_F64,
            reduction::REDUCE_DIM_SHADER_F64,
        ];
        for (precision, sources) in [
            (NumericPrecision::F32, f32_sources),
            (NumericPrecision::F64, f64_sources),
        ] {
            let global = reduction::global_source(precision);
            assert!(!global.contains(FLOAT_MARKER));
            assert!(!global.contains(WORD_MARKER));
            assert!(global.contains("fn isNan("));
            for source in sources {
                let selected = inject(source, precision);
                assert!(!selected.contains(FLOAT_MARKER));
                assert!(!selected.contains(WORD_MARKER));
                assert!(!selected.contains("value != value"));
                assert!(!selected.contains("v != v"));
                assert!(selected.contains("fn isNan(") || selected.contains("fn is_nan_"));
            }
            let (first, second) = reduction::build_two_pass_shaders(precision, 64);
            for selected in [first, second] {
                assert!(!selected.contains(FLOAT_MARKER));
                assert!(!selected.contains(WORD_MARKER));
                assert!(selected.contains("fn isNan("));
                assert!(!selected.contains("x != x"));
            }
        }
        for input in [NumericPrecision::F32, NumericPrecision::F64] {
            let selected = reduction::ordered_mean::source(input, NumericPrecision::F32);
            assert!(!selected.contains(FLOAT_MARKER));
            assert!(!selected.contains(WORD_MARKER));
            assert!(selected.contains("fn isNan(value: f64)"));
            assert!(selected.contains("if (isNan(value))"));
            assert!(selected.contains("params.omit_nan == 0u"));
        }
        assert!(reduction::REDUCE_DIM_SHADER_F32.contains("op_any_omit"));
        assert!(reduction::REDUCE_DIM_SHADER_F64.contains("op_all_omit"));
        assert!(scan::CUMSUM_SHADER_F32.contains("omit_nan = (params.flags & 2u)"));
        assert!(scan::CUMMIN_SHADER_F64.contains("omit_nan = (params.flags & 2u)"));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn raw_word_map_and_mask_share_the_same_precision_selected_test() {
        use crate::backend::wgpu::shaders::nan;

        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            let expected = word_test(precision);
            for source in [nan::map_source(precision), nan::mask_source(precision)] {
                assert!(source.contains(expected));
                assert!(!source.contains(WORD_MARKER));
                assert!(source.contains("data: array<u32>"));
                assert!(!source.contains("value != value"));
            }
            let mapped = nan::map_source(precision);
            assert!(
                mapped.contains("select(bits, 0u, nan)") || mapped.contains("select(low, 0u, nan)")
            );
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn finite_word_test_preserves_separate_range_policy() {
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            let body = inject_finite(FINITE_MARKER, precision);
            assert!(!body.contains(FINITE_MARKER));
            assert!(!body.contains("value == value"));
            match precision {
                NumericPrecision::F32 => assert!(body.contains("0x7f800000u")),
                NumericPrecision::F64 => assert!(body.contains("0x7ff00000u")),
            }
            let symbols = crate::backend::wgpu::shaders::comms::modulate_constellation_shader(
                precision, 4, 64,
            );
            let bits = crate::backend::wgpu::shaders::comms::modulate_bits_constellation_shader(
                precision, 4, 64,
            );
            let sub = crate::backend::wgpu::shaders::sub2ind::build_sub2ind_shader(
                precision,
                &[3],
                &[1],
                &[0],
                64,
            );
            let ind = crate::backend::wgpu::shaders::ind2sub::build_ind2sub_shader(
                precision,
                &[3],
                &[1],
                3,
                64,
            );
            for source in [symbols, bits, sub, ind] {
                assert!(!source.contains(FINITE_MARKER));
                assert!(!source.contains("x == x"));
                assert!(source.contains("fn isFinite("));
                assert!(source.contains("abs(x) <"));
            }
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn interpolation_and_symmetry_use_nan_words_without_changing_branch_policy() {
        use crate::backend::wgpu::shaders::{interp1, symmetry};

        for (precision, interpolation, symmetric) in [
            (
                NumericPrecision::F32,
                interp1::INTERP1_SHADER_F32,
                symmetry::SYMMETRY_SHADER_F32,
            ),
            (
                NumericPrecision::F64,
                interp1::INTERP1_SHADER_F64,
                symmetry::SYMMETRY_SHADER_F64,
            ),
        ] {
            let interpolation = inject(interpolation, precision);
            assert!(interpolation.contains("if isNan(query)"));
            assert!(interpolation.contains("out_of_range_"));
            assert!(!interpolation.contains("query != query"));
            let symmetric = inject(symmetric, precision);
            assert!(symmetric.contains("fn finite_"));
            assert!(symmetric.contains("is_nan_"));
            assert!(!symmetric.contains("value != value"));
        }
    }
}
