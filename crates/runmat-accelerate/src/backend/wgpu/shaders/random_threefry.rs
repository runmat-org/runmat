//! Paired-word Threefry shader source for the request-bound GPU child stream.
//! Uniform and real/complex normal variants share the same counter and rounds.

use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::random::GpuRandomDistribution;

const TEMPLATE: &str = include_str!("random_threefry.wgsl");

pub(crate) fn shader(precision: NumericPrecision, distribution: GpuRandomDistribution) -> String {
    shader_with_complex_normal(precision, distribution, false)
}

pub(crate) fn complex_normal_shader(precision: NumericPrecision) -> String {
    shader_with_complex_normal(precision, GpuRandomDistribution::StandardNormal, true)
}

fn shader_with_complex_normal(
    precision: NumericPrecision,
    distribution: GpuRandomDistribution,
    complex_normal: bool,
) -> String {
    let (dtype, conversion) = match precision {
        NumericPrecision::F64 => (
            "f64",
            "let high = word.y >> 11u; let low = (word.y << 21u) | (word.x >> 11u); return (f64(high) * 4294967296.0 + f64(low)) * (1.0 / 9007199254740992.0);",
        ),
        NumericPrecision::F32 => (
            "f32",
            "return f32(word.y >> 8u) * (1.0 / 16777216.0);",
        ),
    };
    let sample = match distribution {
        GpuRandomDistribution::Uniform => "uniform_from_word(words[0])",
        GpuRandomDistribution::StandardNormal if complex_normal => {
            "(sqrt(-2.0 * log(1.0 - uniform_from_word(words[0]))) * cos(6.283185307179586 * uniform_from_word(words[1]))) * 0.70710678118654752440"
        }
        GpuRandomDistribution::StandardNormal => {
            "sqrt(-2.0 * log(1.0 - uniform_from_word(words[0]))) * cos(6.283185307179586 * uniform_from_word(words[1]))"
        }
    };
    TEMPLATE
        .replace("@DTYPE@", dtype)
        .replace("@CONVERSION@", conversion)
        .replace("@SAMPLE@", sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

    #[test]
    fn every_variant_uses_the_same_paired_word_rounds() {
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            for distribution in [
                GpuRandomDistribution::Uniform,
                GpuRandomDistribution::StandardNormal,
            ] {
                let source = shader(precision, distribution);
                assert!(source.contains("round < 20u"));
                assert!(source.contains("counter_for_item(global_idx)"));
                assert!(!source.contains("@DTYPE@"));
                assert!(!source.contains("@SAMPLE@"));
                assert!(!source.contains("@CONVERSION@"));
                validate_generated(
                    &format!("random-threefry-{precision:?}-{distribution:?}"),
                    &source,
                    64,
                    8,
                    match precision {
                        NumericPrecision::F32 => RequiredCaps::None,
                        NumericPrecision::F64 => RequiredCaps::Float64,
                    },
                );
            }
        }
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            let source = complex_normal_shader(precision);
            assert!(source.contains("0.70710678118654752440"));
            assert!(source.contains("counter_for_item(global_idx)"));
            validate_generated(
                &format!("random-threefry-complex-normal-{precision:?}"),
                &source,
                64,
                8,
                if precision == NumericPrecision::F64 {
                    RequiredCaps::Float64
                } else {
                    RequiredCaps::None
                },
            );
        }
    }
}
