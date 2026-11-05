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
    let uniform_function =
        format!("fn uniform_from_word(word: vec2<u32>) -> {dtype} {{\n    {conversion}\n}}");
    TEMPLATE
        .replace("@DTYPE@", dtype)
        .replace("@UNIFORM_FUNCTION@", &uniform_function)
        .replace("@SAMPLE@", sample)
        .replace("@BATCH_FIELDS@", "")
}

/// Fixed-work native integer sampler. The typed request owner admits only a
/// power-of-two span, so the mask is an unbiased exact reduction and this
/// shader has no unbounded rejection loop hidden inside a dispatch.
pub(crate) fn fixed_integer_shader(
    class: runmat_accelerate_api::NumericElementType,
) -> Option<String> {
    use runmat_accelerate_api::NumericElementType;
    // Integer sampling uses the paired-word counter directly; only floating
    // distributions need the uniform conversion helper.
    let (dtype, sample) = match class {
        NumericElementType::F32 => (
            "f32",
            "f32(bitcast<i32>(params.integer_lower + (words[0].x & params.integer_mask)))",
        ),
        NumericElementType::F64 => (
            "f64",
            "f64(bitcast<i32>(params.integer_lower + (words[0].x & params.integer_mask)))",
        ),
        NumericElementType::I32 | NumericElementType::U32 => (
            "u32",
            "params.integer_lower + (words[0].x & params.integer_mask)",
        ),
        _ => return None,
    };
    Some(
        TEMPLATE
            .replace("@DTYPE@", dtype)
            .replace("@UNIFORM_FUNCTION@", "")
            .replace("@SAMPLE@", sample)
            .replace("@BATCH_FIELDS@", ""),
    )
}

/// One batch of at most eight rejection attempts per unresolved item. The
/// returned count is copied to host only after physical queue settlement;
/// the coordinator may then await cooperation before admitting another batch.
pub(crate) fn integer_batch_shader(
    class: runmat_accelerate_api::NumericElementType,
) -> Option<String> {
    use runmat_accelerate_api::NumericElementType;
    let (dtype, write_value) = match class {
        NumericElementType::F32 => ("f32", "Out.data[global_idx] = f32(bitcast<i32>(value.x));"),
        NumericElementType::F64 => ("f64", "Out.data[global_idx] = f64(bitcast<i32>(value.x));"),
        NumericElementType::I32 | NumericElementType::U32 => {
            ("u32", "Out.data[global_idx] = value.x;")
        }
        _ => return None,
    };
    let prefix = TEMPLATE.split_once("@compute")?.0;
    let batch_fields = "span0: u32, span1: u32, accept_limit0: u32, accept_limit1: u32, lower0: u32, lower1: u32, attempt0: u32, attempt1: u32, attempts: u32, initial: u32, _batch_pad0: u32, _batch_pad1: u32,";
    let mut source = prefix
        .replace("@DTYPE@", dtype)
        .replace("@UNIFORM_FUNCTION@", "")
        .replace("@BATCH_FIELDS@", batch_fields);
    source.push_str(include_str!("integer_divmod64.wgsl"));
    source.push_str(
        r#"
struct Pending { data: array<u32>, };
struct Remaining { value: atomic<u32>, };
@group(0) @binding(2) var<storage, read_write> Accepted: Pending;
@group(0) @binding(3) var<storage, read_write> RemainingCount: Remaining;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.chunk) { return; }
    let global_idx = params.offset + gid.x;
    if (params.initial == 0u && Accepted.data[global_idx] != 0u) { return; }
    var attempt_index = 0u;
    loop {
        if (attempt_index >= params.attempts) { break; }
        let attempt = add64(vec2<u32>(params.attempt0, params.attempt1), vec2<u32>(attempt_index, 0u));
        let words = threefry(counter_for_item_attempt(global_idx, attempt)).lanes;
        let word = words[0];
        if (!ge64(word.x, word.y, params.accept_limit0, params.accept_limit1)) {
            var remainder = vec2<u32>(word.x, word.y);
            if (params.span0 != 0u || params.span1 != 0u) {
                let division = divmod64(word.x, word.y, params.span0, params.span1);
                remainder = division.zw;
            }
            let value = add64(vec2<u32>(params.lower0, params.lower1), remainder);
            @WRITE_VALUE@
            Accepted.data[global_idx] = 1u;
            return;
        }
        attempt_index = attempt_index + 1u;
    }
    Accepted.data[global_idx] = 0u;
    atomicAdd(&RemainingCount.value, 1u);
}
"#,
    );
    Some(source.replace("@WRITE_VALUE@", write_value))
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
                assert!(source.contains("fn uniform_from_word("));
                assert!(!source.contains("@UNIFORM_FUNCTION@"));
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

    #[test]
    fn fixed_integer_variants_keep_exact_counter_and_mask() {
        for class in [
            runmat_accelerate_api::NumericElementType::F32,
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::NumericElementType::I32,
            runmat_accelerate_api::NumericElementType::U32,
        ] {
            let source = fixed_integer_shader(class).expect("supported fixed-work class");
            assert!(source.contains("counter_for_item(global_idx)"));
            assert!(source.contains("words[0].x & params.integer_mask"));
            assert!(source.contains("round < 20u"));
            assert!(!source.contains("uniform_from_word"));
            assert!(!source.contains("@UNIFORM_FUNCTION@"));
            validate_generated(
                &format!("random-threefry-integer-{class:?}"),
                &source,
                64,
                8,
                if class == runmat_accelerate_api::NumericElementType::F64 {
                    RequiredCaps::Float64
                } else {
                    RequiredCaps::None
                },
            );
        }
    }

    #[test]
    fn bounded_rejection_variants_share_the_integer_remainder_kernel() {
        for class in [
            runmat_accelerate_api::NumericElementType::F32,
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::NumericElementType::I32,
            runmat_accelerate_api::NumericElementType::U32,
        ] {
            let source = integer_batch_shader(class).expect("supported native batch class");
            assert!(source.contains("counter_for_item_attempt(global_idx, attempt)"));
            assert!(source.contains("fn divmod64("));
            assert!(source.contains("atomicAdd(&RemainingCount.value, 1u)"));
            assert!(!source.contains("uniform_from_word"));
            assert!(!source.contains("@UNIFORM_FUNCTION@"));
            for lanes in [32, 64] {
                validate_generated(
                    &format!("random-threefry-integer-batch-{class:?}-{lanes}"),
                    &source,
                    lanes,
                    8,
                    if class == runmat_accelerate_api::NumericElementType::F64 {
                        RequiredCaps::Float64
                    } else {
                        RequiredCaps::None
                    },
                );
            }
        }
    }
}
