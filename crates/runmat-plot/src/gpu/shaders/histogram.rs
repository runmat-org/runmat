//! Exact unweighted histogram classification. Numerical edges are binary64
//! storage words, not rendering floats; no native F64 or INT64 instructions.
use crate::gpu::ScalarType;

pub fn source(scalar: ScalarType, workgroup_size: u32) -> String {
    let sample = match scalar {
        ScalarType::F32 => "fn sample_words(index: u32) -> vec2<u32> { return runmat_f32_to_binary64_words(samples[index]); }",
        ScalarType::F64 => "fn sample_words(index: u32) -> vec2<u32> { return vec2<u32>(samples[2u * index], samples[2u * index + 1u]); }",
    };
    format!(
        "{}\n{}\n{sample}",
        runmat_accelerate_api::wgpu_ieee_words::BINARY32_TO_BINARY64_WORDS_WGSL,
        super::compute_source(COMMON, workgroup_size)
    )
}

const COMMON: &str = r#"
const WORKGROUP_SIZE: u32 = {{WORKGROUP_SIZE}}u;
struct HistogramParams { sample_count: u32, bin_count: u32, stride: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> samples: array<u32>;
@group(0) @binding(1) var<storage, read> edges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: HistogramParams;

fn word_nan(value: vec2<u32>) -> bool {
    let magnitude = value.y & 0x7fffffffu;
    return magnitude > 0x7ff00000u || (magnitude == 0x7ff00000u && value.x != 0u);
}
fn word_less(a: vec2<u32>, b: vec2<u32>) -> bool {
    let am = a.y & 0x7fffffffu;
    let bm = b.y & 0x7fffffffu;
    // IEEE numerical comparison, not totalOrder: signed zeros compare equal.
    if am == 0u && a.x == 0u && bm == 0u && b.x == 0u { return false; }
    let an = (a.y & 0x80000000u) != 0u;
    let bn = (b.y & 0x80000000u) != 0u;
    if an != bn { return an; }
    if an { return am > bm || (am == bm && a.x > b.x); }
    return am < bm || (am == bm && a.x < b.x);
}
fn bin_index(value: vec2<u32>) -> u32 {
    // Preserve the actual CPU owner's NaN-last and clipped-outlier policy.
    if word_nan(value) { return params.bin_count - 1u; }
    var lower = 1u;
    var upper = params.bin_count;
    loop {
        if lower >= upper { break; }
        let middle = lower + (upper - lower) / 2u;
        if word_less(value, edges[middle]) { upper = middle; }
        else { lower = middle + 1u; }
    }
    return lower - 1u;
}
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var index = gid.x;
    loop {
        if index >= params.sample_count { break; }
        atomicAdd(&counts[bin_index(sample_words(index))], 1u);
        // Avoid overflow even at the closed u32 sample-count bound.
        if params.sample_count - index <= params.stride { break; }
        index = index + params.stride;
    }
}
"#;
