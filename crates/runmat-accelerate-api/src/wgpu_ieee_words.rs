//! Shared exact storage-word widening, without native floating instructions.
//! Low/high are the existing little-endian binary64 storage-word ABI. This
//! helper neither requires native F64/INT64 nor certifies preceding float math.

pub const BINARY32_TO_BINARY64_WORDS_WGSL: &str = r#"
fn runmat_f32_to_binary64_words(bits: u32) -> vec2<u32> {
    let sign = bits & 0x80000000u;
    let encoded = (bits >> 23u) & 255u;
    var fraction = bits & 0x7fffffu;
    if encoded == 255u {
        return vec2<u32>(0u, sign | 0x7ff00000u | select(0u, 0x80000u, fraction != 0u));
    }
    if encoded == 0u && fraction == 0u { return vec2<u32>(0u, sign); }
    var exponent = i32(encoded) + 896;
    if encoded == 0u {
        let highest = 31u - countLeadingZeros(fraction);
        exponent = i32(highest) + 874;
        fraction = (fraction << (23u - highest)) & 0x7fffffu;
    }
    return vec2<u32>(fraction << 29u, sign | (u32(exponent) << 20u) | (fraction >> 3u));
}
"#;
