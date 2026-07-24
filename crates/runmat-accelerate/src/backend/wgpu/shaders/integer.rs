pub fn comparison_shader(scalar_type: &str, workgroup_size: u32) -> String {
    INTEGER_COMPARISON_SHADER
        .replace("$SCALAR", scalar_type)
        .replace("@WG@", &workgroup_size.to_string())
}

pub fn minmax_shader(workgroup_size: u32) -> String {
    INTEGER_MINMAX_SHADER.replace("@WG@", &workgroup_size.to_string())
}

pub fn arithmetic_shader(workgroup_size: u32) -> String {
    INTEGER_ARITHMETIC_SHADER.replace("@WG@", &workgroup_size.to_string())
}

const INTEGER_COMPARISON_SHADER: &str = r#"
struct Words { data: array<u32> };
struct Output { data: array<$SCALAR> };
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
alias PackedArray = array<PackedValue, 128>;

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
    integer_type: u32,
    rank: u32,
    _pad0: u32,
    _pad1: u32,
    out_shape: PackedArray,
    a_shape: PackedArray,
    b_shape: PackedArray,
    a_strides: PackedArray,
    b_strides: PackedArray,
};

@group(0) @binding(0) var<storage, read> A: Words;
@group(0) @binding(1) var<storage, read> B: Words;
@group(0) @binding(2) var<storage, read_write> Out: Output;
@group(0) @binding(3) var<uniform> params: Params;

fn signed8(word: u32) -> i32 {
    return bitcast<i32>(word << 24u) >> 24;
}

fn signed16(word: u32) -> i32 {
    return bitcast<i32>(word << 16u) >> 16;
}

fn source_indices(index: u32) -> vec2<u32> {
    if (params.rank == 0u) { return vec2<u32>(index, index); }
    var tmp = index;
    var a_index = 0u;
    var b_index = 0u;
    var d = 0u;
    loop {
        if (d >= params.rank) { break; }
        let coordinate = tmp % params.out_shape[d].value;
        tmp = tmp / params.out_shape[d].value;
        let a_coordinate = select(coordinate, 0u, params.a_shape[d].value == 1u);
        let b_coordinate = select(coordinate, 0u, params.b_shape[d].value == 1u);
        a_index = a_index + a_coordinate * params.a_strides[d].value;
        b_index = b_index + b_coordinate * params.b_strides[d].value;
        d = d + 1u;
    }
    return vec2<u32>(a_index, b_index);
}

fn compare_words(index: u32) -> i32 {
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    let sources = source_indices(index);
    let a_lane = sources.x * lanes;
    let b_lane = sources.y * lanes;
    let a_low = A.data[a_lane];
    let b_low = B.data[b_lane];

    switch params.integer_type {
        case 0u: {
            let a = signed8(a_low);
            let b = signed8(b_low);
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 1u: {
            let a = signed16(a_low);
            let b = signed16(b_low);
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 2u: {
            let a = bitcast<i32>(a_low);
            let b = bitcast<i32>(b_low);
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 3u: {
            let a_high = bitcast<i32>(A.data[a_lane + 1u]);
            let b_high = bitcast<i32>(B.data[b_lane + 1u]);
            if (a_high < b_high) { return -1; }
            if (a_high > b_high) { return 1; }
            if (a_low < b_low) { return -1; }
            if (a_low > b_low) { return 1; }
        }
        case 4u: {
            let a = a_low & 0xffu;
            let b = b_low & 0xffu;
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 5u: {
            let a = a_low & 0xffffu;
            let b = b_low & 0xffffu;
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 6u: {
            if (a_low < b_low) { return -1; }
            if (a_low > b_low) { return 1; }
        }
        case 7u: {
            let a_high = A.data[a_lane + 1u];
            let b_high = B.data[b_lane + 1u];
            if (a_high < b_high) { return -1; }
            if (a_high > b_high) { return 1; }
            if (a_low < b_low) { return -1; }
            if (a_low > b_low) { return 1; }
        }
        default: { return 0; }
    }
    return 0;
}

fn compare_result(ordering: i32) -> bool {
    switch params.op {
        case 0u: { return ordering == 0; }
        case 1u: { return ordering != 0; }
        case 2u: { return ordering < 0; }
        case 3u: { return ordering <= 0; }
        case 4u: { return ordering > 0; }
        case 5u: { return ordering >= 0; }
        default: { return false; }
    }
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= params.len) { return; }
    let index = params.offset + local;
    if (index >= params.total) { return; }
    Out.data[index] = select(0.0, 1.0, compare_result(compare_words(index)));
}
"#;

const INTEGER_MINMAX_SHADER: &str = r#"
struct Words { data: array<u32> };
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
alias PackedArray = array<PackedValue, 128>;

struct Params {
    len: u32,
    select_min: u32,
    offset: u32,
    total: u32,
    integer_type: u32,
    rank: u32,
    _pad0: u32,
    _pad1: u32,
    out_shape: PackedArray,
    a_shape: PackedArray,
    b_shape: PackedArray,
    a_strides: PackedArray,
    b_strides: PackedArray,
};

@group(0) @binding(0) var<storage, read> A: Words;
@group(0) @binding(1) var<storage, read> B: Words;
@group(0) @binding(2) var<storage, read_write> Out: Words;
@group(0) @binding(3) var<uniform> params: Params;

fn signed8_minmax(word: u32) -> i32 {
    return bitcast<i32>(word << 24u) >> 24;
}

fn signed16_minmax(word: u32) -> i32 {
    return bitcast<i32>(word << 16u) >> 16;
}

fn source_indices_minmax(index: u32) -> vec2<u32> {
    if (params.rank == 0u) { return vec2<u32>(index, index); }
    var tmp = index;
    var a_index = 0u;
    var b_index = 0u;
    var d = 0u;
    loop {
        if (d >= params.rank) { break; }
        let coordinate = tmp % params.out_shape[d].value;
        tmp = tmp / params.out_shape[d].value;
        let a_coordinate = select(coordinate, 0u, params.a_shape[d].value == 1u);
        let b_coordinate = select(coordinate, 0u, params.b_shape[d].value == 1u);
        a_index = a_index + a_coordinate * params.a_strides[d].value;
        b_index = b_index + b_coordinate * params.b_strides[d].value;
        d = d + 1u;
    }
    return vec2<u32>(a_index, b_index);
}

fn compare_words_minmax(index: u32) -> i32 {
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    let sources = source_indices_minmax(index);
    let a_lane = sources.x * lanes;
    let b_lane = sources.y * lanes;
    let a_low = A.data[a_lane];
    let b_low = B.data[b_lane];

    switch params.integer_type {
        case 0u: {
            let a = signed8_minmax(a_low);
            let b = signed8_minmax(b_low);
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 1u: {
            let a = signed16_minmax(a_low);
            let b = signed16_minmax(b_low);
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 2u: {
            let a = bitcast<i32>(a_low);
            let b = bitcast<i32>(b_low);
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 3u: {
            let a_high = bitcast<i32>(A.data[a_lane + 1u]);
            let b_high = bitcast<i32>(B.data[b_lane + 1u]);
            if (a_high < b_high) { return -1; }
            if (a_high > b_high) { return 1; }
            if (a_low < b_low) { return -1; }
            if (a_low > b_low) { return 1; }
        }
        case 4u: {
            let a = a_low & 0xffu;
            let b = b_low & 0xffu;
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 5u: {
            let a = a_low & 0xffffu;
            let b = b_low & 0xffffu;
            if (a < b) { return -1; }
            if (a > b) { return 1; }
        }
        case 6u: {
            if (a_low < b_low) { return -1; }
            if (a_low > b_low) { return 1; }
        }
        case 7u: {
            let a_high = A.data[a_lane + 1u];
            let b_high = B.data[b_lane + 1u];
            if (a_high < b_high) { return -1; }
            if (a_high > b_high) { return 1; }
            if (a_low < b_low) { return -1; }
            if (a_low > b_low) { return 1; }
        }
        default: { return 0; }
    }
    return 0;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= params.len) { return; }
    let index = params.offset + local;
    if (index >= params.total) { return; }
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    let sources = source_indices_minmax(index);
    let lane = index * lanes;
    let a_lane = sources.x * lanes;
    let b_lane = sources.y * lanes;
    let ordering = compare_words_minmax(index);
    let choose_a = select(ordering >= 0, ordering <= 0, params.select_min != 0u);
    Out.data[lane] = select(B.data[b_lane], A.data[a_lane], choose_a);
    if (lanes == 2u) {
        Out.data[lane + 1u] = select(B.data[b_lane + 1u], A.data[a_lane + 1u], choose_a);
    }
}
"#;

const INTEGER_ARITHMETIC_SHADER: &str = r#"
struct Words { data: array<u32> };
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
alias PackedArray = array<PackedValue, 128>;
struct Params { len: u32, op: u32, offset: u32, total: u32, integer_type: u32, rank: u32, _pad0: u32, _pad1: u32, out_shape: PackedArray, a_shape: PackedArray, b_shape: PackedArray, a_strides: PackedArray, b_strides: PackedArray };
@group(0) @binding(0) var<storage, read> A: Words;
@group(0) @binding(1) var<storage, read> B: Words;
@group(0) @binding(2) var<storage, read_write> Out: Words;
@group(0) @binding(3) var<uniform> params: Params;

fn sx8(x: u32) -> i32 { return bitcast<i32>(x << 24u) >> 24; }
fn sx16(x: u32) -> i32 { return bitcast<i32>(x << 16u) >> 16; }
fn signed32(a: i32, b: i32, minv: i32, maxv: i32) -> i32 {
    let r = select(a + b, a - b, params.op == 1u);
    let overflow = select((a < 0) != (b < 0) && (r < 0) != (a < 0), (a < 0) == (b < 0) && (r < 0) != (a < 0), params.op == 1u);
    if (!overflow) { return r; }
    return select(maxv, minv, a < 0);
}
fn unsigned32(a: u32, b: u32, maxv: u32) -> u32 {
    if (params.op == 1u) { return select(a - b, 0u, a < b); }
    let r = a + b;
    return select(r, maxv, r < a);
}
fn mul_hi_u32(a: u32, b: u32) -> u32 {
    let a_hi = a >> 16u;
    let a_lo = a & 0xffffu;
    let b_hi = b >> 16u;
    let b_lo = b & 0xffffu;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let mid = (p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu);
    return p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
}
fn multiply_unsigned32(a: u32, b: u32, maxv: u32) -> u32 {
    let lo = a * b;
    let hi = mul_hi_u32(a, b);
    return select(lo, maxv, hi != 0u || lo > maxv);
}
fn multiply_signed32(a: i32, b: i32, min_bits: u32, max_bits: u32) -> u32 {
    let negative = (a < 0) != (b < 0);
    let a_bits = bitcast<u32>(a);
    let b_bits = bitcast<u32>(b);
    let a_abs = select(a_bits, 0u - a_bits, a < 0);
    let b_abs = select(b_bits, 0u - b_bits, b < 0);
    let lo = a_abs * b_abs;
    let hi = mul_hi_u32(a_abs, b_abs);
    let limit = select(max_bits, min_bits, negative);
    if (hi != 0u || lo > limit) { return select(max_bits, min_bits, negative); }
    return select(lo, 0u - lo, negative);
}
fn add_with_carry(a: u32, b: u32, carry: u32) -> vec2<u32> {
    let first = a + b;
    let second = first + carry;
    return vec2<u32>(second, select(0u, 1u, first < a) + select(0u, 1u, second < first));
}
fn multiply_unsigned64(a0: u32, a1: u32, b0: u32, b1: u32) -> vec4<u32> {
    let p00_lo = a0 * b0;
    let p00_hi = mul_hi_u32(a0, b0);
    let p01_lo = a0 * b1;
    let p01_hi = mul_hi_u32(a0, b1);
    let p10_lo = a1 * b0;
    let p10_hi = mul_hi_u32(a1, b0);
    let p11_lo = a1 * b1;
    let p11_hi = mul_hi_u32(a1, b1);
    let word1 = add_with_carry(p00_hi, p01_lo, 0u);
    let word1_final = add_with_carry(word1.x, p10_lo, 0u);
    let word2 = add_with_carry(p01_hi, p10_hi, word1.y + word1_final.y);
    let word2_final = add_with_carry(word2.x, p11_lo, 0u);
    return vec4<u32>(p00_lo, word1_final.x, word2_final.x, p11_hi + word2.y + word2_final.y);
}
fn negate64(lo: u32, hi: u32) -> vec2<u32> {
    let neg_lo = 0u - lo;
    return vec2<u32>(neg_lo, ~hi + select(0u, 1u, neg_lo == 0u));
}
fn multiply_signed64(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let a_negative = bitcast<i32>(a1) < 0;
    let b_negative = bitcast<i32>(b1) < 0;
    let negative = a_negative != b_negative;
    let abs_a = select(vec2<u32>(a0, a1), negate64(a0, a1), a_negative);
    let abs_b = select(vec2<u32>(b0, b1), negate64(b0, b1), b_negative);
    let product = multiply_unsigned64(abs_a.x, abs_a.y, abs_b.x, abs_b.y);
    let limit = select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), negative);
    if (product.z != 0u || product.w != 0u || product.y > limit.y || (product.y == limit.y && product.x > limit.x)) {
        return limit;
    }
    return select(vec2<u32>(product.x, product.y), negate64(product.x, product.y), negative);
}
fn write_multiply64(lane: u32, a0: u32, a1: u32, b0: u32, b1: u32, signed: bool) {
    if (signed) {
        let result = multiply_signed64(a0, a1, b0, b1);
        Out.data[lane] = result.x; Out.data[lane + 1u] = result.y;
        return;
    }
    let product = multiply_unsigned64(a0, a1, b0, b1);
    if (product.z != 0u || product.w != 0u) {
        Out.data[lane] = 0xffffffffu; Out.data[lane + 1u] = 0xffffffffu;
        return;
    }
    Out.data[lane] = product.x; Out.data[lane + 1u] = product.y;
}
fn write64(lane: u32, a0: u32, a1: u32, b0: u32, b1: u32, signed: bool) {
    let sub = params.op == 1u;
    let lo = select(a0 + b0, a0 - b0, sub);
    let carry = select(select(0u, 1u, lo > a0), select(0u, 1u, a0 < b0), sub);
    let hi = select(a1 + b1 + carry, a1 - b1 - carry, sub);
    if (!signed) {
        let overflow = select(hi < a1 || (carry != 0u && hi == a1), a1 < b1 || (a1 == b1 && a0 < b0), sub);
        if (overflow) { Out.data[lane] = select(0xffffffffu, 0u, sub); Out.data[lane + 1u] = select(0xffffffffu, 0u, sub); return; }
        Out.data[lane] = lo; Out.data[lane + 1u] = hi; return;
    }
    let sa = bitcast<i32>(a1) < 0;
    let sb = bitcast<i32>(b1) < 0;
    let sr = bitcast<i32>(hi) < 0;
    let overflow = select(sa != sb && sr != sa, sa == sb && sr != sa, sub);
    if (overflow) { let negative = sa; Out.data[lane] = select(0xffffffffu, 0u, negative); Out.data[lane + 1u] = select(0x7fffffffu, 0x80000000u, negative); return; }
    Out.data[lane] = lo; Out.data[lane + 1u] = hi;
}
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.len) { return; }
    let index = params.offset + gid.x;
    if (index >= params.total) { return; }
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    var a_index = index;
    var b_index = index;
    if (params.rank != 0u) {
        var tmp = index;
        var d: u32 = 0u;
        a_index = 0u;
        b_index = 0u;
        loop {
            if (d >= params.rank) { break; }
            let dim = params.out_shape[d].value;
            let coordinate = tmp % dim;
            tmp = tmp / dim;
            let a_coordinate = select(coordinate, 0u, params.a_shape[d].value == 1u);
            let b_coordinate = select(coordinate, 0u, params.b_shape[d].value == 1u);
            a_index = a_index + a_coordinate * params.a_strides[d].value;
            b_index = b_index + b_coordinate * params.b_strides[d].value;
            d = d + 1u;
        }
    }
    let lane = index * lanes;
    let a_lane = a_index * lanes;
    let b_lane = b_index * lanes;
    let a = A.data[a_lane]; let b = B.data[b_lane];
    switch params.integer_type {
        case 0u: { if (params.op == 2u) { Out.data[lane] = multiply_signed32(sx8(a), sx8(b), 0x80u, 0x7fu); } else { Out.data[lane] = bitcast<u32>(signed32(sx8(a), sx8(b), -128, 127)); } }
        case 1u: { if (params.op == 2u) { Out.data[lane] = multiply_signed32(sx16(a), sx16(b), 0x8000u, 0x7fffu); } else { Out.data[lane] = bitcast<u32>(signed32(sx16(a), sx16(b), -32768, 32767)); } }
        case 2u: { if (params.op == 2u) { Out.data[lane] = multiply_signed32(bitcast<i32>(a), bitcast<i32>(b), 0x80000000u, 0x7fffffffu); } else { Out.data[lane] = bitcast<u32>(signed32(bitcast<i32>(a), bitcast<i32>(b), -2147483648, 2147483647)); } }
        case 3u: { if (params.op == 2u) { write_multiply64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], true); } else { write64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], true); } }
        case 4u: { if (params.op == 2u) { Out.data[lane] = multiply_unsigned32(a & 0xffu, b & 0xffu, 0xffu); } else { Out.data[lane] = min(unsigned32(a & 0xffu, b & 0xffu, 0xffu), 0xffu); } }
        case 5u: { if (params.op == 2u) { Out.data[lane] = multiply_unsigned32(a & 0xffffu, b & 0xffffu, 0xffffu); } else { Out.data[lane] = min(unsigned32(a & 0xffffu, b & 0xffffu, 0xffffu), 0xffffu); } }
        case 6u: { if (params.op == 2u) { Out.data[lane] = multiply_unsigned32(a, b, 0xffffffffu); } else { Out.data[lane] = unsigned32(a, b, 0xffffffffu); } }
        case 7u: { if (params.op == 2u) { write_multiply64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], false); } else { write64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], false); } }
        default: {}
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::{arithmetic_shader, comparison_shader, minmax_shader};

    #[test]
    fn comparison_shader_substitutes_precision_and_workgroup_size() {
        let shader = comparison_shader("f64", 128);
        assert!(shader.contains("array<f64>"));
        assert!(shader.contains("@workgroup_size(128)"));
        assert!(!shader.contains("$SCALAR"));
        assert!(!shader.contains("@WG@"));
    }

    #[test]
    fn minmax_shader_substitutes_workgroup_size() {
        let shader = minmax_shader(128);
        assert!(shader.contains("@workgroup_size(128)"));
        assert!(!shader.contains("@WG@"));
    }

    #[test]
    fn arithmetic_shader_substitutes_workgroup_size() {
        let shader = arithmetic_shader(128);
        assert!(shader.contains("@workgroup_size(128)"));
        assert!(!shader.contains("@WG@"));
    }
}
