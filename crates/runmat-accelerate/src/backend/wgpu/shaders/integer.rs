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

pub fn extrema_dim_shader(scalar_type: &str, workgroup_size: u32) -> String {
    INTEGER_EXTREMA_DIM_SHADER
        .replace("$SCALAR", scalar_type)
        .replace("@WG@", &workgroup_size.to_string())
}

pub fn reduce_dim_shader(workgroup_size: u32) -> String {
    INTEGER_REDUCE_DIM_SHADER.replace("@WG@", &workgroup_size.to_string())
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
    let overflow = select(
        (a < 0) == (b < 0) && (r < 0) != (a < 0),
        (a < 0) != (b < 0) && (r < 0) != (a < 0),
        params.op == 1u,
    );
    if (!overflow) { return r; }
    return select(maxv, minv, a < 0);
}
fn signed_narrow(a: i32, b: i32, minv: i32, maxv: i32) -> i32 {
    let r = select(a + b, a - b, params.op == 1u);
    if (r < minv) { return minv; }
    if (r > maxv) { return maxv; }
    return r;
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
fn power_unsigned32(base: u32, exponent: u32, maxv: u32) -> u32 {
    var result = 1u;
    var factor = base;
    var power = exponent;
    loop {
        if (power == 0u) { break; }
        if ((power & 1u) != 0u) { result = multiply_unsigned32(result, factor, maxv); }
        power = power >> 1u;
        if (power != 0u) { factor = multiply_unsigned32(factor, factor, maxv); }
    }
    return result;
}
fn power_signed32(base: i32, exponent: i32, min_bits: u32, max_bits: u32) -> u32 {
    if (exponent < 0) {
        let magnitude = 0u - bitcast<u32>(exponent);
        if (base == 0) { return max_bits; }
        if (base == 1) { return 1u; }
        if (base == -1) { return select(0xffffffffu, 1u, (magnitude & 1u) == 0u); }
        if (base == 2 && magnitude == 1u) { return 1u; }
        if (base == -2 && magnitude == 1u) { return 0xffffffffu; }
        return 0u;
    }
    var result = 1u;
    var factor = bitcast<u32>(base);
    var power = bitcast<u32>(exponent);
    loop {
        if (power == 0u) { break; }
        if ((power & 1u) != 0u) { result = multiply_signed32(bitcast<i32>(result), bitcast<i32>(factor), min_bits, max_bits); }
        power = power >> 1u;
        if (power != 0u) { factor = multiply_signed32(bitcast<i32>(factor), bitcast<i32>(factor), min_bits, max_bits); }
    }
    return result;
}
fn divide_unsigned32(a: u32, b: u32, maxv: u32) -> u32 {
    if (b == 0u) {
        return select(0u, maxv, a != 0u);
    }
    var quotient = a / b;
    let remainder = a % b;
    if (remainder >= b - remainder) {
        quotient = min(quotient + 1u, maxv);
    }
    return quotient;
}
fn divide_signed32(a: i32, b: i32, min_bits: u32, max_bits: u32) -> u32 {
    if (b == 0) {
        if (a < 0) { return min_bits; }
        if (a > 0) { return max_bits; }
        return 0u;
    }
    let negative = (a < 0) != (b < 0);
    let a_bits = bitcast<u32>(a);
    let b_bits = bitcast<u32>(b);
    let a_abs = select(a_bits, 0u - a_bits, a < 0);
    let b_abs = select(b_bits, 0u - b_bits, b < 0);
    var quotient = a_abs / b_abs;
    let remainder = a_abs % b_abs;
    if (remainder >= b_abs - remainder) { quotient = quotient + 1u; }
    if (negative) {
        if (quotient > min_bits) { return min_bits; }
        return 0u - quotient;
    }
    if (quotient > max_bits) { return max_bits; }
    return quotient;
}
fn remainder_unsigned32(a: u32, b: u32, is_modulus: bool) -> u32 {
    if (b == 0u) { return select(0u, a, is_modulus); }
    return a % b;
}
fn remainder_signed32(a: i32, b: i32, is_modulus: bool) -> u32 {
    if (b == 0) { return select(0u, bitcast<u32>(a), is_modulus); }
    var remainder = a % b;
    if (is_modulus && remainder != 0 && (remainder < 0) != (b < 0)) { remainder = remainder + b; }
    return bitcast<u32>(remainder);
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
fn ge64(a0: u32, a1: u32, b0: u32, b1: u32) -> bool {
    return a1 > b1 || (a1 == b1 && a0 >= b0);
}
fn sub64(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let borrow = select(0u, 1u, a0 < b0);
    return vec2<u32>(a0 - b0, a1 - b1 - borrow);
}
fn increment64(lo: u32, hi: u32) -> vec2<u32> {
    let out_lo = lo + 1u;
    return vec2<u32>(out_lo, hi + select(0u, 1u, out_lo == 0u));
}
fn divmod64(n0: u32, n1: u32, d0: u32, d1: u32) -> vec4<u32> {
    var q0 = 0u;
    var q1 = 0u;
    var r0 = 0u;
    var r1 = 0u;
    var bit = 64u;
    loop {
        if (bit == 0u) { break; }
        bit = bit - 1u;
        var input_bit = 0u;
        if (bit >= 32u) { input_bit = (n1 >> (bit - 32u)) & 1u; }
        else { input_bit = (n0 >> bit) & 1u; }
        let carry = r0 >> 31u;
        r0 = (r0 << 1u) | input_bit;
        r1 = (r1 << 1u) | carry;
        if (ge64(r0, r1, d0, d1)) {
            let remainder = sub64(r0, r1, d0, d1);
            r0 = remainder.x;
            r1 = remainder.y;
            if (bit >= 32u) { q1 = q1 | (1u << (bit - 32u)); }
            else { q0 = q0 | (1u << bit); }
        }
    }
    return vec4<u32>(q0, q1, r0, r1);
}
fn should_round64(r0: u32, r1: u32, d0: u32, d1: u32) -> bool {
    let complement = sub64(d0, d1, r0, r1);
    return ge64(r0, r1, complement.x, complement.y);
}
fn write_divide64(lane: u32, a0: u32, a1: u32, b0: u32, b1: u32, is_signed: bool) {
    if (b0 == 0u && b1 == 0u) {
        if (!is_signed) {
            let nonzero = a0 != 0u || a1 != 0u;
            Out.data[lane] = select(0u, 0xffffffffu, nonzero);
            Out.data[lane + 1u] = select(0u, 0xffffffffu, nonzero);
            return;
        }
        let negative = bitcast<i32>(a1) < 0;
        let nonzero = a0 != 0u || a1 != 0u;
        if (negative) { Out.data[lane] = 0u; Out.data[lane + 1u] = 0x80000000u; }
        else if (nonzero) { Out.data[lane] = 0xffffffffu; Out.data[lane + 1u] = 0x7fffffffu; }
        else { Out.data[lane] = 0u; Out.data[lane + 1u] = 0u; }
        return;
    }
    let a_negative = is_signed && bitcast<i32>(a1) < 0;
    let b_negative = is_signed && bitcast<i32>(b1) < 0;
    let negative = a_negative != b_negative;
    let magnitude_a = select(vec2<u32>(a0, a1), negate64(a0, a1), a_negative);
    let magnitude_b = select(vec2<u32>(b0, b1), negate64(b0, b1), b_negative);
    var division = divmod64(magnitude_a.x, magnitude_a.y, magnitude_b.x, magnitude_b.y);
    if (should_round64(division.z, division.w, magnitude_b.x, magnitude_b.y)) {
        let incremented = increment64(division.x, division.y);
        division.x = incremented.x;
        division.y = incremented.y;
    }
    if (!is_signed) {
        Out.data[lane] = division.x;
        Out.data[lane + 1u] = division.y;
        return;
    }
    if (negative) {
        if (division.y > 0x80000000u || (division.y == 0x80000000u && division.x != 0u)) {
            Out.data[lane] = 0u; Out.data[lane + 1u] = 0x80000000u;
        } else {
            let result = negate64(division.x, division.y);
            Out.data[lane] = result.x; Out.data[lane + 1u] = result.y;
        }
    } else if (division.y > 0x7fffffffu) {
        Out.data[lane] = 0xffffffffu; Out.data[lane + 1u] = 0x7fffffffu;
    } else {
        Out.data[lane] = division.x; Out.data[lane + 1u] = division.y;
    }
}
fn write_remainder64(lane: u32, a0: u32, a1: u32, b0: u32, b1: u32, is_signed: bool, is_modulus: bool) {
    if (b0 == 0u && b1 == 0u) {
        Out.data[lane] = select(0u, a0, is_modulus);
        Out.data[lane + 1u] = select(0u, a1, is_modulus);
        return;
    }
    if (!is_signed) {
        let division = divmod64(a0, a1, b0, b1);
        Out.data[lane] = division.z; Out.data[lane + 1u] = division.w;
        return;
    }
    let a_negative = bitcast<i32>(a1) < 0;
    let b_negative = bitcast<i32>(b1) < 0;
    let abs_a = select(vec2<u32>(a0, a1), negate64(a0, a1), a_negative);
    let abs_b = select(vec2<u32>(b0, b1), negate64(b0, b1), b_negative);
    let division = divmod64(abs_a.x, abs_a.y, abs_b.x, abs_b.y);
    var remainder = vec2<u32>(division.z, division.w);
    if (remainder.x != 0u || remainder.y != 0u) {
        var negative = a_negative;
        if (is_modulus && a_negative != b_negative) {
            remainder = sub64(abs_b.x, abs_b.y, remainder.x, remainder.y);
            negative = b_negative;
        }
        if (negative) { remainder = negate64(remainder.x, remainder.y); }
    }
    Out.data[lane] = remainder.x; Out.data[lane + 1u] = remainder.y;
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
fn multiply_unsigned64_saturating(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let product = multiply_unsigned64(a0, a1, b0, b1);
    if (product.z != 0u || product.w != 0u) { return vec2<u32>(0xffffffffu, 0xffffffffu); }
    return vec2<u32>(product.x, product.y);
}
fn power_unsigned64(base0: u32, base1: u32, exponent0: u32, exponent1: u32) -> vec2<u32> {
    var result = vec2<u32>(1u, 0u);
    var factor = vec2<u32>(base0, base1);
    var power = vec2<u32>(exponent0, exponent1);
    loop {
        if (power.x == 0u && power.y == 0u) { break; }
        if ((power.x & 1u) != 0u) { result = multiply_unsigned64_saturating(result.x, result.y, factor.x, factor.y); }
        power.x = (power.x >> 1u) | (power.y << 31u);
        power.y = power.y >> 1u;
        if (power.x != 0u || power.y != 0u) { factor = multiply_unsigned64_saturating(factor.x, factor.y, factor.x, factor.y); }
    }
    return result;
}
fn power_signed64(base0: u32, base1: u32, exponent0: u32, exponent1: u32) -> vec2<u32> {
    if (bitcast<i32>(exponent1) < 0) {
        let magnitude = negate64(exponent0, exponent1);
        if (base0 == 0u && base1 == 0u) { return vec2<u32>(0xffffffffu, 0x7fffffffu); }
        if (base0 == 1u && base1 == 0u) { return vec2<u32>(1u, 0u); }
        if (base0 == 0xffffffffu && base1 == 0xffffffffu) { return select(vec2<u32>(0xffffffffu, 0xffffffffu), vec2<u32>(1u, 0u), (magnitude.x & 1u) == 0u); }
        if (base0 == 2u && base1 == 0u && magnitude.x == 1u && magnitude.y == 0u) { return vec2<u32>(1u, 0u); }
        if (base0 == 0xfffffffeu && base1 == 0xffffffffu && magnitude.x == 1u && magnitude.y == 0u) { return vec2<u32>(0xffffffffu, 0xffffffffu); }
        return vec2<u32>(0u, 0u);
    }
    var result = vec2<u32>(1u, 0u);
    var factor = vec2<u32>(base0, base1);
    var power = vec2<u32>(exponent0, exponent1);
    loop {
        if (power.x == 0u && power.y == 0u) { break; }
        if ((power.x & 1u) != 0u) { result = multiply_signed64(result.x, result.y, factor.x, factor.y); }
        power.x = (power.x >> 1u) | (power.y << 31u);
        power.y = power.y >> 1u;
        if (power.x != 0u || power.y != 0u) { factor = multiply_signed64(factor.x, factor.y, factor.x, factor.y); }
    }
    return result;
}
fn write_multiply64(lane: u32, a0: u32, a1: u32, b0: u32, b1: u32, is_signed: bool) {
    if (is_signed) {
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
fn write64(lane: u32, a0: u32, a1: u32, b0: u32, b1: u32, is_signed: bool) {
    let subtract = params.op == 1u;
    var lo: u32;
    var hi: u32;
    var unsigned_overflow = false;
    if (subtract) {
        let borrow = select(0u, 1u, a0 < b0);
        lo = a0 - b0;
        hi = a1 - b1 - borrow;
        unsigned_overflow = a1 < b1 || (a1 == b1 && borrow != 0u);
    } else {
        let carry = select(0u, 1u, a0 + b0 < a0);
        lo = a0 + b0;
        hi = a1 + b1 + carry;
        unsigned_overflow = hi < a1 || (carry != 0u && hi == a1);
    }
    if (!is_signed) {
        if (unsigned_overflow) {
            if (subtract) {
                Out.data[lane] = 0u;
                Out.data[lane + 1u] = 0u;
            } else {
                Out.data[lane] = 0xffffffffu;
                Out.data[lane + 1u] = 0xffffffffu;
            }
            return;
        }
        Out.data[lane] = lo;
        Out.data[lane + 1u] = hi;
        return;
    }
    let a_negative = bitcast<i32>(a1) < 0;
    let b_negative = bitcast<i32>(b1) < 0;
    let result_negative = bitcast<i32>(hi) < 0;
    let overflow = select(
        a_negative == b_negative && result_negative != a_negative,
        a_negative != b_negative && result_negative != a_negative,
        subtract,
    );
    if (overflow) {
        if (a_negative) {
            Out.data[lane] = 0u;
            Out.data[lane + 1u] = 0x80000000u;
        } else {
            Out.data[lane] = 0xffffffffu;
            Out.data[lane + 1u] = 0x7fffffffu;
        }
        return;
    }
    Out.data[lane] = lo;
    Out.data[lane + 1u] = hi;
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
        case 0u: { if (params.op >= 5u) { Out.data[lane] = remainder_signed32(sx8(a), sx8(b), params.op == 6u); } else if (params.op == 2u) { Out.data[lane] = multiply_signed32(sx8(a), sx8(b), 0x80u, 0x7fu); } else if (params.op == 3u) { Out.data[lane] = divide_signed32(sx8(a), sx8(b), 0x80u, 0x7fu); } else if (params.op == 4u) { Out.data[lane] = power_signed32(sx8(a), sx8(b), 0x80u, 0x7fu); } else { Out.data[lane] = bitcast<u32>(signed_narrow(sx8(a), sx8(b), -128, 127)); } }
        case 1u: { if (params.op >= 5u) { Out.data[lane] = remainder_signed32(sx16(a), sx16(b), params.op == 6u); } else if (params.op == 2u) { Out.data[lane] = multiply_signed32(sx16(a), sx16(b), 0x8000u, 0x7fffu); } else if (params.op == 3u) { Out.data[lane] = divide_signed32(sx16(a), sx16(b), 0x8000u, 0x7fffu); } else if (params.op == 4u) { Out.data[lane] = power_signed32(sx16(a), sx16(b), 0x8000u, 0x7fffu); } else { Out.data[lane] = bitcast<u32>(signed_narrow(sx16(a), sx16(b), -32768, 32767)); } }
        case 2u: { if (params.op >= 5u) { Out.data[lane] = remainder_signed32(bitcast<i32>(a), bitcast<i32>(b), params.op == 6u); } else if (params.op == 2u) { Out.data[lane] = multiply_signed32(bitcast<i32>(a), bitcast<i32>(b), 0x80000000u, 0x7fffffffu); } else if (params.op == 3u) { Out.data[lane] = divide_signed32(bitcast<i32>(a), bitcast<i32>(b), 0x80000000u, 0x7fffffffu); } else if (params.op == 4u) { Out.data[lane] = power_signed32(bitcast<i32>(a), bitcast<i32>(b), 0x80000000u, 0x7fffffffu); } else { Out.data[lane] = bitcast<u32>(signed32(bitcast<i32>(a), bitcast<i32>(b), -2147483648, 2147483647)); } }
        case 3u: { if (params.op >= 5u) { write_remainder64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], true, params.op == 6u); } else if (params.op == 2u) { write_multiply64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], true); } else if (params.op == 3u) { write_divide64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], true); } else if (params.op == 4u) { let value = power_signed64(a, A.data[a_lane + 1u], b, B.data[b_lane + 1u]); Out.data[lane] = value.x; Out.data[lane + 1u] = value.y; } else { write64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], true); } }
        case 4u: { if (params.op >= 5u) { Out.data[lane] = remainder_unsigned32(a & 0xffu, b & 0xffu, params.op == 6u); } else if (params.op == 2u) { Out.data[lane] = multiply_unsigned32(a & 0xffu, b & 0xffu, 0xffu); } else if (params.op == 3u) { Out.data[lane] = divide_unsigned32(a & 0xffu, b & 0xffu, 0xffu); } else if (params.op == 4u) { Out.data[lane] = power_unsigned32(a & 0xffu, b & 0xffu, 0xffu); } else { Out.data[lane] = min(unsigned32(a & 0xffu, b & 0xffu, 0xffu), 0xffu); } }
        case 5u: { if (params.op >= 5u) { Out.data[lane] = remainder_unsigned32(a & 0xffffu, b & 0xffffu, params.op == 6u); } else if (params.op == 2u) { Out.data[lane] = multiply_unsigned32(a & 0xffffu, b & 0xffffu, 0xffffu); } else if (params.op == 3u) { Out.data[lane] = divide_unsigned32(a & 0xffffu, b & 0xffffu, 0xffffu); } else if (params.op == 4u) { Out.data[lane] = power_unsigned32(a & 0xffffu, b & 0xffffu, 0xffffu); } else { Out.data[lane] = min(unsigned32(a & 0xffffu, b & 0xffffu, 0xffffu), 0xffffu); } }
        case 6u: { if (params.op >= 5u) { Out.data[lane] = remainder_unsigned32(a, b, params.op == 6u); } else if (params.op == 2u) { Out.data[lane] = multiply_unsigned32(a, b, 0xffffffffu); } else if (params.op == 3u) { Out.data[lane] = divide_unsigned32(a, b, 0xffffffffu); } else if (params.op == 4u) { Out.data[lane] = power_unsigned32(a, b, 0xffffffffu); } else { Out.data[lane] = unsigned32(a, b, 0xffffffffu); } }
        case 7u: { if (params.op >= 5u) { write_remainder64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], false, params.op == 6u); } else if (params.op == 2u) { write_multiply64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], false); } else if (params.op == 3u) { write_divide64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], false); } else if (params.op == 4u) { let value = power_unsigned64(a, A.data[a_lane + 1u], b, B.data[b_lane + 1u]); Out.data[lane] = value.x; Out.data[lane + 1u] = value.y; } else { write64(lane, a, A.data[a_lane + 1u], b, B.data[b_lane + 1u], false); } }
        default: {}
    }
}
"#;

const INTEGER_REDUCE_DIM_SHADER: &str = r#"
struct Words { data: array<u32> };
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
alias PackedArray = array<PackedValue, 128>;
struct Params {
    rank: u32,
    kept_count: u32,
    reduce_count: u32,
    op: u32,
    rows: u32,
    cols: u32,
    integer_type: u32,
    slice_offset: u32,
    kept_sizes: PackedArray,
    reduce_sizes: PackedArray,
    kept_strides: PackedArray,
    reduce_strides: PackedArray,
};

@group(0) @binding(0) var<storage, read> InBuf: Words;
@group(0) @binding(1) var<storage, read_write> OutBuf: Words;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile_lo: array<u32, @WG@u>;
var<workgroup> tile_hi: array<u32, @WG@u>;

fn sx8_reduce(x: u32) -> i32 { return bitcast<i32>(x << 24u) >> 24; }
fn sx16_reduce(x: u32) -> i32 { return bitcast<i32>(x << 16u) >> 16; }

fn add_with_carry_reduce(a: u32, b: u32, carry: u32) -> vec2<u32> {
    let first = a + b;
    let second = first + carry;
    return vec2<u32>(second, select(0u, 1u, first < a) + select(0u, 1u, second < first));
}

fn mul_hi_u32_reduce(a: u32, b: u32) -> u32 {
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

fn multiply_unsigned64_reduce(a0: u32, a1: u32, b0: u32, b1: u32) -> vec4<u32> {
    let p00_lo = a0 * b0;
    let p00_hi = mul_hi_u32_reduce(a0, b0);
    let p01_lo = a0 * b1;
    let p01_hi = mul_hi_u32_reduce(a0, b1);
    let p10_lo = a1 * b0;
    let p10_hi = mul_hi_u32_reduce(a1, b0);
    let p11_lo = a1 * b1;
    let p11_hi = mul_hi_u32_reduce(a1, b1);
    let word1 = add_with_carry_reduce(p00_hi, p01_lo, 0u);
    let word1_final = add_with_carry_reduce(word1.x, p10_lo, 0u);
    let word2 = add_with_carry_reduce(p01_hi, p10_hi, word1.y + word1_final.y);
    let word2_final = add_with_carry_reduce(word2.x, p11_lo, 0u);
    return vec4<u32>(p00_lo, word1_final.x, word2_final.x, p11_hi + word2.y + word2_final.y);
}

fn negate64_reduce(lo: u32, hi: u32) -> vec2<u32> {
    let neg_lo = 0u - lo;
    return vec2<u32>(neg_lo, ~hi + select(0u, 1u, neg_lo == 0u));
}

fn add64_reduce(a0: u32, a1: u32, b0: u32, b1: u32, is_signed: bool) -> vec2<u32> {
    let carry = select(0u, 1u, a0 + b0 < a0);
    let lo = a0 + b0;
    let hi = a1 + b1 + carry;
    let unsigned_overflow = hi < a1 || (carry != 0u && hi == a1);
    if (!is_signed) {
        return select(vec2<u32>(lo, hi), vec2<u32>(0xffffffffu, 0xffffffffu), unsigned_overflow);
    }
    let a_negative = bitcast<i32>(a1) < 0;
    let b_negative = bitcast<i32>(b1) < 0;
    let result_negative = bitcast<i32>(hi) < 0;
    if (a_negative == b_negative && result_negative != a_negative) {
        return select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), a_negative);
    }
    return vec2<u32>(lo, hi);
}

fn multiply_signed64_reduce(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let a_negative = bitcast<i32>(a1) < 0;
    let b_negative = bitcast<i32>(b1) < 0;
    let negative = a_negative != b_negative;
    let abs_a = select(vec2<u32>(a0, a1), negate64_reduce(a0, a1), a_negative);
    let abs_b = select(vec2<u32>(b0, b1), negate64_reduce(b0, b1), b_negative);
    let product = multiply_unsigned64_reduce(abs_a.x, abs_a.y, abs_b.x, abs_b.y);
    let limit = select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), negative);
    if (product.z != 0u || product.w != 0u || product.y > limit.y || (product.y == limit.y && product.x > limit.x)) {
        return limit;
    }
    return select(vec2<u32>(product.x, product.y), negate64_reduce(product.x, product.y), negative);
}

fn multiply_unsigned64_saturating_reduce(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let product = multiply_unsigned64_reduce(a0, a1, b0, b1);
    if (product.z != 0u || product.w != 0u) { return vec2<u32>(0xffffffffu, 0xffffffffu); }
    return vec2<u32>(product.x, product.y);
}

fn signed32_add_reduce(a: i32, b: i32, minv: i32, maxv: i32) -> i32 {
    let r = a + b;
    if ((a < 0) == (b < 0) && (r < 0) != (a < 0)) {
        return select(maxv, minv, a < 0);
    }
    return r;
}

fn signed_narrow_add_reduce(a: i32, b: i32, minv: i32, maxv: i32) -> i32 {
    let r = a + b;
    if (r < minv) { return minv; }
    if (r > maxv) { return maxv; }
    return r;
}

fn unsigned_add_reduce(a: u32, b: u32, maxv: u32) -> u32 {
    let r = a + b;
    return select(r, maxv, r < a || r > maxv);
}

fn multiply_unsigned32_reduce(a: u32, b: u32, maxv: u32) -> u32 {
    let lo = a * b;
    let hi = mul_hi_u32_reduce(a, b);
    return select(lo, maxv, hi != 0u || lo > maxv);
}

fn multiply_signed32_reduce(a: i32, b: i32, min_bits: u32, max_bits: u32) -> u32 {
    let negative = (a < 0) != (b < 0);
    let a_bits = bitcast<u32>(a);
    let b_bits = bitcast<u32>(b);
    let a_abs = select(a_bits, 0u - a_bits, a < 0);
    let b_abs = select(b_bits, 0u - b_bits, b < 0);
    let lo = a_abs * b_abs;
    let hi = mul_hi_u32_reduce(a_abs, b_abs);
    let limit = select(max_bits, min_bits, negative);
    if (hi != 0u || lo > limit) { return select(max_bits, min_bits, negative); }
    return select(lo, 0u - lo, negative);
}

fn combine_reduce(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let is_product = params.op == 1u;
    switch params.integer_type {
        case 0u: {
            let v = select(
                bitcast<u32>(signed_narrow_add_reduce(sx8_reduce(a0), sx8_reduce(b0), -128, 127)),
                multiply_signed32_reduce(sx8_reduce(a0), sx8_reduce(b0), 0x80u, 0x7fu),
                is_product,
            );
            return vec2<u32>(v, 0u);
        }
        case 1u: {
            let v = select(
                bitcast<u32>(signed_narrow_add_reduce(sx16_reduce(a0), sx16_reduce(b0), -32768, 32767)),
                multiply_signed32_reduce(sx16_reduce(a0), sx16_reduce(b0), 0x8000u, 0x7fffu),
                is_product,
            );
            return vec2<u32>(v, 0u);
        }
        case 2u: {
            let v = select(
                bitcast<u32>(signed32_add_reduce(bitcast<i32>(a0), bitcast<i32>(b0), -2147483648, 2147483647)),
                multiply_signed32_reduce(bitcast<i32>(a0), bitcast<i32>(b0), 0x80000000u, 0x7fffffffu),
                is_product,
            );
            return vec2<u32>(v, 0u);
        }
        case 3u: {
            return select(
                add64_reduce(a0, a1, b0, b1, true),
                multiply_signed64_reduce(a0, a1, b0, b1),
                is_product,
            );
        }
        case 4u: {
            return vec2<u32>(select(unsigned_add_reduce(a0 & 0xffu, b0 & 0xffu, 0xffu), multiply_unsigned32_reduce(a0 & 0xffu, b0 & 0xffu, 0xffu), is_product), 0u);
        }
        case 5u: {
            return vec2<u32>(select(unsigned_add_reduce(a0 & 0xffffu, b0 & 0xffffu, 0xffffu), multiply_unsigned32_reduce(a0 & 0xffffu, b0 & 0xffffu, 0xffffu), is_product), 0u);
        }
        case 6u: {
            return vec2<u32>(select(unsigned_add_reduce(a0, b0, 0xffffffffu), multiply_unsigned32_reduce(a0, b0, 0xffffffffu), is_product), 0u);
        }
        case 7u: {
            return select(
                add64_reduce(a0, a1, b0, b1, false),
                multiply_unsigned64_saturating_reduce(a0, a1, b0, b1),
                is_product,
            );
        }
        default: { return vec2<u32>(0u, 0u); }
    }
}

fn map_col_to_base(col: u32) -> u32 {
    var rem = col;
    var base = 0u;
    var j = 0u;
    loop {
        if (j >= params.kept_count) { break; }
        let size = params.kept_sizes[j].value;
        if (size != 0u) {
            let coord = rem % size;
            rem = rem / size;
            base = base + coord * params.kept_strides[j].value;
        }
        j = j + 1u;
    }
    return base;
}

fn map_row_offset(row: u32) -> u32 {
    var rem = row;
    var offset = 0u;
    var j = 0u;
    loop {
        if (j >= params.reduce_count) { break; }
        let size = params.reduce_sizes[j].value;
        if (size != 0u) {
            let coord = rem % size;
            rem = rem / size;
            offset = offset + coord * params.reduce_strides[j].value;
        }
        j = j + 1u;
    }
    return offset;
}

fn identity_value() -> vec2<u32> {
    if (params.op == 1u) {
        return vec2<u32>(1u, 0u);
    }
    return vec2<u32>(0u, 0u);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let slice = params.slice_offset + wid.x;
    if (slice >= params.cols) { return; }
    let base = map_col_to_base(slice);
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    var acc = identity_value();
    var i = lid.x;
    loop {
        if (i >= params.rows) { break; }
        let in_lane = (base + map_row_offset(i)) * lanes;
        var high_word = 0u;
        if (lanes == 2u) {
            high_word = InBuf.data[in_lane + 1u];
        }
        acc = combine_reduce(acc.x, acc.y, InBuf.data[in_lane], high_word);
        i = i + @WG@u;
    }
    tile_lo[lid.x] = acc.x;
    tile_hi[lid.x] = acc.y;
    workgroupBarrier();
    var offset = @WG@u / 2u;
    loop {
        if (offset == 0u) { break; }
        if (lid.x < offset) {
            let combined = combine_reduce(tile_lo[lid.x], tile_hi[lid.x], tile_lo[lid.x + offset], tile_hi[lid.x + offset]);
            tile_lo[lid.x] = combined.x;
            tile_hi[lid.x] = combined.y;
        }
        workgroupBarrier();
        offset = offset / 2u;
    }
    if (lid.x == 0u) {
        let out_lane = slice * lanes;
        OutBuf.data[out_lane] = tile_lo[0u];
        if (lanes == 2u) {
            OutBuf.data[out_lane + 1u] = tile_hi[0u];
        }
    }
}
"#;

const INTEGER_EXTREMA_DIM_SHADER: &str = r#"
struct Words { data: array<u32> };
struct Indices { data: array<$SCALAR> };

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    select_min: u32,
    integer_type: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Words;
@group(0) @binding(1) var<storage, read_write> OutVals: Words;
@group(0) @binding(2) var<storage, read_write> OutIdx: Indices;
@group(0) @binding(3) var<uniform> params: Params;

fn signed8_extrema(word: u32) -> i32 { return bitcast<i32>(word << 24u) >> 24; }
fn signed16_extrema(word: u32) -> i32 { return bitcast<i32>(word << 16u) >> 16; }

// Returns negative when the left value is smaller, positive when larger, and
// zero when equal. The high word is compared first for 64-bit values.
fn compare_extrema(left_lane: u32, right_lane: u32) -> i32 {
    let left_low = InBuf.data[left_lane];
    let right_low = InBuf.data[right_lane];
    switch params.integer_type {
        case 0u: {
            let left = signed8_extrema(left_low); let right = signed8_extrema(right_low);
            if (left < right) { return -1; } if (left > right) { return 1; }
        }
        case 1u: {
            let left = signed16_extrema(left_low); let right = signed16_extrema(right_low);
            if (left < right) { return -1; } if (left > right) { return 1; }
        }
        case 2u: {
            let left = bitcast<i32>(left_low); let right = bitcast<i32>(right_low);
            if (left < right) { return -1; } if (left > right) { return 1; }
        }
        case 3u: {
            let left_high = bitcast<i32>(InBuf.data[left_lane + 1u]);
            let right_high = bitcast<i32>(InBuf.data[right_lane + 1u]);
            if (left_high < right_high) { return -1; } if (left_high > right_high) { return 1; }
            if (left_low < right_low) { return -1; } if (left_low > right_low) { return 1; }
        }
        case 4u: {
            let left = left_low & 0xffu; let right = right_low & 0xffu;
            if (left < right) { return -1; } if (left > right) { return 1; }
        }
        case 5u: {
            let left = left_low & 0xffffu; let right = right_low & 0xffffu;
            if (left < right) { return -1; } if (left > right) { return 1; }
        }
        case 6u: {
            if (left_low < right_low) { return -1; } if (left_low > right_low) { return 1; }
        }
        case 7u: {
            let left_high = InBuf.data[left_lane + 1u]; let right_high = InBuf.data[right_lane + 1u];
            if (left_high < right_high) { return -1; } if (left_high > right_high) { return 1; }
            if (left_low < right_low) { return -1; } if (left_low > right_low) { return 1; }
        }
        default: {}
    }
    return 0;
}

fn write_extrema(out_index: u32, input_index: u32) {
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    let source_lane = input_index * lanes;
    let target_lane = out_index * lanes;
    OutVals.data[target_lane] = InBuf.data[source_lane];
    if (lanes == 2u) { OutVals.data[target_lane + 1u] = InBuf.data[source_lane + 1u]; }
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_index = gid.x;
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    if (params.dim == 0u) {
        if (out_index >= params.cols || params.rows == 0u) { return; }
        var best = out_index * params.rows;
        var row = 1u;
        loop {
            if (row >= params.rows) { break; }
            let candidate = row + out_index * params.rows;
            let order = compare_extrema(candidate * lanes, best * lanes);
            if (select(order > 0, order < 0, params.select_min == 1u)) { best = candidate; }
            row = row + 1u;
        }
        write_extrema(out_index, best);
        OutIdx.data[out_index] = $SCALAR((best % params.rows) + 1u);
    } else {
        if (out_index >= params.rows || params.cols == 0u) { return; }
        var best = out_index;
        var col = 1u;
        loop {
            if (col >= params.cols) { break; }
            let candidate = out_index + col * params.rows;
            let order = compare_extrema(candidate * lanes, best * lanes);
            if (select(order > 0, order < 0, params.select_min == 1u)) { best = candidate; }
            col = col + 1u;
        }
        write_extrema(out_index, best);
        OutIdx.data[out_index] = $SCALAR((best / params.rows) + 1u);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::{arithmetic_shader, comparison_shader, extrema_dim_shader, minmax_shader};

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

    #[test]
    fn extrema_dim_shader_substitutes_precision_and_workgroup_size() {
        let shader = extrema_dim_shader("f64", 128);
        assert!(shader.contains("array<f64>"));
        assert!(shader.contains("@workgroup_size(128)"));
        assert!(!shader.contains("$SCALAR"));
        assert!(!shader.contains("@WG@"));
    }
}
