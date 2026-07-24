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

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
    integer_type: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

fn compare_words(index: u32) -> i32 {
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    let lane = index * lanes;
    let a_low = A.data[lane];
    let b_low = B.data[lane];

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
            let a_high = bitcast<i32>(A.data[lane + 1u]);
            let b_high = bitcast<i32>(B.data[lane + 1u]);
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
            let a_high = A.data[lane + 1u];
            let b_high = B.data[lane + 1u];
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

struct Params {
    len: u32,
    select_min: u32,
    offset: u32,
    total: u32,
    integer_type: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

fn compare_words_minmax(index: u32) -> i32 {
    let lanes = select(1u, 2u, params.integer_type == 3u || params.integer_type == 7u);
    let lane = index * lanes;
    let a_low = A.data[lane];
    let b_low = B.data[lane];

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
            let a_high = bitcast<i32>(A.data[lane + 1u]);
            let b_high = bitcast<i32>(B.data[lane + 1u]);
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
            let a_high = A.data[lane + 1u];
            let b_high = B.data[lane + 1u];
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
    let lane = index * lanes;
    let ordering = compare_words_minmax(index);
    let choose_a = select(ordering >= 0, ordering <= 0, params.select_min != 0u);
    Out.data[lane] = select(B.data[lane], A.data[lane], choose_a);
    if (lanes == 2u) {
        Out.data[lane + 1u] = select(B.data[lane + 1u], A.data[lane + 1u], choose_a);
    }
}
"#;

const INTEGER_ARITHMETIC_SHADER: &str = r#"
struct Words { data: array<u32> };
struct Params { len: u32, op: u32, offset: u32, total: u32, integer_type: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
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
    let lane = index * lanes;
    let a = A.data[lane]; let b = B.data[lane];
    switch params.integer_type {
        case 0u: { Out.data[lane] = bitcast<u32>(signed32(sx8(a), sx8(b), -128, 127)); }
        case 1u: { Out.data[lane] = bitcast<u32>(signed32(sx16(a), sx16(b), -32768, 32767)); }
        case 2u: { Out.data[lane] = bitcast<u32>(signed32(bitcast<i32>(a), bitcast<i32>(b), -2147483648, 2147483647)); }
        case 3u: { write64(lane, a, A.data[lane + 1u], b, B.data[lane + 1u], true); }
        case 4u: { Out.data[lane] = min(unsigned32(a & 0xffu, b & 0xffu, 0xffu), 0xffu); }
        case 5u: { Out.data[lane] = min(unsigned32(a & 0xffffu, b & 0xffffu, 0xffffu), 0xffffu); }
        case 6u: { Out.data[lane] = unsigned32(a, b, 0xffffffffu); }
        case 7u: { write64(lane, a, A.data[lane + 1u], b, B.data[lane + 1u], false); }
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
