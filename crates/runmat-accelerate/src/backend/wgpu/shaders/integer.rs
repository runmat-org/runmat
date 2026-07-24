pub fn comparison_shader(scalar_type: &str, workgroup_size: u32) -> String {
    INTEGER_COMPARISON_SHADER
        .replace("$SCALAR", scalar_type)
        .replace("@WG@", &workgroup_size.to_string())
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

#[cfg(test)]
mod tests {
    use super::comparison_shader;

    #[test]
    fn comparison_shader_substitutes_precision_and_workgroup_size() {
        let shader = comparison_shader("f64", 128);
        assert!(shader.contains("array<f64>"));
        assert!(shader.contains("@workgroup_size(128)"));
        assert!(!shader.contains("$SCALAR"));
        assert!(!shader.contains("@WG@"));
    }
}
