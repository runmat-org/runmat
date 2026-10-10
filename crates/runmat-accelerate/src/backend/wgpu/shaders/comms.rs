use crate::backend::wgpu::types::NumericPrecision;

pub(crate) fn modulate_constellation_shader(
    precision: NumericPrecision,
    order: usize,
    workgroup_size: u32,
) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let max_val = match precision {
        NumericPrecision::F64 => "1.7976931348623157e308",
        NumericPrecision::F32 => "3.4028234663852886e38",
    };
    let source = format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct ErrorState {{
    state: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}};

struct Params {{
    len: u32,
}};

@group(0) @binding(0) var<storage, read> Symbols: Tensor;
@group(0) @binding(1) var<storage, read> Constellation: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<storage, read_write> Error: ErrorState;
@group(0) @binding(4) var<uniform> params: Params;

const ORDER: u32 = {order}u;
const MAX_FINITE: {ty} = {ty}({max_val});

fn isFinite(value: {ty}) -> bool {{ @IEEE_FINITE_PREDICATE@ }}

fn isfinite_scalar(x: {ty}) -> bool {{
    return isFinite(x) && (abs(x) < MAX_FINITE);
}}

fn set_error(code: u32, index: u32) {{
    let packed_index = min(index, 0x3ffffffeu);
    let packed = (code << 30u) | packed_index;
    atomicMin(&Error.state, packed);
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= params.len {{
        return;
    }}
    let raw = Symbols.data[idx];
    if !isfinite_scalar(raw) {{
        set_error(1u, idx);
        Out.data[idx * 2u] = {ty}(0.0);
        Out.data[idx * 2u + 1u] = {ty}(0.0);
        return;
    }}
    if raw < {ty}(0.0) {{
        set_error(3u, idx);
        Out.data[idx * 2u] = {ty}(0.0);
        Out.data[idx * 2u + 1u] = {ty}(0.0);
        return;
    }}
    if raw > {ty}(ORDER - 1u) {{
        set_error(2u, idx);
        Out.data[idx * 2u] = {ty}(0.0);
        Out.data[idx * 2u + 1u] = {ty}(0.0);
        return;
    }}

    if raw != floor(raw) {{
        set_error(3u, idx);
        Out.data[idx * 2u] = {ty}(0.0);
        Out.data[idx * 2u + 1u] = {ty}(0.0);
        return;
    }}

    let symbol = u32(raw);
    if symbol >= ORDER {{
        set_error(2u, idx);
        Out.data[idx * 2u] = {ty}(0.0);
        Out.data[idx * 2u + 1u] = {ty}(0.0);
        return;
    }}

    let point = symbol * 2u;
    Out.data[idx * 2u] = Constellation.data[point];
    Out.data[idx * 2u + 1u] = Constellation.data[point + 1u];
}}
"#,
        ty = ty,
        order = order,
        max_val = max_val,
        workgroup_size = workgroup_size,
    );
    super::nan_predicate::inject_finite(&source, precision)
}

pub(crate) fn modulate_bits_constellation_shader(
    precision: NumericPrecision,
    order: usize,
    workgroup_size: u32,
) -> String {
    let ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let max_val = match precision {
        NumericPrecision::F64 => "1.7976931348623157e308",
        NumericPrecision::F32 => "3.4028234663852886e38",
    };
    let source = format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct ErrorState {{
    state: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}};

struct Params {{
    output_len: u32,
    input_rows: u32,
    output_rows: u32,
    bits_per_symbol: u32,
}};

@group(0) @binding(0) var<storage, read> Bits: Tensor;
@group(0) @binding(1) var<storage, read> Constellation: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<storage, read_write> Error: ErrorState;
@group(0) @binding(4) var<uniform> params: Params;

const ORDER: u32 = {order}u;
const MAX_FINITE: {ty} = {ty}({max_val});

fn isFinite(value: {ty}) -> bool {{ @IEEE_FINITE_PREDICATE@ }}

fn isfinite_scalar(x: {ty}) -> bool {{
    return isFinite(x) && (abs(x) < MAX_FINITE);
}}

fn set_error(code: u32, index: u32) {{
    let packed_index = min(index, 0x3ffffffeu);
    let packed = (code << 30u) | packed_index;
    atomicMin(&Error.state, packed);
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let out_idx = gid.x;
    if out_idx >= params.output_len {{
        return;
    }}
    let channel = out_idx / params.output_rows;
    let group = out_idx - channel * params.output_rows;
    var symbol: u32 = 0u;
    for (var bit_idx: u32 = 0u; bit_idx < params.bits_per_symbol; bit_idx = bit_idx + 1u) {{
        let input_row = group * params.bits_per_symbol + bit_idx;
        let input_idx = input_row + channel * params.input_rows;
        let raw = Bits.data[input_idx];
        if !isfinite_scalar(raw) {{
            set_error(1u, input_idx);
            return;
        }}
        if raw != {ty}(0.0) && raw != {ty}(1.0) {{
            set_error(2u, input_idx);
            return;
        }}
        symbol = (symbol << 1u) | u32(raw);
    }}
    if symbol >= ORDER {{
        set_error(3u, out_idx);
        return;
    }}

    let point = symbol * 2u;
    Out.data[out_idx * 2u] = Constellation.data[point];
    Out.data[out_idx * 2u + 1u] = Constellation.data[point + 1u];
}}
"#,
        ty = ty,
        order = order,
        max_val = max_val,
        workgroup_size = workgroup_size,
    );
    super::nan_predicate::inject_finite(&source, precision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulation_shaders_require_exact_symbols_and_bits() {
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            let ty = match precision {
                NumericPrecision::F32 => "f32",
                NumericPrecision::F64 => "f64",
            };
            let symbol = modulate_constellation_shader(precision, 4, 64);
            assert!(symbol.contains("raw != floor(raw)"));
            assert!(!symbol.contains("EPSILON"));

            let bits = modulate_bits_constellation_shader(precision, 4, 64);
            assert!(bits.contains(&format!("raw != {ty}(0.0) && raw != {ty}(1.0)")));
            assert!(!bits.contains("BIT_TOL"));
            assert!(!bits.contains("round(raw)"));
        }
    }
}
