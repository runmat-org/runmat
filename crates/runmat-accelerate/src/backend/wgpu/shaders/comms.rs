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
    format!(
        r#"
struct Tensor {{
    data: array<{ty}>,
}};

struct ErrorState {{
    code: u32,
    index: u32,
    _pad0: u32,
    _pad1: u32,
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
const EPSILON: {ty} = {epsilon};
const MAX_FINITE: {ty} = {ty}({max_val});

fn isfinite_scalar(x: {ty}) -> bool {{
    return (x == x) && (abs(x) < MAX_FINITE);
}}

fn set_error(code: u32, index: u32) {{
    if Error.code != 0u {{
        return;
    }}
    Error.code = code;
    Error.index = index;
}}

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= params.len {{
        return;
    }}
    if Error.code != 0u {{
        return;
    }}

    let raw = Symbols.data[idx];
    if !isfinite_scalar(raw) {{
        set_error(1u, idx);
        Out.data[idx * 2u] = {ty}(0.0);
        Out.data[idx * 2u + 1u] = {ty}(0.0);
        return;
    }}
    if raw < {ty}(0.0) || raw > {ty}(ORDER - 1u) + {ty}(0.5) {{
        set_error(2u, idx);
        Out.data[idx * 2u] = {ty}(0.0);
        Out.data[idx * 2u + 1u] = {ty}(0.0);
        return;
    }}

    let rounded = round(raw);
    if abs(rounded - raw) > EPSILON {{
        set_error(3u, idx);
        Out.data[idx * 2u] = {ty}(0.0);
        Out.data[idx * 2u + 1u] = {ty}(0.0);
        return;
    }}

    let symbol = u32(rounded);
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
        epsilon = match precision {
            NumericPrecision::F64 => "1.0e-9",
            NumericPrecision::F32 => "0.0",
        },
        max_val = max_val,
        workgroup_size = workgroup_size,
    )
}
