use crate::backend::wgpu::types::NumericPrecision;

pub fn triangular_linsolve_shader(
    precision: NumericPrecision,
    transpose: bool,
    lower: bool,
) -> String {
    let scalar_ty = match precision {
        NumericPrecision::F64 => "f64",
        NumericPrecision::F32 => "f32",
    };
    let a_at = if transpose {
        "return A.data[col + row * rows];"
    } else {
        "return A.data[row + col * rows];"
    };
    let body = if lower {
        format!(
            "var sum: {scalar_ty} = B.data[idx];
             var k: u32 = 0u;
             loop {{
               if (k >= target_row) {{ break; }}
               sum = sum - a_at(target_row, k, rows) * Prev.data[k + col * rows];
               k = k + 1u;
             }}
             let diag = a_at(target_row, target_row, rows);
             Out.data[idx] = sum / diag;"
        )
    } else {
        format!(
            "var sum: {scalar_ty} = B.data[idx];
             var k: u32 = target_row + 1u;
             loop {{
               if (k >= rows) {{ break; }}
               sum = sum - a_at(target_row, k, rows) * Prev.data[k + col * rows];
               k = k + 1u;
             }}
             let diag = a_at(target_row, target_row, rows);
             Out.data[idx] = sum / diag;"
        )
    };
    format!(
        r#"
struct Tensor {{ data: array<{scalar_ty}>, }};

struct Params {{
    len: u32,
    offset: u32,
    total: u32,
    rows: u32,
    rhs_cols: u32,
    target_row: u32,
    _pad0: u32,
    _pad1: u32,
}};

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read> Prev: Tensor;
@group(0) @binding(3) var<storage, read_write> Out: Tensor;
@group(0) @binding(4) var<uniform> params: Params;

fn a_at(row: u32, col: u32, rows: u32) -> {scalar_ty} {{
    {a_at}
}}

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if (local >= params.len) {{
        return;
    }}
    let idx = params.offset + local;
    if (idx >= params.total) {{
        return;
    }}
    let rows: u32 = params.rows;
    let cols_rhs: u32 = params.rhs_cols;
    let target_row: u32 = params.target_row;
    let row = idx % rows;
    let col = idx / rows;
    if (col >= cols_rhs) {{
        return;
    }}
    if (row != target_row) {{
        Out.data[idx] = Prev.data[idx];
        return;
    }}
    {body}
}}
"#
    )
}
