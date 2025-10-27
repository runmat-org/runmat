use std::fmt::Write;

pub fn build_ind2sub_shader(
    scalar_ty: &str,
    dims: &[u32],
    strides: &[u32],
    total: u32,
    workgroup_size: u32,
    epsilon: &str,
) -> String {
    assert_eq!(dims.len(), strides.len());

    let mut shader = String::new();
    writeln!(shader, "struct Tensor {{ data: array<{scalar_ty}>; }};").unwrap();
    writeln!(
        shader,
        "struct ErrorState {{ code: atomic<u32>, dim: atomic<u32>, extra: atomic<u32>, pad: atomic<u32> }};"
    )
    .unwrap();
    writeln!(
        shader,
        "struct Params {{ len: u32, _pad0: u32, _pad1: u32, _pad2: u32, }};"
    )
    .unwrap();
    writeln!(
        shader,
        "@group(0) @binding(0) var<storage, read> input: Tensor;"
    )
    .unwrap();
    for (idx, _) in dims.iter().enumerate() {
        writeln!(
            shader,
            "@group(0) @binding({}) var<storage, read_write> output{}: Tensor;",
            idx + 1,
            idx
        )
        .unwrap();
    }
    let error_binding = dims.len() + 1;
    writeln!(
        shader,
        "@group(0) @binding({}) var<storage, read_write> error: ErrorState;",
        error_binding
    )
    .unwrap();
    let params_binding = dims.len() + 2;
    writeln!(
        shader,
        "@group(0) @binding({}) var<uniform> params: Params;",
        params_binding
    )
    .unwrap();
    writeln!(shader, "const EPSILON: {scalar_ty} = {epsilon};").unwrap();
    writeln!(shader, "const ONE: {scalar_ty} = {scalar_ty}(1.0);").unwrap();
    writeln!(shader, "const TOTAL_F: {scalar_ty} = {scalar_ty}({total});").unwrap();
    writeln!(shader, "const TOTAL: u32 = {total}u;").unwrap();
    for (idx, dim) in dims.iter().enumerate() {
        writeln!(shader, "const DIM_{idx}: u32 = {dim}u;").unwrap();
    }
    for (idx, stride) in strides.iter().enumerate() {
        writeln!(shader, "const STRIDE_{idx}: u32 = {stride}u;").unwrap();
    }
    writeln!(
        shader,
        "fn set_error(code: u32, extra: u32) {{
    let res = atomicCompareExchangeWeak(&error.code, 0u, code);
    if (!res.exchanged) {{
        return;
    }}
    atomicStore(&error.dim, 0u);
    atomicStore(&error.extra, extra);
}}"
    )
    .unwrap();
    writeln!(
        shader,
        "@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.len) {{
        return;
    }}
    if (atomicLoad(&error.code) != 0u) {{
        return;
    }}
    let raw: {scalar_ty} = input.data[idx];
    if (!isFinite(raw)) {{
        set_error(1u, 0u);
        return;
    }}
    let rounded: {scalar_ty} = round(raw);
    if (abs(rounded - raw) > EPSILON) {{
        set_error(2u, 0u);
        return;
    }}
    if (rounded < ONE) {{
        set_error(3u, 0u);
        return;
    }}
    if (rounded > TOTAL_F) {{
        set_error(4u, 0u);
        return;
    }}
    let rounded_u32: u32 = u32(rounded);
    if (rounded_u32 == 0u) {{
        set_error(3u, 0u);
        return;
    }}
    let zero_based: u32 = rounded_u32 - 1u;
"
    )
    .unwrap();

    for (idx, _) in dims.iter().enumerate() {
        writeln!(
            shader,
            "    let coord_{idx}: u32 = (zero_based / STRIDE_{idx}) % DIM_{idx};
    output{idx}.data[idx] = {scalar_ty}(coord_{idx} + 1u);"
        )
        .unwrap();
    }

    writeln!(shader, "}}").unwrap();

    shader
}
