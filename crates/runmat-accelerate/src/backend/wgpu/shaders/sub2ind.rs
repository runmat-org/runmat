use std::fmt::Write;

pub fn build_sub2ind_shader(
    scalar_ty: &str,
    dims: &[u32],
    strides: &[u32],
    scalar_mask: &[u32],
    workgroup_size: u32,
    epsilon: &str,
) -> String {
    assert_eq!(dims.len(), strides.len());
    assert_eq!(dims.len(), scalar_mask.len());

    let mut shader = String::new();
    writeln!(shader, "struct Tensor {{ data: array<{scalar_ty}>; }};").unwrap();
    writeln!(
        shader,
        "struct ErrorState {{ code: atomic<u32>, dim: atomic<u32>, extra: atomic<u32>, pad: atomic<u32> }};"
    )
    .unwrap();
    writeln!(shader, "struct Params {{ len: u32, }}").unwrap();
    for (idx, _) in dims.iter().enumerate() {
        writeln!(
            shader,
            "@group(0) @binding({}) var<storage, read> input{}: Tensor;",
            idx, idx
        )
        .unwrap();
    }
    writeln!(
        shader,
        "@group(0) @binding({}) var<storage, read_write> output: Tensor;",
        dims.len()
    )
    .unwrap();
    writeln!(
        shader,
        "@group(0) @binding({}) var<storage, read_write> error: ErrorState;",
        dims.len() + 1
    )
    .unwrap();
    writeln!(
        shader,
        "@group(0) @binding({}) var<uniform> params: Params;",
        dims.len() + 2
    )
    .unwrap();
    writeln!(shader, "const EPSILON: {scalar_ty} = {epsilon};").unwrap();
    for (idx, dim) in dims.iter().enumerate() {
        writeln!(shader, "const DIM_{idx}: u32 = {dim}u;").unwrap();
    }
    for (idx, stride) in strides.iter().enumerate() {
        writeln!(shader, "const STRIDE_{idx}: u32 = {stride}u;").unwrap();
    }
    for (idx, mask) in scalar_mask.iter().enumerate() {
        writeln!(shader, "const SCALAR_MASK_{idx}: u32 = {mask}u;").unwrap();
    }
    writeln!(
        shader,
        "fn set_error(code: u32, dim: u32, extra: u32) {{
    let res = atomicCompareExchangeWeak(&error.code, 0u, code);
    if (!res.exchanged) {{
        return;
    }}
    atomicStore(&error.dim, dim);
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
    var offset: u32 = 0u;"
    )
    .unwrap();

    for (idx, _) in dims.iter().enumerate() {
        let dim_id = (idx + 1) as u32;
        writeln!(
            shader,
            "    {{
        let raw: {scalar_ty} = if (SCALAR_MASK_{idx} != 0u) {{
            input{idx}.data[0u]
        }} else {{
            input{idx}.data[idx]
        }};
        if (!isFinite(raw)) {{
            set_error(1u, {dim_id}u, 0u);
            return;
        }}
        let rounded: {scalar_ty} = round(raw);
        if (abs(rounded - raw) > EPSILON) {{
            set_error(2u, {dim_id}u, 0u);
            return;
        }}
        let int_val: i32 = i32(rounded);
        if (int_val < 1 || int_val > i32(DIM_{idx})) {{
            set_error(3u, {dim_id}u, u32(int_val));
            return;
        }}
        let term: u32 = (u32(int_val) - 1u) * STRIDE_{idx};
        offset = offset + term;
    }}"
        )
        .unwrap();
    }

    writeln!(
        shader,
        "    output.data[idx] = {scalar_ty}(offset + 1u);
}}"
    )
    .unwrap();

    shader
}
