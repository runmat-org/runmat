pub const CONV1D_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    signal_len: u32,
    kernel_len: u32,
    output_len: u32,
    start_offset: u32,
};

@group(0) @binding(0) var<storage, read> Signal: Tensor;
@group(0) @binding(1) var<storage, read> Kernel: Tensor;
@group(0) @binding(2) var<storage, read_write> Output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx >= params.output_len) {
        return;
    }

    let full_index = params.start_offset + idx;
    let signal_len = params.signal_len;
    let kernel_len = params.kernel_len;
    var acc: f64 = 0.0;

    let full_index_i32 = i32(full_index);
    var k: u32 = 0u;
    loop {
        if (k >= kernel_len) {
            break;
        }
        let signal_pos = full_index_i32 - i32(k);
        if (signal_pos >= 0 && signal_pos < i32(signal_len)) {
            let s_idx = u32(signal_pos);
            acc = acc + Signal.data[s_idx] * Kernel.data[k];
        }
        k = k + 1u;
    }

    Output.data[idx] = acc;
}
"#;

pub const CONV1D_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    signal_len: u32,
    kernel_len: u32,
    output_len: u32,
    start_offset: u32,
};

@group(0) @binding(0) var<storage, read> Signal: Tensor;
@group(0) @binding(1) var<storage, read> Kernel: Tensor;
@group(0) @binding(2) var<storage, read_write> Output: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx >= params.output_len) {
        return;
    }

    let full_index = params.start_offset + idx;
    let signal_len = params.signal_len;
    let kernel_len = params.kernel_len;
    var acc: f32 = 0.0;

    let full_index_i32 = i32(full_index);
    var k: u32 = 0u;
    loop {
        if (k >= kernel_len) {
            break;
        }
        let signal_pos = full_index_i32 - i32(k);
        if (signal_pos >= 0 && signal_pos < i32(signal_len)) {
            let s_idx = u32(signal_pos);
            acc = acc + Signal.data[s_idx] * Kernel.data[k];
        }
        k = k + 1u;
    }

    Output.data[idx] = acc;
}
"#;
