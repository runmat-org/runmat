pub const FFT_INIT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    target_len: u32,
    inner_stride: u32,
    current_len: u32,
    copy_len: u32,
    input_complex: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }

    let inner = idx % params.inner_stride;
    let tmp = idx / params.inner_stride;
    let p = tmp % params.target_len;
    let outer = tmp / params.target_len;

    var re = 0.0;
    var im = 0.0;
    if p < params.copy_len {
        let src = outer * params.current_len * params.inner_stride + inner + p * params.inner_stride;
        if params.input_complex != 0u {
            let base = src * 2u;
            re = Input.data[base];
            im = Input.data[base + 1u];
        } else {
            re = Input.data[src];
        }
    }

    let dst = idx * 2u;
    Output.data[dst] = re;
    Output.data[dst + 1u] = im;
}
"#;

pub const FFT_STAGE_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    target_len: u32,
    inner_stride: u32,
    stage_span: u32,
    stage_half: u32,
    twiddle_step: u32,
    inverse: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }

    let inner = idx % params.inner_stride;
    let tmp = idx / params.inner_stride;
    let p = tmp % params.target_len;
    let outer = tmp / params.target_len;

    let within = p % params.stage_span;
    let j = within % params.stage_half;
    let block = p - within;
    let low = block + j;
    let high = low + params.stage_half;

    let low_idx = outer * params.target_len * params.inner_stride + inner + low * params.inner_stride;
    let high_idx = outer * params.target_len * params.inner_stride + inner + high * params.inner_stride;

    let low_base = low_idx * 2u;
    let high_base = high_idx * 2u;
    let ar = Input.data[low_base];
    let ai = Input.data[low_base + 1u];
    let br = Input.data[high_base];
    let bi = Input.data[high_base + 1u];

    var out_re = 0.0;
    var out_im = 0.0;
    if within < params.stage_half {
        out_re = ar + br;
        out_im = ai + bi;
    } else {
        let diff_re = ar - br;
        let diff_im = ai - bi;
        let angle = -6.283185307179586 * f64(j * params.twiddle_step) / f64(params.target_len);
        let wr = cos(angle);
        var wi = sin(angle);
        if params.inverse != 0u {
            wi = -wi;
        }
        out_re = diff_re * wr - diff_im * wi;
        out_im = diff_re * wi + diff_im * wr;
    }
    let out_base = idx * 2u;
    Output.data[out_base] = out_re;
    Output.data[out_base + 1u] = out_im;
}
"#;

pub const FFT_REORDER_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    target_len: u32,
    inner_stride: u32,
    log2_len: u32,
    inverse: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn bit_reverse(value: u32, bits: u32) -> u32 {
    var v = value;
    var r = 0u;
    var i = 0u;
    loop {
        if i >= bits {
            break;
        }
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
        i = i + 1u;
    }
    return r;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }

    let inner = idx % params.inner_stride;
    let tmp = idx / params.inner_stride;
    let p = tmp % params.target_len;
    let outer = tmp / params.target_len;

    let rev = bit_reverse(p, params.log2_len);
    let src_idx = outer * params.target_len * params.inner_stride + inner + rev * params.inner_stride;

    let src_base = src_idx * 2u;
    let dst_base = idx * 2u;
    var re = Input.data[src_base];
    var im = Input.data[src_base + 1u];
    if params.inverse != 0u {
        let scale = 1.0 / f64(params.target_len);
        re = re * scale;
        im = im * scale;
    }
    Output.data[dst_base] = re;
    Output.data[dst_base + 1u] = im;
}
"#;

pub const FFT_INIT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    target_len: u32,
    inner_stride: u32,
    current_len: u32,
    copy_len: u32,
    input_complex: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }

    let inner = idx % params.inner_stride;
    let tmp = idx / params.inner_stride;
    let p = tmp % params.target_len;
    let outer = tmp / params.target_len;

    var re = 0.0;
    var im = 0.0;
    if p < params.copy_len {
        let src = outer * params.current_len * params.inner_stride + inner + p * params.inner_stride;
        if params.input_complex != 0u {
            let base = src * 2u;
            re = Input.data[base];
            im = Input.data[base + 1u];
        } else {
            re = Input.data[src];
        }
    }

    let dst = idx * 2u;
    Output.data[dst] = re;
    Output.data[dst + 1u] = im;
}
"#;

pub const FFT_STAGE_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    target_len: u32,
    inner_stride: u32,
    stage_span: u32,
    stage_half: u32,
    twiddle_step: u32,
    inverse: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }

    let inner = idx % params.inner_stride;
    let tmp = idx / params.inner_stride;
    let p = tmp % params.target_len;
    let outer = tmp / params.target_len;

    let within = p % params.stage_span;
    let j = within % params.stage_half;
    let block = p - within;
    let low = block + j;
    let high = low + params.stage_half;

    let low_idx = outer * params.target_len * params.inner_stride + inner + low * params.inner_stride;
    let high_idx = outer * params.target_len * params.inner_stride + inner + high * params.inner_stride;

    let low_base = low_idx * 2u;
    let high_base = high_idx * 2u;
    let ar = Input.data[low_base];
    let ai = Input.data[low_base + 1u];
    let br = Input.data[high_base];
    let bi = Input.data[high_base + 1u];

    var out_re = 0.0;
    var out_im = 0.0;
    if within < params.stage_half {
        out_re = ar + br;
        out_im = ai + bi;
    } else {
        let diff_re = ar - br;
        let diff_im = ai - bi;
        let angle = -6.283185307179586 * f32(j * params.twiddle_step) / f32(params.target_len);
        let wr = cos(angle);
        var wi = sin(angle);
        if params.inverse != 0u {
            wi = -wi;
        }
        out_re = diff_re * wr - diff_im * wi;
        out_im = diff_re * wi + diff_im * wr;
    }
    let out_base = idx * 2u;
    Output.data[out_base] = out_re;
    Output.data[out_base + 1u] = out_im;
}
"#;

pub const FFT_REORDER_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    target_len: u32,
    inner_stride: u32,
    log2_len: u32,
    inverse: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn bit_reverse(value: u32, bits: u32) -> u32 {
    var v = value;
    var r = 0u;
    var i = 0u;
    loop {
        if i >= bits {
            break;
        }
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
        i = i + 1u;
    }
    return r;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }

    let inner = idx % params.inner_stride;
    let tmp = idx / params.inner_stride;
    let p = tmp % params.target_len;
    let outer = tmp / params.target_len;

    let rev = bit_reverse(p, params.log2_len);
    let src_idx = outer * params.target_len * params.inner_stride + inner + rev * params.inner_stride;

    let src_base = src_idx * 2u;
    let dst_base = idx * 2u;
    var re = Input.data[src_base];
    var im = Input.data[src_base + 1u];
    if params.inverse != 0u {
        let scale = 1.0 / f32(params.target_len);
        re = re * scale;
        im = im * scale;
    }
    Output.data[dst_base] = re;
    Output.data[dst_base + 1u] = im;
}
"#;

pub const FFT_DIRECT_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    target_len: u32,
    inner_stride: u32,
    current_len: u32,
    copy_len: u32,
    input_complex: u32,
    inverse: u32,
    phase_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn load_complex(data: ptr<storage, array<f64>, read>, index: u32) -> vec2<f64> {
    let base = index * 2u;
    return vec2<f64>((*data)[base], (*data)[base + 1u]);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }

    let inner = idx % params.inner_stride;
    let tmp = idx / params.inner_stride;
    let k = tmp % params.target_len;
    let outer = tmp / params.target_len;

    var sum_re = 0.0;
    var sum_im = 0.0;
    var n = 0u;
    loop {
        if n >= params.copy_len {
            break;
        }
        let src = outer * params.current_len * params.inner_stride + inner + n * params.inner_stride;
        var xr = 0.0;
        var xi = 0.0;
        if params.input_complex != 0u {
            let base = src * 2u;
            xr = Input.data[base];
            xi = Input.data[base + 1u];
        } else {
            xr = Input.data[src];
        }

        let angle = f64(params.phase_scale) * f64(n) * f64(k);
        let wr = cos(angle);
        var wi = sin(angle);
        if params.inverse != 0u {
            wi = -wi;
        }
        sum_re = sum_re + xr * wr - xi * wi;
        sum_im = sum_im + xr * wi + xi * wr;
        n = n + 1u;
    }

    if params.inverse != 0u {
        let scale = 1.0 / f64(params.target_len);
        sum_re = sum_re * scale;
        sum_im = sum_im * scale;
    }

    let dst = idx * 2u;
    Output.data[dst] = sum_re;
    Output.data[dst + 1u] = sum_im;
}
"#;

pub const FFT_DIRECT_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    target_len: u32,
    inner_stride: u32,
    current_len: u32,
    copy_len: u32,
    input_complex: u32,
    inverse: u32,
    phase_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }

    let inner = idx % params.inner_stride;
    let tmp = idx / params.inner_stride;
    let k = tmp % params.target_len;
    let outer = tmp / params.target_len;

    var sum_re = 0.0;
    var sum_im = 0.0;
    var n = 0u;
    loop {
        if n >= params.copy_len {
            break;
        }
        let src = outer * params.current_len * params.inner_stride + inner + n * params.inner_stride;
        var xr = 0.0;
        var xi = 0.0;
        if params.input_complex != 0u {
            let base = src * 2u;
            xr = Input.data[base];
            xi = Input.data[base + 1u];
        } else {
            xr = Input.data[src];
        }

        let angle = params.phase_scale * f32(n) * f32(k);
        let wr = cos(angle);
        var wi = sin(angle);
        if params.inverse != 0u {
            wi = -wi;
        }
        sum_re = sum_re + xr * wr - xi * wi;
        sum_im = sum_im + xr * wi + xi * wr;
        n = n + 1u;
    }

    if params.inverse != 0u {
        let scale = 1.0 / f32(params.target_len);
        sum_re = sum_re * scale;
        sum_im = sum_im * scale;
    }

    let dst = idx * 2u;
    Output.data[dst] = sum_re;
    Output.data[dst + 1u] = sum_im;
}
"#;

pub const FFT_EXTRACT_REAL_SHADER_F64: &str = r#"
struct InputTensor {
    data: array<f64>,
};

struct OutputTensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> Input: InputTensor;
@group(0) @binding(1) var<storage, read_write> Output: OutputTensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }
    Output.data[idx] = Input.data[idx * 2u];
}
"#;

pub const FFT_EXTRACT_REAL_SHADER_F32: &str = r#"
struct InputTensor {
    data: array<f32>,
};

struct OutputTensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    offset: u32,
    total: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> Input: InputTensor;
@group(0) @binding(1) var<storage, read_write> Output: OutputTensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    if local_index >= params.len {
        return;
    }
    let idx = params.offset + local_index;
    if idx >= params.total {
        return;
    }
    Output.data[idx] = Input.data[idx * 2u];
}
"#;

pub const FFT_STAGE3_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  stage_span: u32, stage_third: u32, twiddle_step: u32, inverse: u32,
  _pad0: u32, _pad1: u32, _pad2: u32,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> Twiddles: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }

  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;

  let within = p % params.stage_span;
  let j = within % params.stage_third;
  let region = within / params.stage_third;
  let block = p - within;
  let i0 = block + j;
  let i1 = i0 + params.stage_third;
  let i2 = i1 + params.stage_third;

  let stride2 = params.inner_stride * 2u;
  let base = (outer * params.target_len * params.inner_stride + inner) * 2u;
  let a_base = base + i0 * stride2;
  let b_base = base + i1 * stride2;
  let c_base = base + i2 * stride2;
  let ar = Input.data[a_base];
  let ai = Input.data[a_base + 1u];
  let br = Input.data[b_base];
  let bi = Input.data[b_base + 1u];
  let cr = Input.data[c_base];
  let ci = Input.data[c_base + 1u];

  let sqrt3_over_2 = 0.86602540378443864676;
  var w1i = -sqrt3_over_2;
  var w2i = sqrt3_over_2;
  if params.inverse != 0u {
    w1i = sqrt3_over_2;
    w2i = -sqrt3_over_2;
  }

  let a0r = ar + br + cr;
  let a0i = ai + bi + ci;
  let a1r = ar + (br * -0.5 - bi * w1i) + (cr * -0.5 - ci * w2i);
  let a1i = ai + (br * w1i + bi * -0.5) + (cr * w2i + ci * -0.5);
  let a2r = ar + (br * -0.5 - bi * w2i) + (cr * -0.5 - ci * w1i);
  let a2i = ai + (br * w2i + bi * -0.5) + (cr * w1i + ci * -0.5);

  let tw_idx = (j * params.twiddle_step) % params.target_len;
  let tw2_idx = (2u * j * params.twiddle_step) % params.target_len;
  let tw_base = tw_idx * 2u;
  let tw2_base = tw2_idx * 2u;
  let twr = Twiddles.data[tw_base];
  var twi = Twiddles.data[tw_base + 1u];
  let tw2r = Twiddles.data[tw2_base];
  var tw2i = Twiddles.data[tw2_base + 1u];
  if params.inverse != 0u {
    twi = -twi;
    tw2i = -tw2i;
  }

  var out_r = a0r;
  var out_i = a0i;
  if region == 1u {
    out_r = a1r * twr - a1i * twi;
    out_i = a1r * twi + a1i * twr;
  }
  if region == 2u {
    out_r = a2r * tw2r - a2i * tw2i;
    out_i = a2r * tw2i + a2i * tw2r;
  }
  let out_base = idx * 2u;
  Output.data[out_base] = out_r;
  Output.data[out_base + 1u] = out_i;
}
"#;

pub const FFT_STAGE3_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  stage_span: u32, stage_third: u32, twiddle_step: u32, inverse: u32,
  _pad0: u32, _pad1: u32, _pad2: u32,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> Twiddles: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }

  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;

  let within = p % params.stage_span;
  let j = within % params.stage_third;
  let region = within / params.stage_third;
  let block = p - within;
  let i0 = block + j;
  let i1 = i0 + params.stage_third;
  let i2 = i1 + params.stage_third;

  let stride2 = params.inner_stride * 2u;
  let base = (outer * params.target_len * params.inner_stride + inner) * 2u;
  let a_base = base + i0 * stride2;
  let b_base = base + i1 * stride2;
  let c_base = base + i2 * stride2;
  let ar = Input.data[a_base];
  let ai = Input.data[a_base + 1u];
  let br = Input.data[b_base];
  let bi = Input.data[b_base + 1u];
  let cr = Input.data[c_base];
  let ci = Input.data[c_base + 1u];

  let sqrt3_over_2 = 0.86602540378443864676;
  var w1i = -sqrt3_over_2;
  var w2i = sqrt3_over_2;
  if params.inverse != 0u {
    w1i = sqrt3_over_2;
    w2i = -sqrt3_over_2;
  }

  let a0r = ar + br + cr;
  let a0i = ai + bi + ci;
  let a1r = ar + (br * -0.5 - bi * w1i) + (cr * -0.5 - ci * w2i);
  let a1i = ai + (br * w1i + bi * -0.5) + (cr * w2i + ci * -0.5);
  let a2r = ar + (br * -0.5 - bi * w2i) + (cr * -0.5 - ci * w1i);
  let a2i = ai + (br * w2i + bi * -0.5) + (cr * w1i + ci * -0.5);

  let tw_idx = (j * params.twiddle_step) % params.target_len;
  let tw2_idx = (2u * j * params.twiddle_step) % params.target_len;
  let tw_base = tw_idx * 2u;
  let tw2_base = tw2_idx * 2u;
  let twr = Twiddles.data[tw_base];
  var twi = Twiddles.data[tw_base + 1u];
  let tw2r = Twiddles.data[tw2_base];
  var tw2i = Twiddles.data[tw2_base + 1u];
  if params.inverse != 0u {
    twi = -twi;
    tw2i = -tw2i;
  }

  var out_r = a0r;
  var out_i = a0i;
  if region == 1u {
    out_r = a1r * twr - a1i * twi;
    out_i = a1r * twi + a1i * twr;
  }
  if region == 2u {
    out_r = a2r * tw2r - a2i * tw2i;
    out_i = a2r * tw2i + a2i * tw2r;
  }
  let out_base = idx * 2u;
  Output.data[out_base] = out_r;
  Output.data[out_base + 1u] = out_i;
}
"#;

pub const FFT_REORDER3_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  digits: u32, inverse: u32, _pad0: u32,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn digit_reverse_3(value: u32, digits: u32) -> u32 {
  var v = value; var r = 0u; var i = 0u;
  loop {
    if i >= digits { break; }
    r = r * 3u + (v % 3u);
    v = v / 3u;
    i = i + 1u;
  }
  return r;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  let rev = digit_reverse_3(p, params.digits);
  let src_idx = outer * params.target_len * params.inner_stride + inner + rev * params.inner_stride;
  let src_base = src_idx * 2u;
  let dst_base = idx * 2u;
  var re = Input.data[src_base];
  var im = Input.data[src_base + 1u];
  if params.inverse != 0u {
    let s = 1.0 / f64(params.target_len);
    re = re * s; im = im * s;
  }
  Output.data[dst_base] = re;
  Output.data[dst_base + 1u] = im;
}
"#;

pub const FFT_REORDER3_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  digits: u32, inverse: u32, _pad0: u32,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn digit_reverse_3(value: u32, digits: u32) -> u32 {
  var v = value; var r = 0u; var i = 0u;
  loop {
    if i >= digits { break; }
    r = r * 3u + (v % 3u);
    v = v / 3u;
    i = i + 1u;
  }
  return r;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  let rev = digit_reverse_3(p, params.digits);
  let src_idx = outer * params.target_len * params.inner_stride + inner + rev * params.inner_stride;
  let src_base = src_idx * 2u;
  let dst_base = idx * 2u;
  var re = Input.data[src_base];
  var im = Input.data[src_base + 1u];
  if params.inverse != 0u {
    let s = 1.0 / f32(params.target_len);
    re = re * s; im = im * s;
  }
  Output.data[dst_base] = re;
  Output.data[dst_base + 1u] = im;
}
"#;

pub const FFT_STAGE5_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  stage_span: u32, stage_fifth: u32, twiddle_step: u32, inverse: u32,
  _pad0: u32, _pad1: u32, _pad2: u32,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> Twiddles: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }

  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;

  let within = p % params.stage_span;
  let j = within % params.stage_fifth;
  let region = within / params.stage_fifth;
  let block = p - within;
  let i0 = block + j;
  let i1 = i0 + params.stage_fifth;
  let i2 = i1 + params.stage_fifth;
  let i3 = i2 + params.stage_fifth;
  let i4 = i3 + params.stage_fifth;

  let stride2 = params.inner_stride * 2u;
  let base = (outer * params.target_len * params.inner_stride + inner) * 2u;
  let x0b = base + i0 * stride2;
  let x1b = base + i1 * stride2;
  let x2b = base + i2 * stride2;
  let x3b = base + i3 * stride2;
  let x4b = base + i4 * stride2;
  let x0r = Input.data[x0b]; let x0i = Input.data[x0b + 1u];
  let x1r = Input.data[x1b]; let x1i = Input.data[x1b + 1u];
  let x2r = Input.data[x2b]; let x2i = Input.data[x2b + 1u];
  let x3r = Input.data[x3b]; let x3i = Input.data[x3b + 1u];
  let x4r = Input.data[x4b]; let x4i = Input.data[x4b + 1u];

  let c1 = 0.30901699437494745;
  let s1 = 0.9510565162951535;
  let c2 = -0.8090169943749473;
  let s2 = 0.5877852522924732;
  let sign = select(-1.0, 1.0, params.inverse != 0u);
  let r = region;

  var w1r = 1.0; var w1i = 0.0;
  var w2r = 1.0; var w2i = 0.0;
  var w3r = 1.0; var w3i = 0.0;
  var w4r = 1.0; var w4i = 0.0;
  if r == 1u {
    w1r = c1; w1i = sign * s1;
    w2r = c2; w2i = sign * s2;
    w3r = c2; w3i = -sign * s2;
    w4r = c1; w4i = -sign * s1;
  } else if r == 2u {
    w1r = c2; w1i = sign * s2;
    w2r = c1; w2i = -sign * s1;
    w3r = c1; w3i = sign * s1;
    w4r = c2; w4i = -sign * s2;
  } else if r == 3u {
    w1r = c2; w1i = -sign * s2;
    w2r = c1; w2i = sign * s1;
    w3r = c1; w3i = -sign * s1;
    w4r = c2; w4i = sign * s2;
  } else if r == 4u {
    w1r = c1; w1i = -sign * s1;
    w2r = c2; w2i = -sign * s2;
    w3r = c2; w3i = sign * s2;
    w4r = c1; w4i = sign * s1;
  }

  var yr = x0r;
  var yi = x0i;
  yr = yr + (x1r * w1r - x1i * w1i);
  yi = yi + (x1r * w1i + x1i * w1r);
  yr = yr + (x2r * w2r - x2i * w2i);
  yi = yi + (x2r * w2i + x2i * w2r);
  yr = yr + (x3r * w3r - x3i * w3i);
  yi = yi + (x3r * w3i + x3i * w3r);
  yr = yr + (x4r * w4r - x4i * w4i);
  yi = yi + (x4r * w4i + x4i * w4r);

  let tw_idx = (r * j * params.twiddle_step) % params.target_len;
  let tw_base = tw_idx * 2u;
  let twr = Twiddles.data[tw_base];
  var twi = Twiddles.data[tw_base + 1u];
  if params.inverse != 0u { twi = -twi; }
  let out_r = yr * twr - yi * twi;
  let out_i = yr * twi + yi * twr;
  let out_base = idx * 2u;
  Output.data[out_base] = out_r;
  Output.data[out_base + 1u] = out_i;
}
"#;

pub const FFT_STAGE5_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  stage_span: u32, stage_fifth: u32, twiddle_step: u32, inverse: u32,
  _pad0: u32, _pad1: u32, _pad2: u32,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> Twiddles: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }

  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;

  let within = p % params.stage_span;
  let j = within % params.stage_fifth;
  let region = within / params.stage_fifth;
  let block = p - within;
  let i0 = block + j;
  let i1 = i0 + params.stage_fifth;
  let i2 = i1 + params.stage_fifth;
  let i3 = i2 + params.stage_fifth;
  let i4 = i3 + params.stage_fifth;

  let stride2 = params.inner_stride * 2u;
  let base = (outer * params.target_len * params.inner_stride + inner) * 2u;
  let x0b = base + i0 * stride2;
  let x1b = base + i1 * stride2;
  let x2b = base + i2 * stride2;
  let x3b = base + i3 * stride2;
  let x4b = base + i4 * stride2;
  let x0r = Input.data[x0b]; let x0i = Input.data[x0b + 1u];
  let x1r = Input.data[x1b]; let x1i = Input.data[x1b + 1u];
  let x2r = Input.data[x2b]; let x2i = Input.data[x2b + 1u];
  let x3r = Input.data[x3b]; let x3i = Input.data[x3b + 1u];
  let x4r = Input.data[x4b]; let x4i = Input.data[x4b + 1u];

  let c1 = 0.30901699437494745;
  let s1 = 0.9510565162951535;
  let c2 = -0.8090169943749473;
  let s2 = 0.5877852522924732;
  let sign = select(-1.0, 1.0, params.inverse != 0u);
  let r = region;

  var w1r = 1.0; var w1i = 0.0;
  var w2r = 1.0; var w2i = 0.0;
  var w3r = 1.0; var w3i = 0.0;
  var w4r = 1.0; var w4i = 0.0;
  if r == 1u {
    w1r = c1; w1i = sign * s1;
    w2r = c2; w2i = sign * s2;
    w3r = c2; w3i = -sign * s2;
    w4r = c1; w4i = -sign * s1;
  } else if r == 2u {
    w1r = c2; w1i = sign * s2;
    w2r = c1; w2i = -sign * s1;
    w3r = c1; w3i = sign * s1;
    w4r = c2; w4i = -sign * s2;
  } else if r == 3u {
    w1r = c2; w1i = -sign * s2;
    w2r = c1; w2i = sign * s1;
    w3r = c1; w3i = -sign * s1;
    w4r = c2; w4i = sign * s2;
  } else if r == 4u {
    w1r = c1; w1i = -sign * s1;
    w2r = c2; w2i = -sign * s2;
    w3r = c2; w3i = sign * s2;
    w4r = c1; w4i = sign * s1;
  }

  var yr = x0r;
  var yi = x0i;
  yr = yr + (x1r * w1r - x1i * w1i);
  yi = yi + (x1r * w1i + x1i * w1r);
  yr = yr + (x2r * w2r - x2i * w2i);
  yi = yi + (x2r * w2i + x2i * w2r);
  yr = yr + (x3r * w3r - x3i * w3i);
  yi = yi + (x3r * w3i + x3i * w3r);
  yr = yr + (x4r * w4r - x4i * w4i);
  yi = yi + (x4r * w4i + x4i * w4r);

  let tw_idx = (r * j * params.twiddle_step) % params.target_len;
  let tw_base = tw_idx * 2u;
  let twr = Twiddles.data[tw_base];
  var twi = Twiddles.data[tw_base + 1u];
  if params.inverse != 0u { twi = -twi; }
  let out_r = yr * twr - yi * twi;
  let out_i = yr * twi + yi * twr;
  let out_base = idx * 2u;
  Output.data[out_base] = out_r;
  Output.data[out_base + 1u] = out_i;
}
"#;

pub const FFT_REORDER5_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  digits: u32, inverse: u32, _pad0: u32,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn digit_reverse_5(value: u32, digits: u32) -> u32 {
  var v = value; var r = 0u; var i = 0u;
  loop {
    if i >= digits { break; }
    r = r * 5u + (v % 5u);
    v = v / 5u;
    i = i + 1u;
  }
  return r;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  let rev = digit_reverse_5(p, params.digits);
  let src_idx = outer * params.target_len * params.inner_stride + inner + rev * params.inner_stride;
  let src_base = src_idx * 2u;
  let dst_base = idx * 2u;
  var re = Input.data[src_base];
  var im = Input.data[src_base + 1u];
  if params.inverse != 0u {
    let s = 1.0 / f64(params.target_len);
    re = re * s; im = im * s;
  }
  Output.data[dst_base] = re;
  Output.data[dst_base + 1u] = im;
}
"#;

pub const FFT_REORDER5_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  digits: u32, inverse: u32, _pad0: u32,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn digit_reverse_5(value: u32, digits: u32) -> u32 {
  var v = value; var r = 0u; var i = 0u;
  loop {
    if i >= digits { break; }
    r = r * 5u + (v % 5u);
    v = v / 5u;
    i = i + 1u;
  }
  return r;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  let rev = digit_reverse_5(p, params.digits);
  let src_idx = outer * params.target_len * params.inner_stride + inner + rev * params.inner_stride;
  let src_base = src_idx * 2u;
  let dst_base = idx * 2u;
  var re = Input.data[src_base];
  var im = Input.data[src_base + 1u];
  if params.inverse != 0u {
    let s = 1.0 / f32(params.target_len);
    re = re * s; im = im * s;
  }
  Output.data[dst_base] = re;
  Output.data[dst_base + 1u] = im;
}
"#;

pub const FFT_REORDER_MIXED_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  stage_count: u32, inverse: u32, _pad0: u32,
  radices: array<u32, 16>,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> params: Params;

fn digit_reverse_mixed(value: u32) -> u32 {
  var v = value;
  var r = 0u;
  var i = 0u;
  loop {
    if i >= params.stage_count { break; }
    let radix = params.radices[i];
    let d = v % radix;
    v = v / radix;
    r = r * radix + d;
    i = i + 1u;
  }
  return r;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  let rev = digit_reverse_mixed(p);
  let src_idx = outer * params.target_len * params.inner_stride + inner + rev * params.inner_stride;
  let src_base = src_idx * 2u;
  let dst_base = idx * 2u;
  var re = Input.data[src_base];
  var im = Input.data[src_base + 1u];
  if params.inverse != 0u {
    let s = 1.0 / f64(params.target_len);
    re = re * s; im = im * s;
  }
  Output.data[dst_base] = re;
  Output.data[dst_base + 1u] = im;
}
"#;

pub const FFT_REORDER_MIXED_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params {
  len: u32, offset: u32, total: u32, target_len: u32, inner_stride: u32,
  stage_count: u32, inverse: u32, _pad0: u32,
  radices: array<u32, 16>,
};
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> params: Params;

fn digit_reverse_mixed(value: u32) -> u32 {
  var v = value;
  var r = 0u;
  var i = 0u;
  loop {
    if i >= params.stage_count { break; }
    let radix = params.radices[i];
    let d = v % radix;
    v = v / radix;
    r = r * radix + d;
    i = i + 1u;
  }
  return r;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_index = gid.x;
  if local_index >= params.len { return; }
  let idx = params.offset + local_index;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  let rev = digit_reverse_mixed(p);
  let src_idx = outer * params.target_len * params.inner_stride + inner + rev * params.inner_stride;
  let src_base = src_idx * 2u;
  let dst_base = idx * 2u;
  var re = Input.data[src_base];
  var im = Input.data[src_base + 1u];
  if params.inverse != 0u {
    let s = 1.0 / f32(params.target_len);
    re = re * s; im = im * s;
  }
  Output.data[dst_base] = re;
  Output.data[dst_base + 1u] = im;
}
"#;

pub const FFT_BLUESTEIN_PREP_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params { len:u32, offset:u32, total:u32, target_len:u32, inner_stride:u32, current_len:u32, copy_len:u32, input_complex:u32, };
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> Chirp: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
fn cmul(ar:f64, ai:f64, br:f64, bi:f64) -> vec2<f64> { return vec2<f64>(ar*br-ai*bi, ar*bi+ai*br); }
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.len { return; }
  let idx = params.offset + i;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  var xr = 0.0; var xi = 0.0;
  if p < params.copy_len {
    let src = outer * params.current_len * params.inner_stride + inner + p * params.inner_stride;
    if params.input_complex != 0u {
      let b = src * 2u; xr = Input.data[b]; xi = Input.data[b+1u];
    } else {
      xr = Input.data[src];
    }
  }
  let cb = p * 2u;
  let c = cmul(xr, xi, Chirp.data[cb], Chirp.data[cb+1u]);
  let d = idx * 2u;
  Output.data[d] = c.x; Output.data[d+1u] = c.y;
}
"#;

pub const FFT_BLUESTEIN_PREP_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params { len:u32, offset:u32, total:u32, target_len:u32, inner_stride:u32, current_len:u32, copy_len:u32, input_complex:u32, };
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> Chirp: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
fn cmul(ar:f32, ai:f32, br:f32, bi:f32) -> vec2<f32> { return vec2<f32>(ar*br-ai*bi, ar*bi+ai*br); }
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.len { return; }
  let idx = params.offset + i;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  var xr = 0.0; var xi = 0.0;
  if p < params.copy_len {
    let src = outer * params.current_len * params.inner_stride + inner + p * params.inner_stride;
    if params.input_complex != 0u {
      let b = src * 2u; xr = Input.data[b]; xi = Input.data[b+1u];
    } else {
      xr = Input.data[src];
    }
  }
  let cb = p * 2u;
  let c = cmul(xr, xi, Chirp.data[cb], Chirp.data[cb+1u]);
  let d = idx * 2u;
  Output.data[d] = c.x; Output.data[d+1u] = c.y;
}
"#;

pub const FFT_BLUESTEIN_KERNEL_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params { len:u32, offset:u32, total:u32, target_len:u32, m_len:u32, inverse:u32, _pad0:u32, _pad1:u32, };
@group(0) @binding(0) var<storage, read_write> Output: Tensor;
@group(0) @binding(1) var<storage, read> Chirp: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.len { return; }
  let idx = params.offset + i;
  if idx >= params.total { return; }
  var re = 0.0; var im = 0.0;
  if idx < params.target_len {
    let b = idx * 2u;
    re = Chirp.data[b];
    im = -Chirp.data[b+1u];
  } else if idx > params.m_len - params.target_len {
    let k = params.m_len - idx;
    let b = k * 2u;
    re = Chirp.data[b];
    im = -Chirp.data[b+1u];
  }
  let d = idx * 2u;
  Output.data[d] = re;
  Output.data[d+1u] = im;
}
"#;

pub const FFT_BLUESTEIN_KERNEL_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params { len:u32, offset:u32, total:u32, target_len:u32, m_len:u32, inverse:u32, _pad0:u32, _pad1:u32, };
@group(0) @binding(0) var<storage, read_write> Output: Tensor;
@group(0) @binding(1) var<storage, read> Chirp: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.len { return; }
  let idx = params.offset + i;
  if idx >= params.total { return; }
  var re = 0.0; var im = 0.0;
  if idx < params.target_len {
    let b = idx * 2u;
    re = Chirp.data[b];
    im = -Chirp.data[b+1u];
  } else if idx > params.m_len - params.target_len {
    let k = params.m_len - idx;
    let b = k * 2u;
    re = Chirp.data[b];
    im = -Chirp.data[b+1u];
  }
  let d = idx * 2u;
  Output.data[d] = re;
  Output.data[d+1u] = im;
}
"#;

pub const FFT_POINTWISE_BROADCAST_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params { len:u32, offset:u32, total:u32, m_len:u32, inner_stride:u32, _pad0:u32, _pad1:u32, _pad2:u32, };
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.len { return; }
  let idx = params.offset + i;
  if idx >= params.total { return; }
  let p = (idx / params.inner_stride) % params.m_len;
  let ab = idx * 2u;
  let bb = p * 2u;
  let ar = A.data[ab]; let ai = A.data[ab+1u];
  let br = B.data[bb]; let bi = B.data[bb+1u];
  Out.data[ab] = ar*br - ai*bi;
  Out.data[ab+1u] = ar*bi + ai*br;
}
"#;

pub const FFT_POINTWISE_BROADCAST_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params { len:u32, offset:u32, total:u32, m_len:u32, inner_stride:u32, _pad0:u32, _pad1:u32, _pad2:u32, };
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read> B: Tensor;
@group(0) @binding(2) var<storage, read_write> Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.len { return; }
  let idx = params.offset + i;
  if idx >= params.total { return; }
  let p = (idx / params.inner_stride) % params.m_len;
  let ab = idx * 2u;
  let bb = p * 2u;
  let ar = A.data[ab]; let ai = A.data[ab+1u];
  let br = B.data[bb]; let bi = B.data[bb+1u];
  Out.data[ab] = ar*br - ai*bi;
  Out.data[ab+1u] = ar*bi + ai*br;
}
"#;

pub const FFT_BLUESTEIN_FINALIZE_SHADER_F64: &str = r#"
struct Tensor { data: array<f64>, };
struct Params { len:u32, offset:u32, total:u32, target_len:u32, m_len:u32, inner_stride:u32, inverse:u32, _pad0:u32, };
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> Chirp: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.len { return; }
  let idx = params.offset + i;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  let src = outer * params.m_len * params.inner_stride + inner + p * params.inner_stride;
  let ib = src * 2u;
  let ob = idx * 2u;
  let cb = p * 2u;
  let ar = Input.data[ib]; let ai = Input.data[ib+1u];
  let br = Chirp.data[cb]; let bi = Chirp.data[cb+1u];
  var rr = ar*br - ai*bi;
  var ri = ar*bi + ai*br;
  if params.inverse != 0u {
    let s = 1.0 / f64(params.target_len);
    rr = rr * s; ri = ri * s;
  }
  Output.data[ob] = rr;
  Output.data[ob+1u] = ri;
}
"#;

pub const FFT_BLUESTEIN_FINALIZE_SHADER_F32: &str = r#"
struct Tensor { data: array<f32>, };
struct Params { len:u32, offset:u32, total:u32, target_len:u32, m_len:u32, inner_stride:u32, inverse:u32, _pad0:u32, };
@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<storage, read> Chirp: Tensor;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if i >= params.len { return; }
  let idx = params.offset + i;
  if idx >= params.total { return; }
  let inner = idx % params.inner_stride;
  let tmp = idx / params.inner_stride;
  let p = tmp % params.target_len;
  let outer = tmp / params.target_len;
  let src = outer * params.m_len * params.inner_stride + inner + p * params.inner_stride;
  let ib = src * 2u;
  let ob = idx * 2u;
  let cb = p * 2u;
  let ar = Input.data[ib]; let ai = Input.data[ib+1u];
  let br = Chirp.data[cb]; let bi = Chirp.data[cb+1u];
  var rr = ar*br - ai*bi;
  var ri = ar*bi + ai*br;
  if params.inverse != 0u {
    let s = 1.0 / f32(params.target_len);
    rr = rr * s; ri = ri * s;
  }
  Output.data[ob] = rr;
  Output.data[ob+1u] = ri;
}
"#;
