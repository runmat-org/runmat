pub fn build_two_pass_shaders(scalar_ty: &str, wg: u32) -> (String, String) {
    let zero = if scalar_ty == "f64" {
        "f64(0.0)"
    } else {
        "0.0"
    };
    let cast = if scalar_ty == "f64" { "f64(" } else { "" };
    let half = wg / 2;
    let pass1 = format!(
        "struct Tensor {{ data: array<{st}> }};\nstruct P1Params {{ nrows:u32,ncols:u32,ld:u32,flags:u32,chunks:u32 }}\n@group(0) @binding(0) var<storage,read> input0: Tensor;\n@group(0) @binding(1) var<storage,read_write> partials: Tensor;\n@group(0) @binding(2) var<uniform> params: P1Params;\nvar<workgroup> tile: array<f32,{wg}>;\nfn isNan(x: {st}) -> bool {{ return x != x; }}\n@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n  let slice = wid.x; let chunk = wid.y;\n  if (slice >= params.ncols || chunk >= params.chunks) {{ return; }}\n  let start = chunk * {wg}u; let end = min(params.nrows, start + {wg}u);\n  var acc: {st} = {zero};\n  var i = start + lid.x;\n  loop {{ if (i >= end) {{ break; }} let v = input0.data[(slice * params.ld) + i]; if ((params.flags & 1u)==1u) {{ if (!isNan(v)) {{ acc = acc + v; }} }} else {{ if (isNan(v)) {{ acc = v; }} else {{ acc = acc + v; }} }} i += {wg}u; }}\n  tile[lid.x] = acc; workgroupBarrier();\n  var off: u32 = {half}u; loop {{ if (off==0u) {{ break; }} if (lid.x < off) {{ let a = tile[lid.x]; let b = tile[lid.x+off]; tile[lid.x] = a + b; }} workgroupBarrier(); off = off/2u; }}\n  if (lid.x==0u) {{ partials.data[(slice * params.chunks) + chunk] = {cast}tile[0u]; }}\n}}",
        st = scalar_ty,
        wg = wg,
        zero = zero,
        half = half,
        cast = cast
    );
    let pass2 = format!(
        "struct Tensor {{ data: array<{st}> }};\nstruct P2Params {{ ncols:u32,chunks:u32,flags:u32 }}\n@group(0) @binding(0) var<storage,read> partials: Tensor;\n@group(0) @binding(1) var<storage,read_write> output: Tensor;\n@group(0) @binding(2) var<uniform> params: P2Params;\nvar<workgroup> tile: array<f32,{wg}>;\nfn isNan(x: {st}) -> bool {{ return x != x; }}\n@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n  let slice = wid.x; if (slice >= params.ncols) {{ return; }}\n  var acc: {st} = {zero}; var c = lid.x;\n  loop {{ if (c >= params.chunks) {{ break; }} let v = partials.data[(slice * params.chunks) + c]; if ((params.flags & 1u)==1u) {{ if (!isNan(v)) {{ acc = acc + v; }} }} else {{ if (isNan(v)) {{ acc = v; }} else {{ acc = acc + v; }} }} c += {wg}u; }}\n  tile[lid.x] = acc; workgroupBarrier();\n  var off: u32 = {half}u; loop {{ if (off==0u) {{ break; }} if (lid.x < off) {{ let a = tile[lid.x]; let b = tile[lid.x+off]; tile[lid.x] = a + b; }} workgroupBarrier(); off = off/2u; }}\n  if (lid.x==0u) {{ output.data[slice] = {cast}tile[0u]; }}\n}}",
        st = scalar_ty,
        wg = wg,
        zero = zero,
        half = half,
        cast = cast
    );
    (pass1, pass2)
}

pub const REDUCE_GLOBAL_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<f64, 256>;

fn combine(a: f64, b: f64, op: u32) -> f64 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { if b < a { return b; } return a; }
        case 2u: { if b > a { return b; } return a; }
        default: { return a; }
    }
}

fn identity(op: u32) -> f64 {
    switch op {
        case 0u: { return 0.0; }
        case 1u: { return f64(1.0) / f64(0.0); }
        case 2u: { return -f64(1.0) / f64(0.0); }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let base = wid.x * 512u;
    let idx = base + lid.x;
    var acc = identity(params.op);
    if idx < params.len {
        acc = InBuf.data[idx];
    }
    if idx + 256u < params.len {
        acc = combine(acc, InBuf.data[idx + 256u], params.op);
    }
    tile[lid.x] = acc;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if lid.x < stride { tile[lid.x] = combine(tile[lid.x], tile[lid.x + stride], params.op); }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if lid.x == 0u { OutBuf.data[wid.x] = tile[0u]; }
}
"#;

pub const REDUCE_GLOBAL_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    len: u32,
    op: u32,
    offset: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<f32, 256>;

fn combine(a: f32, b: f32, op: u32) -> f32 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { return select(a, b, b < a); }
        case 2u: { return select(a, b, b > a); }
        default: { return a; }
    }
}

fn identity(op: u32) -> f32 {
    switch op {
        case 0u: { return 0.0f; }
        case 1u: { return 1.0f / 0.0f; }
        case 2u: { return -1.0f / 0.0f; }
        default: { return 0.0f; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let base = wid.x * 512u;
    let idx = base + lid.x;
    var acc = identity(params.op);
    if idx < params.len {
        acc = InBuf.data[idx];
    }
    if idx + 256u < params.len {
        acc = combine(acc, InBuf.data[idx + 256u], params.op);
    }
    tile[lid.x] = acc;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if lid.x < stride { tile[lid.x] = combine(tile[lid.x], tile[lid.x + stride], params.op); }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if lid.x == 0u { OutBuf.data[wid.x] = tile[0u]; }
}
"#;

pub const REDUCE_DIM_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f64) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols { return; }
        var acc: f64 = 0.0;
        var saw_nan: bool = false;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) { saw_nan = true; } else { acc = acc + v; }
        }
        if saw_nan { OutBuf.data[idx] = f64(0.0) / f64(0.0); }
        else { if params.op == 1u { acc = acc / f64(params.rows); } OutBuf.data[idx] = acc; }
    } else {
        if idx >= params.rows { return; }
        var acc: f64 = 0.0;
        var saw_nan: bool = false;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) { saw_nan = true; } else { acc = acc + v; }
        }
        if saw_nan { OutBuf.data[idx] = f64(0.0) / f64(0.0); }
        else { if params.op == 1u { acc = acc / f64(params.cols); } OutBuf.data[idx] = acc; }
    }
}
"#;

pub const REDUCE_DIM_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

fn isNan(x: f32) -> bool { return x != x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols { return; }
        var acc: f32 = 0.0;
        var saw_nan: bool = false;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) { saw_nan = true; } else { acc = acc + v; }
        }
        if saw_nan { OutBuf.data[idx] = f32(0.0) / f32(0.0); }
        else { if params.op == 1u { acc = acc / f32(params.rows); } OutBuf.data[idx] = acc; }
    } else {
        if idx >= params.rows { return; }
        var acc: f32 = 0.0;
        var saw_nan: bool = false;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) { saw_nan = true; } else { acc = acc + v; }
        }
        if saw_nan { OutBuf.data[idx] = f32(0.0) / f32(0.0); }
        else { if params.op == 1u { acc = acc / f32(params.cols); } OutBuf.data[idx] = acc; }
    }
}
"#;

pub const REDUCE_DIM_MINMAX_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutVals: Tensor;
@group(0) @binding(2) var<storage, read_write> OutIdx: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn better(current: f64, candidate: f64, op: u32) -> bool {
    if op == 0u { return candidate < current; }
    return candidate > current;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols { return; }
        var best: f64;
        if params.op == 0u { best = f64(1.0) / f64(0.0); } else { best = -f64(1.0) / f64(0.0); }
        var best_idx: u32 = 1u;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let value = InBuf.data[linear];
            if r == 0u || better(best, value, params.op) { best = value; best_idx = r + 1u; }
        }
        OutVals.data[idx] = best;
        OutIdx.data[idx] = f64(best_idx);
    } else {
        if idx >= params.rows { return; }
        var best: f64;
        if params.op == 0u { best = f64(1.0) / f64(0.0); } else { best = -f64(1.0) / f64(0.0); }
        var best_idx: u32 = 1u;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let value = InBuf.data[linear];
            if c == 0u || better(best, value, params.op) { best = value; best_idx = c + 1u; }
        }
        OutVals.data[idx] = best;
        OutIdx.data[idx] = f64(best_idx);
    }
}
"#;

pub const REDUCE_DIM_MINMAX_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    rows: u32,
    cols: u32,
    dim: u32,
    op: u32,
};

@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutVals: Tensor;
@group(0) @binding(2) var<storage, read_write> OutIdx: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

fn better(current: f32, candidate: f32, op: u32) -> bool {
    if op == 0u { return candidate < current; }
    return candidate > current;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols { return; }
        var best: f32;
        if params.op == 0u { best = 1.0f / 0.0f; } else { best = -1.0f / 0.0f; }
        var best_idx: u32 = 1u;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let value = InBuf.data[linear];
            if r == 0u || better(best, value, params.op) { best = value; best_idx = r + 1u; }
        }
        OutVals.data[idx] = best;
        OutIdx.data[idx] = f32(best_idx);
    } else {
        if idx >= params.rows { return; }
        var best: f32;
        if params.op == 0u { best = 1.0f / 0.0f; } else { best = -1.0f / 0.0f; }
        var best_idx: u32 = 1u;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let value = InBuf.data[linear];
            if c == 0u || better(best, value, params.op) { best = value; best_idx = c + 1u; }
        }
        OutVals.data[idx] = best;
        OutIdx.data[idx] = f32(best_idx);
    }
}
"#;
