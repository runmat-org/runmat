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

var<workgroup> tile: array<f64, @WG@>;

fn combine(a: f64, b: f64, op: u32) -> f64 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { return a * b; }
        case 2u: { if b < a { return b; } return a; }
        case 3u: { if b > a { return b; } return a; }
        case 4u: { return a + b; }
        default: { return a; }
    }
}

fn identity(op: u32) -> f64 {
    switch op {
        case 0u: { return 0.0; }
        case 1u: { return 1.0; }
        case 2u: { return f64(1.0) / f64(0.0); }
        case 3u: { return -f64(1.0) / f64(0.0); }
        case 4u: { return 0.0; }
        default: { return 0.0; }
    }
}

fn map_value(v: f64, op: u32) -> f64 {
    if op == 4u {
        if v != v { return 1.0; }
        if v != 0.0 { return 1.0; }
        return 0.0;
    }
    return v;
}

@compute @workgroup_size(@WG@)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let base = params.offset + wid.x * 512u;
    let idx = base + lid.x;
    var acc = identity(params.op);
    if idx < params.len {
        let mapped = map_value(InBuf.data[idx], params.op);
        acc = combine(acc, mapped, params.op);
    }
    if idx + 256u < params.len {
        let mapped = map_value(InBuf.data[idx + 256u], params.op);
        acc = combine(acc, mapped, params.op);
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
    if lid.x == 0u { let out_index = (params.offset / 512u) + wid.x; OutBuf.data[out_index] = tile[0u]; }
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

var<workgroup> tile: array<f32, @WG@>;

fn combine(a: f32, b: f32, op: u32) -> f32 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { return a * b; }
        case 2u: { return select(a, b, b < a); }
        case 3u: { return select(a, b, b > a); }
        case 4u: { return a + b; }
        default: { return a; }
    }
}

fn identity(op: u32) -> f32 {
    switch op {
        case 0u: { return 0.0f; }
        case 1u: { return 1.0f; }
        case 2u: { return 1.0f / 0.0f; }
        case 3u: { return -1.0f / 0.0f; }
        case 4u: { return 0.0f; }
        default: { return 0.0f; }
    }
}

fn map_value(v: f32, op: u32) -> f32 {
    if op == 4u {
        if v != v { return 1.0f; }
        if v != 0.0f { return 1.0f; }
        return 0.0f;
    }
    return v;
}

@compute @workgroup_size(@WG@)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let base = params.offset + wid.x * 512u;
    let idx = base + lid.x;
    var acc = identity(params.op);
    if idx < params.len {
        let mapped = map_value(InBuf.data[idx], params.op);
        acc = combine(acc, mapped, params.op);
    }
    if idx + 256u < params.len {
        let mapped = map_value(InBuf.data[idx + 256u], params.op);
        acc = combine(acc, mapped, params.op);
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
    if lid.x == 0u { let out_index = (params.offset / 512u) + wid.x; OutBuf.data[out_index] = tile[0u]; }
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

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let op_any_include = params.op == 3u;
    let op_any_omit = params.op == 4u;
    let op_all_include = params.op == 5u;
    let op_all_omit = params.op == 6u;
    let op_nnz = params.op == 7u;
    let is_any = op_any_include || op_any_omit;
    let is_all = op_all_include || op_all_omit;
    if params.dim == 1u {
        if idx >= params.cols { return; }
        if op_nnz {
            var count: f64 = 0.0;
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                let linear = r + idx * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) || v != 0.0 {
                    count = count + 1.0;
                }
            }
            OutBuf.data[idx] = count;
            return;
        }
        if is_any {
            var any_true: bool = false;
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                let linear = r + idx * params.rows;
                let v = InBuf.data[linear];
                if op_any_include {
                    if isNan(v) || v != 0.0 {
                        any_true = true;
                        break;
                    }
                } else {
                    if (!isNan(v)) && v != 0.0 {
                        any_true = true;
                        break;
                    }
                }
            }
            if any_true { OutBuf.data[idx] = 1.0; } else { OutBuf.data[idx] = 0.0; }
            return;
        }
        if is_all {
            var all_true: bool = true;
            var saw_value: bool = false;
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                let linear = r + idx * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) {
                    if op_all_omit {
                        continue;
                    } else {
                        continue;
                    }
                }
                saw_value = true;
                if v == 0.0 {
                    all_true = false;
                    break;
                }
            }
            if op_all_omit && !saw_value {
                all_true = true;
            }
            if all_true { OutBuf.data[idx] = 1.0; } else { OutBuf.data[idx] = 0.0; }
            return;
        }
        var acc: f64 = 0.0;
        if params.op == 2u { acc = 1.0; }
        var saw_nan: bool = false;
        var any: bool = false;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) {
                saw_nan = true;
            } else {
                any = true;
                if params.op == 2u { acc = acc * v; }
                else { acc = acc + v; }
            }
        }
        if saw_nan {
            OutBuf.data[idx] = f64(0.0) / f64(0.0);
        } else if params.op == 1u {
            OutBuf.data[idx] = acc / f64(params.rows);
        } else if params.op == 2u {
            if (!any) { OutBuf.data[idx] = 1.0; }
            else { OutBuf.data[idx] = acc; }
        } else {
            OutBuf.data[idx] = acc;
        }
    } else {
        if idx >= params.rows { return; }
        if op_nnz {
            var count: f64 = 0.0;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                let linear = idx + c * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) || v != 0.0 {
                    count = count + 1.0;
                }
            }
            OutBuf.data[idx] = count;
            return;
        }
        if is_any {
            var any_true: bool = false;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                let linear = idx + c * params.rows;
                let v = InBuf.data[linear];
                if op_any_include {
                    if isNan(v) || v != 0.0 {
                        any_true = true;
                        break;
                    }
                } else {
                    if (!isNan(v)) && v != 0.0 {
                        any_true = true;
                        break;
                    }
                }
            }
            if any_true { OutBuf.data[idx] = 1.0; } else { OutBuf.data[idx] = 0.0; }
            return;
        }
        if is_all {
            var all_true: bool = true;
            var saw_value: bool = false;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                let linear = idx + c * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) {
                    if op_all_omit {
                        continue;
                    } else {
                        continue;
                    }
                }
                saw_value = true;
                if v == 0.0 {
                    all_true = false;
                    break;
                }
            }
            if op_all_omit && !saw_value {
                all_true = true;
            }
            if all_true { OutBuf.data[idx] = 1.0; } else { OutBuf.data[idx] = 0.0; }
            return;
        }
        var acc: f64 = 0.0;
        if params.op == 2u { acc = 1.0; }
        var saw_nan: bool = false;
        var any: bool = false;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) {
                saw_nan = true;
            } else {
                any = true;
                if params.op == 2u { acc = acc * v; }
                else { acc = acc + v; }
            }
        }
        if saw_nan {
            OutBuf.data[idx] = f64(0.0) / f64(0.0);
        } else if params.op == 1u {
            OutBuf.data[idx] = acc / f64(params.cols);
        } else if params.op == 2u {
            if (!any) { OutBuf.data[idx] = 1.0; }
            else { OutBuf.data[idx] = acc; }
        } else {
            OutBuf.data[idx] = acc;
        }
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

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let op_any_include = params.op == 3u;
    let op_any_omit = params.op == 4u;
    let op_all_include = params.op == 5u;
    let op_all_omit = params.op == 6u;
    let op_nnz = params.op == 7u;
    let is_any = op_any_include || op_any_omit;
    let is_all = op_all_include || op_all_omit;
    if params.dim == 1u {
        if idx >= params.cols { return; }
        if op_nnz {
            var count: f32 = 0.0f;
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                let linear = r + idx * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) || v != 0.0f {
                    count = count + 1.0f;
                }
            }
            OutBuf.data[idx] = count;
            return;
        }
        if is_any {
            var any_true: bool = false;
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                let linear = r + idx * params.rows;
                let v = InBuf.data[linear];
                if op_any_include {
                    if isNan(v) || v != 0.0f {
                        any_true = true;
                        break;
                    }
                } else {
                    if (!isNan(v)) && v != 0.0f {
                        any_true = true;
                        break;
                    }
                }
            }
            if any_true { OutBuf.data[idx] = 1.0f; } else { OutBuf.data[idx] = 0.0f; }
            return;
        }
        if is_all {
            var all_true: bool = true;
            var saw_value: bool = false;
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                let linear = r + idx * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) {
                    if op_all_omit {
                        continue;
                    } else {
                        continue;
                    }
                }
                saw_value = true;
                if v == 0.0f {
                    all_true = false;
                    break;
                }
            }
            if op_all_omit && !saw_value {
                all_true = true;
            }
            if all_true { OutBuf.data[idx] = 1.0f; } else { OutBuf.data[idx] = 0.0f; }
            return;
        }
        var acc: f32 = 0.0f;
        if params.op == 2u { acc = 1.0f; }
        var saw_nan: bool = false;
        var any: bool = false;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            let linear = r + idx * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) {
                saw_nan = true;
            } else {
                any = true;
                if params.op == 2u { acc = acc * v; }
                else { acc = acc + v; }
            }
        }
        if saw_nan {
            OutBuf.data[idx] = f32(0.0) / f32(0.0);
        } else if params.op == 1u {
            OutBuf.data[idx] = acc / f32(params.rows);
        } else if params.op == 2u {
            if (!any) { OutBuf.data[idx] = 1.0f; }
            else { OutBuf.data[idx] = acc; }
        } else {
            OutBuf.data[idx] = acc;
        }
    } else {
        if idx >= params.rows { return; }
        if op_nnz {
            var count: f32 = 0.0f;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                let linear = idx + c * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) || v != 0.0f {
                    count = count + 1.0f;
                }
            }
            OutBuf.data[idx] = count;
            return;
        }
        if is_any {
            var any_true: bool = false;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                let linear = idx + c * params.rows;
                let v = InBuf.data[linear];
                if op_any_include {
                    if isNan(v) || v != 0.0f {
                        any_true = true;
                        break;
                    }
                } else {
                    if (!isNan(v)) && v != 0.0f {
                        any_true = true;
                        break;
                    }
                }
            }
            if any_true { OutBuf.data[idx] = 1.0f; } else { OutBuf.data[idx] = 0.0f; }
            return;
        }
        if is_all {
            var all_true: bool = true;
            var saw_value: bool = false;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                let linear = idx + c * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) {
                    if op_all_omit {
                        continue;
                    } else {
                        continue;
                    }
                }
                saw_value = true;
                if v == 0.0f {
                    all_true = false;
                    break;
                }
            }
            if op_all_omit && !saw_value {
                all_true = true;
            }
            if all_true { OutBuf.data[idx] = 1.0f; } else { OutBuf.data[idx] = 0.0f; }
            return;
        }
        var acc: f32 = 0.0f;
        if params.op == 2u { acc = 1.0f; }
        var saw_nan: bool = false;
        var any: bool = false;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let linear = idx + c * params.rows;
            let v = InBuf.data[linear];
            if isNan(v) {
                saw_nan = true;
            } else {
                any = true;
                if params.op == 2u { acc = acc * v; }
                else { acc = acc + v; }
            }
        }
        if saw_nan {
            OutBuf.data[idx] = f32(0.0) / f32(0.0);
        } else if params.op == 1u {
            OutBuf.data[idx] = acc / f32(params.cols);
        } else if params.op == 2u {
            if (!any) { OutBuf.data[idx] = 1.0f; }
            else { OutBuf.data[idx] = acc; }
        } else {
            OutBuf.data[idx] = acc;
        }
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

@compute @workgroup_size(@WG@)
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

@compute @workgroup_size(@WG@)
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

pub const REDUCE_ND_MEAN_SHADER_F64: &str = r#"
const MAX_RANK: u32 = 128u;
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor { data: array<f64>, };
struct Params {
  rank: u32,
  kept_count: u32,
  reduce_count: u32,
  _pad: u32,
  rows: u32,
  cols: u32,
  _pad2: vec2<u32>,
  kept_sizes: PackedArray,
  reduce_sizes: PackedArray,
  kept_strides: PackedArray,
  reduce_strides: PackedArray,
};
@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<f64, @WG@u>;

fn map_col_to_base(col: u32) -> u32 {
  var rem = col;
  var base: u32 = 0u;
  for (var j: u32 = 0u; j < params.kept_count; j = j + 1u) {
    let size = params.kept_sizes[j].value;
    if (size == 0u) { continue; }
    let coord = rem % size;
    rem = rem / size;
    base = base + coord * params.kept_strides[j].value;
  }
  return base;
}

fn map_row_offset(r: u32) -> u32 {
  var rem = r;
  var off: u32 = 0u;
  for (var j: u32 = 0u; j < params.reduce_count; j = j + 1u) {
    let size = params.reduce_sizes[j].value;
    if (size == 0u) { continue; }
    let coord = rem % size;
    rem = rem / size;
    off = off + coord * params.reduce_strides[j].value;
  }
  return off;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let col = wid.x;
  if (col >= params.cols) { return; }
  let base = map_col_to_base(col);
  var acc: f64 = 0.0;
  var r = lid.x;
  while (r < params.rows) {
    let idx = base + map_row_offset(r);
    acc = acc + InBuf.data[idx];
    r = r + 512u;
  }
  tile[lid.x] = acc;
  workgroupBarrier();
  var off = 256u;
  loop {
    if (off == 0u) { break; }
    if (lid.x < off) { tile[lid.x] = tile[lid.x] + tile[lid.x + off]; }
    workgroupBarrier();
    off = off / 2u;
  }
  if (lid.x == 0u) {
    let count = max(params.rows, 1u);
    OutBuf.data[col] = tile[0u] / f64(count);
  }
}
"#;

pub const REDUCE_ND_MEAN_SHADER_F32: &str = r#"
const MAX_RANK: u32 = 128u;
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor { data: array<f32>, };
struct Params {
  rank: u32,
  kept_count: u32,
  reduce_count: u32,
  _pad: u32,
  rows: u32,
  cols: u32,
  _pad2: vec2<u32>,
  kept_sizes: PackedArray,
  reduce_sizes: PackedArray,
  kept_strides: PackedArray,
  reduce_strides: PackedArray,
};
@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> OutBuf: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> tile: array<f32, @WG@u>;

fn map_col_to_base(col: u32) -> u32 {
  var rem = col;
  var base: u32 = 0u;
  for (var j: u32 = 0u; j < params.kept_count; j = j + 1u) {
    let size = params.kept_sizes[j].value;
    if (size == 0u) { continue; }
    let coord = rem % size;
    rem = rem / size;
    base = base + coord * params.kept_strides[j].value;
  }
  return base;
}

fn map_row_offset(r: u32) -> u32 {
  var rem = r;
  var off: u32 = 0u;
  for (var j: u32 = 0u; j < params.reduce_count; j = j + 1u) {
    let size = params.reduce_sizes[j].value;
    if (size == 0u) { continue; }
    let coord = rem % size;
    rem = rem / size;
    off = off + coord * params.reduce_strides[j].value;
  }
  return off;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let col = wid.x;
  if (col >= params.cols) { return; }
  let base = map_col_to_base(col);
  var acc: f32 = 0.0f;
  var r = lid.x;
  while (r < params.rows) {
    let idx = base + map_row_offset(r);
    acc = acc + InBuf.data[idx];
    r = r + 512u;
  }
  tile[lid.x] = acc;
  workgroupBarrier();
  var off = 256u;
  loop {
    if (off == 0u) { break; }
    if (lid.x < off) { tile[lid.x] = tile[lid.x] + tile[lid.x + off]; }
    workgroupBarrier();
    off = off / 2u;
  }
  if (lid.x == 0u) {
    let count = max(params.rows, 1u);
    OutBuf.data[col] = tile[0u] / f32(count);
  }
}
"#;

pub const REDUCE_ND_MOMENTS_SHADER_F64: &str = r#"
const MAX_RANK: u32 = 128u;
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor { data: array<f64>, };
struct Params {
  rank: u32,
  kept_count: u32,
  reduce_count: u32,
  _pad: u32,
  rows: u32,
  cols: u32,
  _pad2: vec2<u32>,
  kept_sizes: PackedArray,
  reduce_sizes: PackedArray,
  kept_strides: PackedArray,
  reduce_strides: PackedArray,
};
@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> MeanOut: Tensor;
@group(0) @binding(2) var<storage, read_write> Ex2Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_sum: array<f64, @WG@u>;
var<workgroup> tile_sumsq: array<f64, @WG@u>;

fn map_col_to_base(col: u32) -> u32 {
  var rem = col;
  var base: u32 = 0u;
  for (var j: u32 = 0u; j < params.kept_count; j = j + 1u) {
    let size = params.kept_sizes[j].value;
    if (size == 0u) { continue; }
    let coord = rem % size;
    rem = rem / size;
    base = base + coord * params.kept_strides[j].value;
  }
  return base;
}

fn map_row_offset(r: u32) -> u32 {
  var rem = r;
  var off: u32 = 0u;
  for (var j: u32 = 0u; j < params.reduce_count; j = j + 1u) {
    let size = params.reduce_sizes[j].value;
    if (size == 0u) { continue; }
    let coord = rem % size;
    rem = rem / size;
    off = off + coord * params.reduce_strides[j].value;
  }
  return off;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let col = wid.x;
  if (col >= params.cols) { return; }
  let base = map_col_to_base(col);
  var acc: f64 = 0.0;
  var acc2: f64 = 0.0;
  var r = lid.x;
  while (r < params.rows) {
    let idx = base + map_row_offset(r);
    let v = InBuf.data[idx];
    acc = acc + v;
    acc2 = acc2 + v * v;
    r = r + 512u;
  }
  tile_sum[lid.x] = acc;
  tile_sumsq[lid.x] = acc2;
  workgroupBarrier();
  var off = 256u;
  loop {
    if (off == 0u) { break; }
    if (lid.x < off) {
      tile_sum[lid.x] = tile_sum[lid.x] + tile_sum[lid.x + off];
      tile_sumsq[lid.x] = tile_sumsq[lid.x] + tile_sumsq[lid.x + off];
    }
    workgroupBarrier();
    off = off / 2u;
  }
  if (lid.x == 0u) {
    let count = max(params.rows, 1u);
    let denom = f64(count);
    MeanOut.data[col] = tile_sum[0u] / denom;
    Ex2Out.data[col] = tile_sumsq[0u] / denom;
  }
}
"#;

pub const REDUCE_ND_MOMENTS_SHADER_F32: &str = r#"
const MAX_RANK: u32 = 128u;
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
alias PackedArray = array<PackedValue, MAX_RANK>;

struct Tensor { data: array<f32>, };
struct Params {
  rank: u32,
  kept_count: u32,
  reduce_count: u32,
  _pad: u32,
  rows: u32,
  cols: u32,
  _pad2: vec2<u32>,
  kept_sizes: PackedArray,
  reduce_sizes: PackedArray,
  kept_strides: PackedArray,
  reduce_strides: PackedArray,
};
@group(0) @binding(0) var<storage, read> InBuf: Tensor;
@group(0) @binding(1) var<storage, read_write> MeanOut: Tensor;
@group(0) @binding(2) var<storage, read_write> Ex2Out: Tensor;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_sum: array<f32, @WG@u>;
var<workgroup> tile_sumsq: array<f32, @WG@u>;

fn map_col_to_base(col: u32) -> u32 {
  var rem = col;
  var base: u32 = 0u;
  for (var j: u32 = 0u; j < params.kept_count; j = j + 1u) {
    let size = params.kept_sizes[j].value;
    if (size == 0u) { continue; }
    let coord = rem % size;
    rem = rem / size;
    base = base + coord * params.kept_strides[j].value;
  }
  return base;
}

fn map_row_offset(r: u32) -> u32 {
  var rem = r;
  var off: u32 = 0u;
  for (var j: u32 = 0u; j < params.reduce_count; j = j + 1u) {
    let size = params.reduce_sizes[j].value;
    if (size == 0u) { continue; }
    let coord = rem % size;
    rem = rem / size;
    off = off + coord * params.reduce_strides[j].value;
  }
  return off;
}

@compute @workgroup_size(@WG@)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let col = wid.x;
  if (col >= params.cols) { return; }
  let base = map_col_to_base(col);
  var acc: f32 = 0.0f;
  var acc2: f32 = 0.0f;
  var r = lid.x;
  while (r < params.rows) {
    let idx = base + map_row_offset(r);
    let v = InBuf.data[idx];
    acc = acc + v;
    acc2 = acc2 + v * v;
    r = r + 512u;
  }
  tile_sum[lid.x] = acc;
  tile_sumsq[lid.x] = acc2;
  workgroupBarrier();
  var off = 256u;
  loop {
    if (off == 0u) { break; }
    if (lid.x < off) {
      tile_sum[lid.x] = tile_sum[lid.x] + tile_sum[lid.x + off];
      tile_sumsq[lid.x] = tile_sumsq[lid.x] + tile_sumsq[lid.x + off];
    }
    workgroupBarrier();
    off = off / 2u;
  }
  if (lid.x == 0u) {
    let count = max(params.rows, 1u);
    let denom = f32(count);
    MeanOut.data[col] = tile_sum[0u] / denom;
    Ex2Out.data[col] = tile_sumsq[0u] / denom;
  }
}
"#;
