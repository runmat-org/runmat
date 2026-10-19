#[path = "reduction/ordered_mean.rs"]
pub(crate) mod ordered_mean;

#[path = "reduction/global_extrema.rs"]
mod global_extrema;

#[cfg(test)]
#[path = "reduction/two_pass_tests.rs"]
mod two_pass_tests;

#[cfg(test)]
#[path = "reduction/f64_special_tests.rs"]
mod f64_special_tests;

use crate::backend::wgpu::types::NumericPrecision;

/// The global reduction uses the same IEEE Max comparator as ordinary
/// elementwise Max, with a matching Min comparator. The selected shader is
/// complete before pipeline creation; no caller can select the raw template.
pub(crate) fn global_source(precision: NumericPrecision) -> String {
    global_extrema::source(precision)
}

pub(crate) fn two_pass_tile_bytes(precision: NumericPrecision, workgroup_size: u32) -> Option<u64> {
    let element_bytes = match precision {
        NumericPrecision::F32 => 4u64,
        NumericPrecision::F64 => 8u64,
    };
    u64::from(workgroup_size).checked_mul(element_bytes)
}

pub fn build_two_pass_shaders(precision: NumericPrecision, wg: u32) -> (String, String) {
    let scalar_ty = match precision {
        NumericPrecision::F32 => "f32",
        NumericPrecision::F64 => "f64",
    };
    let zero = match precision {
        NumericPrecision::F32 => "0.0",
        NumericPrecision::F64 => "f64(0.0)",
    };
    let half = wg / 2;
    let pass1 = format!(
        "struct Tensor {{ data: array<{st}> }};\nstruct P1Params {{ nrows:u32,ncols:u32,ld:u32,flags:u32,chunks:u32,chunk_stride:u32,chunk_rows:u32,_pad:u32 }}\n@group(0) @binding(0) var<storage,read> input0: Tensor;\n@group(0) @binding(1) var<storage,read_write> partials: Tensor;\n@group(0) @binding(2) var<uniform> params: P1Params;\nvar<workgroup> tile: array<{st},{wg}>;\nfn isNan(value: {st}) -> bool {{ @IEEE_NAN_PREDICATE@ }}\n@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n  let slice = wid.x;\n  let chunk_tile = wid.z;\n  let chunk = chunk_tile * params.chunk_stride + wid.y;\n  if (slice >= params.ncols || chunk >= params.chunks) {{ return; }}\n  let start = chunk * params.chunk_rows;\n  let end = min(params.nrows, start + params.chunk_rows);\n  var acc: {st} = {zero};\n  var i = start + lid.x;\n  loop {{ if (i >= end) {{ break; }} let v = input0.data[(slice * params.ld) + i]; if ((params.flags & 1u)==1u) {{ if (!isNan(v)) {{ acc = acc + v; }} }} else {{ if (isNan(v)) {{ acc = v; }} else {{ acc = acc + v; }} }} i += {wg}u; }}\n  tile[lid.x] = acc; workgroupBarrier();\n  var off: u32 = {half}u; loop {{ if (off==0u) {{ break; }} if (lid.x < off) {{ let a = tile[lid.x]; let b = tile[lid.x+off]; tile[lid.x] = a + b; }} workgroupBarrier(); off = off/2u; }}\n  if (lid.x==0u) {{ partials.data[(slice * params.chunks) + chunk] = tile[0u]; }}\n}}",
        st = scalar_ty,
        wg = wg,
        zero = zero,
        half = half
    );
    let pass2 = format!(
        "struct Tensor {{ data: array<{st}> }};\nstruct P2Params {{ ncols:u32,chunks:u32,flags:u32,scale:{st} }}\n@group(0) @binding(0) var<storage,read> partials: Tensor;\n@group(0) @binding(1) var<storage,read_write> output: Tensor;\n@group(0) @binding(2) var<uniform> params: P2Params;\nvar<workgroup> tile: array<{st},{wg}>;\nfn isNan(value: {st}) -> bool {{ @IEEE_NAN_PREDICATE@ }}\n@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n  let slice = wid.x; if (slice >= params.ncols) {{ return; }}\n  var acc: {st} = {zero}; var c = lid.x;\n  loop {{ if (c >= params.chunks) {{ break; }} let v = partials.data[(slice * params.chunks) + c]; if ((params.flags & 1u)==1u) {{ if (!isNan(v)) {{ acc = acc + v; }} }} else {{ if (isNan(v)) {{ acc = v; }} else {{ acc = acc + v; }} }} c += {wg}u; }}\n  tile[lid.x] = acc; workgroupBarrier();\n  var off: u32 = {half}u; loop {{ if (off==0u) {{ break; }} if (lid.x < off) {{ let a = tile[lid.x]; let b = tile[lid.x+off]; tile[lid.x] = a + b; }} workgroupBarrier(); off = off/2u; }}\n  if (lid.x==0u) {{ output.data[slice] = (tile[0u]) * params.scale; }}\n}}",
        st = scalar_ty,
        wg = wg,
        zero = zero,
        half = half
    );
    (
        super::nan_predicate::inject(&pass1, precision),
        super::nan_predicate::inject(&pass2, precision),
    )
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

fn pos_inf_f64() -> f64 {
    return runmat_f64_from_words(0u, 0x7ff00000u);
}

fn neg_inf_f64() -> f64 {
    return runmat_f64_from_words(0u, 0xfff00000u);
}

fn nan_f64() -> f64 {
    return runmat_f64_from_words(0u, 0x7ff80000u);
}

fn combine(a: f64, b: f64, op: u32) -> f64 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { return a * b; }
        case 2u: { return runmat_real_min(a, b); }
        case 3u: { return runmat_real_max(a, b); }
        case 4u: { return a + b; }
        default: { return a; }
    }
}

fn identity(op: u32) -> f64 {
    switch op {
        case 0u: { return f64(0.0); }
        case 1u: { return f64(1.0); }
        case 2u: { return pos_inf_f64(); }
        case 3u: { return neg_inf_f64(); }
        case 4u: { return f64(0.0); }
        default: { return f64(0.0); }
    }
}

fn isNan(value: f64) -> bool { @IEEE_NAN_PREDICATE@ }

fn map_value(v: f64, op: u32) -> f64 {
    if op == 4u {
        if isNan(v) { return f64(1.0); }
        if v != f64(0.0) { return f64(1.0); }
        return f64(0.0);
    }
    return v;
}

@compute @workgroup_size(@WG@)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let wg: u32 = @WG@u;
    let group_stride: u32 = 2u * wg;
    let base = params.offset + wid.x * group_stride;
    let idx = base + lid.x;
    let end = params.offset + params.len;
    var acc = identity(params.op);
    if idx < end {
        let mapped = map_value(InBuf.data[idx], params.op);
        acc = combine(acc, mapped, params.op);
    }
    if idx + wg < end {
        let mapped = map_value(InBuf.data[idx + wg], params.op);
        acc = combine(acc, mapped, params.op);
    }
    tile[lid.x] = acc;
    workgroupBarrier();

    var stride = wg / 2u;
    loop {
        if stride == 0u { break; }
        if lid.x < stride { tile[lid.x] = combine(tile[lid.x], tile[lid.x + stride], params.op); }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if lid.x == 0u {
        let out_index = (params.offset / group_stride) + wid.x;
        OutBuf.data[out_index] = tile[0u];
    }
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

fn pos_inf_f32() -> f32 {
    var bits: u32 = 0x7f800000u;
    return bitcast<f32>(bits);
}

fn neg_inf_f32() -> f32 {
    var bits: u32 = 0xff800000u;
    return bitcast<f32>(bits);
}

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

fn combine(a: f32, b: f32, op: u32) -> f32 {
    switch op {
        case 0u: { return a + b; }
        case 1u: { return a * b; }
        case 2u: { return runmat_real_min(a, b); }
        case 3u: { return runmat_real_max(a, b); }
        case 4u: { return a + b; }
        default: { return a; }
    }
}

fn identity(op: u32) -> f32 {
    switch op {
        case 0u: { return 0.0f; }
        case 1u: { return 1.0f; }
        case 2u: { return pos_inf_f32(); }
        case 3u: { return neg_inf_f32(); }
        case 4u: { return 0.0f; }
        default: { return 0.0f; }
    }
}

fn isNan(value: f32) -> bool { @IEEE_NAN_PREDICATE@ }

fn map_value(v: f32, op: u32) -> f32 {
    if op == 4u {
        if isNan(v) { return 1.0f; }
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
    let wg: u32 = @WG@u;
    let group_stride: u32 = 2u * wg;
    let base = params.offset + wid.x * group_stride;
    let idx = base + lid.x;
    let end = params.offset + params.len;
    var acc = identity(params.op);
    if idx < end {
        let mapped = map_value(InBuf.data[idx], params.op);
        acc = combine(acc, mapped, params.op);
    }
    if idx + wg < end {
        let mapped = map_value(InBuf.data[idx + wg], params.op);
        acc = combine(acc, mapped, params.op);
    }
    tile[lid.x] = acc;
    workgroupBarrier();

    var stride = wg / 2u;
    loop {
        if stride == 0u { break; }
        if lid.x < stride { tile[lid.x] = combine(tile[lid.x], tile[lid.x + stride], params.op); }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if lid.x == 0u {
        let out_index = (params.offset / group_stride) + wid.x;
        OutBuf.data[out_index] = tile[0u];
    }
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

fn isNan(value: f64) -> bool { @IEEE_NAN_PREDICATE@ }

fn nan_f64() -> f64 {
    return runmat_f64_from_words(0u, 0x7ff80000u);
}

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
            var count: f64 = f64(0.0);
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                let linear = r + idx * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) || v != f64(0.0) {
                    count = count + f64(1.0);
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
                    if isNan(v) || v != f64(0.0) {
                        any_true = true;
                        break;
                    }
                } else {
                    if (!isNan(v)) && v != f64(0.0) {
                        any_true = true;
                        break;
                    }
                }
            }
            if any_true { OutBuf.data[idx] = f64(1.0); } else { OutBuf.data[idx] = f64(0.0); }
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
                if v == f64(0.0) {
                    all_true = false;
                    break;
                }
            }
            if op_all_omit && !saw_value {
                all_true = true;
            }
            if all_true { OutBuf.data[idx] = f64(1.0); } else { OutBuf.data[idx] = f64(0.0); }
            return;
        }
        var acc: f64 = f64(0.0);
        if params.op == 2u { acc = f64(1.0); }
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
            OutBuf.data[idx] = nan_f64();
        } else if params.op == 1u {
            OutBuf.data[idx] = acc / f64(params.rows);
        } else if params.op == 2u {
            if (!any) { OutBuf.data[idx] = f64(1.0); }
            else { OutBuf.data[idx] = acc; }
        } else {
            OutBuf.data[idx] = acc;
        }
    } else {
        if idx >= params.rows { return; }
        if op_nnz {
            var count: f64 = f64(0.0);
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                let linear = idx + c * params.rows;
                let v = InBuf.data[linear];
                if isNan(v) || v != f64(0.0) {
                    count = count + f64(1.0);
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
                    if isNan(v) || v != f64(0.0) {
                        any_true = true;
                        break;
                    }
                } else {
                    if (!isNan(v)) && v != f64(0.0) {
                        any_true = true;
                        break;
                    }
                }
            }
            if any_true { OutBuf.data[idx] = f64(1.0); } else { OutBuf.data[idx] = f64(0.0); }
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
                if v == f64(0.0) {
                    all_true = false;
                    break;
                }
            }
            if op_all_omit && !saw_value {
                all_true = true;
            }
            if all_true { OutBuf.data[idx] = f64(1.0); } else { OutBuf.data[idx] = f64(0.0); }
            return;
        }
        var acc: f64 = f64(0.0);
        if params.op == 2u { acc = f64(1.0); }
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
            OutBuf.data[idx] = nan_f64();
        } else if params.op == 1u {
            OutBuf.data[idx] = acc / f64(params.cols);
        } else if params.op == 2u {
            if (!any) { OutBuf.data[idx] = f64(1.0); }
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

fn isNan(value: f32) -> bool { @IEEE_NAN_PREDICATE@ }

fn nan_f32() -> f32 {
    var bits: u32 = 0x7fc00000u;
    return bitcast<f32>(bits);
}

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
            OutBuf.data[idx] = nan_f32();
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
            OutBuf.data[idx] = nan_f32();
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
    // Fixed Include-NaN: first NaN wins; strict ordering retains first ties.
    if extrema_is_nan(current) { return false; }
    if extrema_is_nan(candidate) { return true; }
    let current_zero = extrema_zero_rank(current);
    let candidate_zero = extrema_zero_rank(candidate);
    if current_zero != 0 && candidate_zero != 0 {
        if op == 0u { return candidate_zero < current_zero; }
        return candidate_zero > current_zero;
    }
    if op == 0u { return candidate < current; }
    return candidate > current;
}

fn extrema_is_nan(value: f64) -> bool {
    let low = runmat_f64_low(value);
    let high = runmat_f64_high(value);
    let magnitude = high & 0x7fffffffu;
    return magnitude > 0x7ff00000u || (magnitude == 0x7ff00000u && low != 0u);
}

fn extrema_zero_rank(value: f64) -> i32 {
    let low = runmat_f64_low(value);
    let high = runmat_f64_high(value);
    if (high & 0x7fffffffu) != 0u || low != 0u { return 0; }
    return select(1, -1, (high & 0x80000000u) != 0u);
}

fn pos_inf_f64() -> f64 {
    return runmat_f64_from_words(0u, 0x7ff00000u);
}

fn neg_inf_f64() -> f64 {
    return runmat_f64_from_words(0u, 0xfff00000u);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols { return; }
        var best: f64;
        if params.op == 0u { best = pos_inf_f64(); } else { best = neg_inf_f64(); }
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
        if params.op == 0u { best = pos_inf_f64(); } else { best = neg_inf_f64(); }
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
    // Fixed Include-NaN: first NaN wins; strict ordering retains first ties.
    if extrema_is_nan(current) { return false; }
    if extrema_is_nan(candidate) { return true; }
    let current_zero = extrema_zero_rank(current);
    let candidate_zero = extrema_zero_rank(candidate);
    if current_zero != 0 && candidate_zero != 0 {
        if op == 0u { return candidate_zero < current_zero; }
        return candidate_zero > current_zero;
    }
    if op == 0u { return candidate < current; }
    return candidate > current;
}

fn extrema_is_nan(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}

fn extrema_zero_rank(value: f32) -> i32 {
    let bits = bitcast<u32>(value);
    if (bits & 0x7fffffffu) != 0u { return 0; }
    return select(1, -1, (bits & 0x80000000u) != 0u);
}

fn pos_inf_f32() -> f32 {
    var bits: u32 = 0x7f800000u;
    return bitcast<f32>(bits);
}

fn neg_inf_f32() -> f32 {
    var bits: u32 = 0xff800000u;
    return bitcast<f32>(bits);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if params.dim == 1u {
        if idx >= params.cols { return; }
        var best: f32;
        if params.op == 0u { best = pos_inf_f32(); } else { best = neg_inf_f32(); }
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
        if params.op == 0u { best = pos_inf_f32(); } else { best = neg_inf_f32(); }
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
