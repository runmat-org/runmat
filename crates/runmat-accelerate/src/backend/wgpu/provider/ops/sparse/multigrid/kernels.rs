//! WGSL sources of the device multigrid. Every reduction uses a fixed tree over a fixed
//! workgroup shape, so results are reproducible for a given device configuration.
//! Large dispatches use two-dimensional grids; kernels linearize `workgroup_id` with
//! `num_workgroups`.

/// How the SpMV writes its row value `v = (A x)_row`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) enum SpmvMode {
    /// `y = A x`
    Assign,
    /// `y = b − A x`
    Residual,
    /// `y −= A x`
    Subtract,
    /// `y += A x`
    Add,
}

const LINEAR_INDEX: &str =
    "fn linear_group(wid: vec3<u32>, nwg: vec3<u32>) -> u32 { return wid.x + wid.y * nwg.x; }";

/// Vector CSR: `lanes` consecutive invocations share one row (strided over its entries) and
/// reduce through workgroup memory; a workgroup covers `workgroup / lanes` rows.
pub(super) fn spmv(workgroup: u32, lanes: u32, mode: SpmvMode) -> String {
    let (extra_binding, write) = match mode {
        SpmvMode::Assign => ("", "y[row] = v;"),
        SpmvMode::Residual => (
            "@group(0) @binding(5) var<storage, read> b: array<f32>;",
            "y[row] = b[row] - v;",
        ),
        SpmvMode::Subtract => ("", "y[row] = y[row] - v;"),
        SpmvMode::Add => ("", "y[row] = y[row] + v;"),
    };
    format!(
        r#"
@group(0) @binding(0) var<storage, read> rp: array<u32>;
@group(0) @binding(1) var<storage, read> ci: array<u32>;
@group(0) @binding(2) var<storage, read> va: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
{extra_binding}
var<workgroup> scratch: array<f32, {workgroup}>;
{LINEAR_INDEX}
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let lane = lid.x % {lanes}u;
    let row = linear_group(wid, nwg) * ({workgroup}u / {lanes}u) + lid.x / {lanes}u;
    let rows = arrayLength(&rp) - 1u;
    var sum: f32 = 0.0;
    if (row < rows) {{
        let end = rp[row + 1u];
        for (var e = rp[row] + lane; e < end; e = e + {lanes}u) {{
            sum = sum + va[e] * x[ci[e]];
        }}
    }}
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var s = {lanes}u / 2u; s > 0u; s = s / 2u) {{
        if (lane < s) {{ scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }}
        workgroupBarrier();
    }}
    if (lane == 0u && row < rows) {{
        let v = scratch[lid.x];
        {write}
    }}
}}
"#
    )
}

/// `z = s · D⁻¹ r` with `block × block` inverse blocks (column-major per node); `s = params[0]`.
pub(super) fn block_inverse(workgroup: u32, block: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> inv: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<f32>;
{LINEAR_INDEX}
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = linear_group(wid, nwg) * {workgroup}u + lid.x;
    if (i >= arrayLength(&z)) {{ return; }}
    let node = i / {block}u;
    let a = i % {block}u;
    let base = node * {block}u;
    var sum: f32 = 0.0;
    for (var c = 0u; c < {block}u; c = c + 1u) {{
        sum = sum + inv[node * {block}u * {block}u + c * {block}u + a] * r[base + c];
    }}
    z[i] = params[0] * sum;
}}
"#
    )
}

/// `y = params[0] · x + params[1] · y`.
pub(super) fn axpby(workgroup: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;
{LINEAR_INDEX}
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = linear_group(wid, nwg) * {workgroup}u + lid.x;
    if (i >= arrayLength(&y)) {{ return; }}
    y[i] = params[0] * x[i] + params[1] * y[i];
}}
"#
    )
}

/// Fused CG update with `α = scalars[3]`: `x += α p`, `r −= α q`.
pub(super) fn cg_update(workgroup: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> p: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read> scalars: array<f32>;
{LINEAR_INDEX}
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = linear_group(wid, nwg) * {workgroup}u + lid.x;
    if (i >= arrayLength(&x)) {{ return; }}
    let alpha = scalars[3];
    x[i] = x[i] + alpha * p[i];
    r[i] = r[i] - alpha * q[i];
}}
"#
    )
}

/// `p = z + β p` with `β = scalars[4]`.
pub(super) fn cg_direction(workgroup: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read_write> p: array<f32>;
@group(0) @binding(2) var<storage, read> scalars: array<f32>;
{LINEAR_INDEX}
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = linear_group(wid, nwg) * {workgroup}u + lid.x;
    if (i >= arrayLength(&p)) {{ return; }}
    p[i] = z[i] + scalars[4] * p[i];
}}
"#
    )
}

/// `dst = src`.
pub(super) fn copy(workgroup: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
{LINEAR_INDEX}
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = linear_group(wid, nwg) * {workgroup}u + lid.x;
    if (i >= arrayLength(&dst)) {{ return; }}
    dst[i] = src[i];
}}
"#
    )
}

/// `dst = 0`.
pub(super) fn zero(workgroup: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
{LINEAR_INDEX}
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let i = linear_group(wid, nwg) * {workgroup}u + lid.x;
    if (i >= arrayLength(&dst)) {{ return; }}
    dst[i] = 0.0;
}}
"#
    )
}

/// Dot stage 1: `groups` workgroups, each invocation summing a fixed grid-stride sequence, then
/// a fixed tree; one partial per workgroup.
pub(super) fn dot_partials(workgroup: u32, groups: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
var<workgroup> scratch: array<f32, {workgroup}>;
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let n = arrayLength(&a);
    var sum: f32 = 0.0;
    for (var i = wid.x * {workgroup}u + lid.x; i < n; i = i + {groups}u * {workgroup}u) {{
        sum = sum + a[i] * b[i];
    }}
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var s = {workgroup}u / 2u; s > 0u; s = s / 2u) {{
        if (lid.x < s) {{ scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{ partials[wid.x] = scratch[0]; }}
}}
"#
    )
}

/// Dot stage 2: one workgroup folds the partials and stores `scalars[slot]`.
pub(super) fn dot_finish(workgroup: u32, slot: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> scalars: array<f32>;
var<workgroup> scratch: array<f32, {workgroup}>;
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
    var sum: f32 = 0.0;
    for (var i = lid.x; i < arrayLength(&partials); i = i + {workgroup}u) {{
        sum = sum + partials[i];
    }}
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var s = {workgroup}u / 2u; s > 0u; s = s / 2u) {{
        if (lid.x < s) {{ scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{ scalars[{slot}] = scratch[0]; }}
}}
"#
    )
}

/// CG scalar steps on slots `ρ = 0, ⟨p,q⟩ = 1, ρ_new = 2, α = 3, β = 4`. `op = 0`: `α = ρ/⟨p,q⟩`
/// (zero when not positive and finite); `op = 1`: `β = ρ_new/ρ`, `ρ = ρ_new` (zero likewise).
pub(super) fn cg_scalars(op: u32) -> String {
    let body = if op == 0 {
        "let pq = s[1]; let rho = s[0]; if (pq > 0.0 && rho == rho && pq < 3.0e38) { s[3] = rho / pq; } else { s[3] = 0.0; }"
    } else {
        "let rho = s[0]; let next = s[2]; if (rho > 0.0 && next == next) { s[4] = next / rho; s[0] = next; } else { s[4] = 0.0; }"
    };
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> s: array<f32>;
@compute @workgroup_size(1)
fn main() {{ {body} }}
"#
    )
}

/// Dense row-major `y = C x`: one workgroup per row, a fixed tree over the columns.
pub(super) fn dense_matvec(workgroup: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> c: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> scratch: array<f32, {workgroup}>;
{LINEAR_INDEX}
@compute @workgroup_size({workgroup})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {{
    let row = linear_group(wid, nwg);
    let n = arrayLength(&y);
    var sum: f32 = 0.0;
    if (row < n) {{
        for (var col = lid.x; col < n; col = col + {workgroup}u) {{
            sum = sum + c[row * n + col] * x[col];
        }}
    }}
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var s = {workgroup}u / 2u; s > 0u; s = s / 2u) {{
        if (lid.x < s) {{ scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }}
        workgroupBarrier();
    }}
    if (lid.x == 0u && row < n) {{ y[row] = scratch[0]; }}
}}
"#
    )
}
