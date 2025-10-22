pub const SCATTER_COL_SHADER_F32: &str = r#"
struct T { data: array<f32> };
@group(0) @binding(0) var<storage, read> V: T;
@group(0) @binding(1) var<storage, read> M: T;
@group(0) @binding(2) var<storage, read_write> Out: T;
struct P { rows:u32, cols:u32, j:u32 }
@group(0) @binding(3) var<uniform> Pm: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x; if (r >= Pm.rows) { return; }
  let dst = r + Pm.j * Pm.rows;
  Out.data[dst] = V.data[r];
}
"#;

pub const SCATTER_ROW_SHADER_F32: &str = r#"
struct T { data: array<f32> };
@group(0) @binding(0) var<storage, read> V: T;
@group(0) @binding(1) var<storage, read> M: T;
@group(0) @binding(2) var<storage, read_write> Out: T;
struct P { rows:u32, cols:u32, i:u32 }
@group(0) @binding(3) var<uniform> Pm: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; if (c >= Pm.cols) { return; }
  let dst = Pm.i + c * Pm.rows;
  Out.data[dst] = V.data[c];
}
"#;


