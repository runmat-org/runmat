pub const SYRK_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct Params {
    rows_total: u32,
    cols: u32,
    lda: u32,
    ldc: u32,
    row_offset: u32,
    chunk_rows: u32,
    flags: u32,
    _pad: u32,
};

const SYRK_FLAG_ACCUMULATE: u32 = 1u;
const SYRK_FLAG_FILL_BOTH: u32 = 2u;

var<workgroup> tile_left: array<array<f64, @MT@>, @MT@>;
var<workgroup> tile_right: array<array<f64, @MT@>, @MT@>;

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let global_row = wid.y * tile + lid.y;
    let global_col = wid.x * tile + lid.x;
    let cols = params.cols;

    let is_active = global_row < cols && global_col < cols && global_row <= global_col;

    let rows_total = params.rows_total;
    let row_offset = params.row_offset;
    let chunk_rows = params.chunk_rows;

    var acc: f64 = 0.0;
    let tiles_k = (chunk_rows + tile - 1u) / tile;

    for (var t: u32 = 0u; t < tiles_k; t = t + 1u) {
        let kk_base = t * tile;

        let left_row = row_offset + kk_base + lid.x;
        var left_val: f64 = 0.0;
        if (global_row < cols && left_row < rows_total) {
            left_val = A.data[left_row + global_row * params.lda];
        }
        tile_left[lid.y][lid.x] = left_val;

        let right_row = row_offset + kk_base + lid.y;
        var right_val: f64 = 0.0;
        if (global_col < cols && right_row < rows_total) {
            right_val = A.data[right_row + global_col * params.lda];
        }
        tile_right[lid.y][lid.x] = right_val;

        workgroupBarrier();

        if (is_active) {
            for (var p: u32 = 0u; p < tile; p = p + 1u) {
                let a_val = tile_left[lid.y][p];
                let b_val = tile_right[p][lid.x];
                acc = acc + a_val * b_val;
            }
        }

        workgroupBarrier();
    }

    if (!is_active) {
        return;
    }

    let partial = acc;
    let upper_index = global_row + global_col * params.ldc;
    let lower_index = global_col + global_row * params.ldc;

    var upper_prev: f64 = 0.0;
    var lower_prev: f64 = 0.0;
    if ((params.flags & SYRK_FLAG_ACCUMULATE) != 0u) {
        upper_prev = Out.data[upper_index];
        if (global_row != global_col && (params.flags & SYRK_FLAG_FILL_BOTH) != 0u) {
            lower_prev = Out.data[lower_index];
        }
    }

    var upper_val: f64 = partial;
    if ((params.flags & SYRK_FLAG_ACCUMULATE) != 0u) {
        upper_val = upper_prev + partial;
    }
    Out.data[upper_index] = upper_val;

    if ((params.flags & SYRK_FLAG_FILL_BOTH) != 0u && global_row != global_col) {
        var lower_val: f64 = partial;
        if ((params.flags & SYRK_FLAG_ACCUMULATE) != 0u) {
            lower_val = lower_prev + partial;
        }
        Out.data[lower_index] = lower_val;
    }
}
"#;

pub const SYRK_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct Params {
    rows_total: u32,
    cols: u32,
    lda: u32,
    ldc: u32,
    row_offset: u32,
    chunk_rows: u32,
    flags: u32,
    _pad: u32,
};

const SYRK_FLAG_ACCUMULATE: u32 = 1u;
const SYRK_FLAG_FILL_BOTH: u32 = 2u;

var<workgroup> tile_left: array<array<f32, @MT@>, @MT@>;
var<workgroup> tile_right: array<array<f32, @MT@>, @MT@>;

@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let global_row = wid.y * tile + lid.y;
    let global_col = wid.x * tile + lid.x;
    let cols = params.cols;

    let is_active = global_row < cols && global_col < cols && global_row <= global_col;

    let rows_total = params.rows_total;
    let row_offset = params.row_offset;
    let chunk_rows = params.chunk_rows;

    var acc: f32 = 0.0;
    let tiles_k = (chunk_rows + tile - 1u) / tile;

    for (var t: u32 = 0u; t < tiles_k; t = t + 1u) {
        let kk_base = t * tile;

        let left_row = row_offset + kk_base + lid.x;
        var left_val: f32 = 0.0;
        if (global_row < cols && left_row < rows_total) {
            left_val = A.data[left_row + global_row * params.lda];
        }
        tile_left[lid.y][lid.x] = left_val;

        let right_row = row_offset + kk_base + lid.y;
        var right_val: f32 = 0.0;
        if (global_col < cols && right_row < rows_total) {
            right_val = A.data[right_row + global_col * params.lda];
        }
        tile_right[lid.y][lid.x] = right_val;

        workgroupBarrier();

        if (is_active) {
            for (var p: u32 = 0u; p < tile; p = p + 1u) {
                let a_val = tile_left[lid.y][p];
                let b_val = tile_right[p][lid.x];
                acc = acc + a_val * b_val;
            }
        }

        workgroupBarrier();
    }

    if (!is_active) {
        return;
    }

    let partial = acc;
    let upper_index = global_row + global_col * params.ldc;
    let lower_index = global_col + global_row * params.ldc;

    var upper_prev: f32 = 0.0;
    var lower_prev: f32 = 0.0;
    if ((params.flags & SYRK_FLAG_ACCUMULATE) != 0u) {
        upper_prev = Out.data[upper_index];
        if (global_row != global_col && (params.flags & SYRK_FLAG_FILL_BOTH) != 0u) {
            lower_prev = Out.data[lower_index];
        }
    }

    var upper_val: f32 = partial;
    if ((params.flags & SYRK_FLAG_ACCUMULATE) != 0u) {
        upper_val = upper_prev + partial;
    }
    Out.data[upper_index] = upper_val;

    if ((params.flags & SYRK_FLAG_FILL_BOTH) != 0u && global_row != global_col) {
        var lower_val: f32 = partial;
        if ((params.flags & SYRK_FLAG_ACCUMULATE) != 0u) {
            lower_val = lower_prev + partial;
        }
        Out.data[lower_index] = lower_val;
    }
}
"#;
