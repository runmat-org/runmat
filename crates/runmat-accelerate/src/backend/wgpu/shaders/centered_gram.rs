pub const CENTERED_GRAM_SHADER_F32: &str = r#"
struct Matrix {
    data: array<f32>,
};

struct Means {
    data: array<f32>,
};

struct Output {
    data: array<f32>,
};

struct Params {
    rows: u32,
    cols: u32,
    lda: u32,
    ldc: u32,
    offset_matrix: u32,
    offset_means: u32,
    offset_out: u32,
    _pad0: u32,
    denom: vec4<f32>,
    _pad1: vec4<f32>,
    _pad2: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> MatrixTensor: Matrix;
@group(0) @binding(1) var<storage, read> MeansTensor: Means;
@group(0) @binding(2) var<storage, read_write> OutTensor: Output;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<array<f32, @MT@>, @MT@>;
var<workgroup> tileB: array<array<f32, @MT@>, @MT@>;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let col_i = wid.y * tile + lid.y;
    let col_j = wid.x * tile + lid.x;
    let rows = params.rows;
    let cols = params.cols;

    var acc: f32 = 0.0;
    var comp: f32 = 0.0;
    let tiles_k = (rows + tile - 1u) / tile;
    for (var t: u32 = 0u; t < tiles_k; t = t + 1u) {
        let row_left = t * tile + lid.x;
        let row_right = t * tile + lid.y;

        var left_val: f32 = 0.0;
        if (col_i < cols && row_left < rows) {
            let idx = params.offset_matrix + row_left + col_i * params.lda;
            let mean = MeansTensor.data[params.offset_means + col_i];
            left_val = MatrixTensor.data[idx] - mean;
        }

        var right_val: f32 = 0.0;
        if (col_j < cols && row_right < rows) {
            let idx = params.offset_matrix + row_right + col_j * params.lda;
            let mean = MeansTensor.data[params.offset_means + col_j];
            right_val = MatrixTensor.data[idx] - mean;
        }

        tileA[lid.y][lid.x] = left_val;
        tileB[lid.y][lid.x] = right_val;
        workgroupBarrier();

        for (var p: u32 = 0u; p < tile; p = p + 1u) {
            let a_val = tileA[lid.y][p];
            let b_val = tileB[p][lid.x];
            let prod = a_val * b_val;
            let y = prod - comp;
            let t_sum = acc + y;
            comp = (t_sum - acc) - y;
            acc = t_sum;
        }
        workgroupBarrier();
    }

    if (col_i < cols && col_j < cols) {
        var value = (acc + comp) * params.denom.x;
        if (col_i == col_j && value < 0.0 && value > -1.0e-12) {
            value = 0.0;
        }
        let idx_out = params.offset_out + col_i + col_j * params.ldc;
        OutTensor.data[idx_out] = value;
    }
}
"#;

pub const CENTERED_GRAM_SHADER_F64: &str = r#"
struct Matrix {
    data: array<f64>,
};

struct Means {
    data: array<f64>,
};

struct Output {
    data: array<f64>,
};

struct Params {
    rows: u32,
    cols: u32,
    lda: u32,
    ldc: u32,
    offset_matrix: u32,
    offset_means: u32,
    offset_out: u32,
    _pad0: u32,
    denom: vec2<f64>,
    _pad1: vec2<f64>,
    _pad2: vec2<f64>,
};

@group(0) @binding(0) var<storage, read> MatrixTensor: Matrix;
@group(0) @binding(1) var<storage, read> MeansTensor: Means;
@group(0) @binding(2) var<storage, read_write> OutTensor: Output;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<array<f64, @MT@>, @MT@>;
var<workgroup> tileB: array<array<f64, @MT@>, @MT@>;

@compute @workgroup_size(@MT@, @MT@, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile = @MT@u;
    let col_i = wid.y * tile + lid.y;
    let col_j = wid.x * tile + lid.x;
    let rows = params.rows;
    let cols = params.cols;

    var acc: f64 = 0.0;
    var comp: f64 = 0.0;
    let tiles_k = (rows + tile - 1u) / tile;
    for (var t: u32 = 0u; t < tiles_k; t = t + 1u) {
        let row_left = t * tile + lid.x;
        let row_right = t * tile + lid.y;

        var left_val: f64 = 0.0;
        if (col_i < cols && row_left < rows) {
            let idx = params.offset_matrix + row_left + col_i * params.lda;
            let mean = MeansTensor.data[params.offset_means + col_i];
            left_val = MatrixTensor.data[idx] - mean;
        }

        var right_val: f64 = 0.0;
        if (col_j < cols && row_right < rows) {
            let idx = params.offset_matrix + row_right + col_j * params.lda;
            let mean = MeansTensor.data[params.offset_means + col_j];
            right_val = MatrixTensor.data[idx] - mean;
        }

        tileA[lid.y][lid.x] = left_val;
        tileB[lid.y][lid.x] = right_val;
        workgroupBarrier();

        for (var p: u32 = 0u; p < tile; p = p + 1u) {
            let a_val = tileA[lid.y][p];
            let b_val = tileB[p][lid.x];
            let prod = a_val * b_val;
            let y = prod - comp;
            let t_sum = acc + y;
            comp = (t_sum - acc) - y;
            acc = t_sum;
        }
        workgroupBarrier();
    }

    if (col_i < cols && col_j < cols) {
        var value = (acc + comp) * params.denom.x;
        if (col_i == col_j && value < 0.0 && value > -1.0e-12) {
            value = 0.0;
        }
        let idx_out = params.offset_out + col_i + col_j * params.ldc;
        OutTensor.data[idx_out] = value;
    }
}
"#;
