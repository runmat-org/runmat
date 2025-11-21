pub const QR_POWER_ITER_CHOL_SHADER: &str = r#"
const MAX_K: u32 = 64u;
const EPS: f32 = 1.0e-6;

struct Matrix {
    data: array<f32>,
};

struct Params {
    cols: u32,
    stride: u32,
    _pad0: vec2<u32>,
}

@group(0) @binding(0)
var<storage, read> Gram : Matrix;

@group(0) @binding(1)
var<storage, read_write> OutR : Matrix;

@group(0) @binding(2)
var<storage, read_write> OutRInv : Matrix;

@group(0) @binding(3)
var<uniform> params : Params;

var<workgroup> R_local : array<array<f32, MAX_K>, MAX_K>;
var<workgroup> RInv_local : array<array<f32, MAX_K>, MAX_K>;

fn gram_at(row: u32, col: u32, stride: u32) -> f32 {
    return Gram.data[row + col * stride];
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(local_invocation_index) local_index: u32) {
    if (local_index != 0u) {
        return;
    }

    let k = params.cols;
    if (k == 0u || k > MAX_K) {
        return;
    }
    let stride = params.stride;

    // Initialise local storage
    for (var col: u32 = 0u; col < MAX_K; col = col + 1u) {
        for (var row: u32 = 0u; row < MAX_K; row = row + 1u) {
            R_local[row][col] = 0.0;
            RInv_local[row][col] = 0.0;
        }
    }

    // Cholesky factorisation with compensated subtraction.
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        var sum = gram_at(j, j, stride);
        var c = 0.0;
        for (var p: u32 = 0u; p < j; p = p + 1u) {
            let term = R_local[p][j] * R_local[p][j];
            let y = term - c;
            let t = sum - y;
            c = (t - sum) + y;
            sum = t;
        }
        sum = max(sum, EPS);
        let diag = sqrt(sum);
        R_local[j][j] = diag;

        if (diag > EPS) {
            for (var i: u32 = j + 1u; i < k; i = i + 1u) {
                var off = gram_at(j, i, stride);
                var c_off = 0.0;
                for (var p: u32 = 0u; p < j; p = p + 1u) {
                    let term = R_local[p][j] * R_local[p][i];
                    let y = term - c_off;
                    let t = off - y;
                    c_off = (t - off) + y;
                    off = t;
                }
                R_local[j][i] = off / diag;
            }
        } else {
            for (var i: u32 = j + 1u; i < k; i = i + 1u) {
                R_local[j][i] = 0.0;
            }
        }
        for (var i: u32 = 0u; i < j; i = i + 1u) {
            R_local[j][i] = 0.0;
        }
    }

    // Invert the upper-triangular factor (Gauss-Jordan style).
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        let diag = R_local[j][j];
        let inv_diag = select(0.0, 1.0 / diag, diag > EPS);
        RInv_local[j][j] = inv_diag;

        var row = i32(j) - 1;
        loop {
            if (row < 0) {
                break;
            }
            let row_u = u32(row);
            var sum = 0.0;
            var c_sum = 0.0;
            for (var p: u32 = row_u + 1u; p <= j; p = p + 1u) {
                let term = R_local[row_u][p] * RInv_local[p][j];
                let y = term - c_sum;
                let t = sum + y;
                c_sum = (t - sum) - y;
                sum = t;
            }
            let diag_row = max(R_local[row_u][row_u], EPS);
            RInv_local[row_u][j] = -sum / diag_row;
            row = row - 1;
        }
    }

    // Zero lower triangles explicitly for determinism.
    for (var col: u32 = 0u; col < k; col = col + 1u) {
        for (var row: u32 = col + 1u; row < k; row = row + 1u) {
            RInv_local[row][col] = 0.0;
            R_local[row][col] = 0.0;
        }
    }

    for (var col: u32 = 0u; col < k; col = col + 1u) {
        for (var row: u32 = 0u; row < k; row = row + 1u) {
            let idx = row + col * stride;
            OutR.data[idx] = R_local[row][col];
            OutRInv.data[idx] = RInv_local[row][col];
        }
    }
}
"#;
