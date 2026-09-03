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

fn gram_at(row: u32, col: u32, stride: u32) -> f32 {
    return Gram.data[row + col * stride];
}

fn r_at(row: u32, col: u32, stride: u32) -> f32 {
    return OutR.data[row + col * stride];
}

fn set_r(row: u32, col: u32, stride: u32, value: f32) {
    OutR.data[row + col * stride] = value;
}

fn r_inv_at(row: u32, col: u32, stride: u32) -> f32 {
    return OutRInv.data[row + col * stride];
}

fn set_r_inv(row: u32, col: u32, stride: u32, value: f32) {
    OutRInv.data[row + col * stride] = value;
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

    // This kernel is intentionally serial: use the output storage as its
    // factorization workspace so memory scales with the active matrix rather
    // than reserving two MAX_K-square workgroup arrays.
    for (var col: u32 = 0u; col < k; col = col + 1u) {
        for (var row: u32 = 0u; row < k; row = row + 1u) {
            set_r(row, col, stride, 0.0);
            set_r_inv(row, col, stride, 0.0);
        }
    }

    // Cholesky factorisation with compensated subtraction.
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        var sum = gram_at(j, j, stride);
        var c = 0.0;
        for (var p: u32 = 0u; p < j; p = p + 1u) {
            let term = r_at(p, j, stride) * r_at(p, j, stride);
            let y = term - c;
            let t = sum - y;
            c = (t - sum) + y;
            sum = t;
        }
        sum = max(sum, EPS);
        let diag = sqrt(sum);
        set_r(j, j, stride, diag);

        if (diag > EPS) {
            for (var i: u32 = j + 1u; i < k; i = i + 1u) {
                var off = gram_at(j, i, stride);
                var c_off = 0.0;
                for (var p: u32 = 0u; p < j; p = p + 1u) {
                    let term = r_at(p, j, stride) * r_at(p, i, stride);
                    let y = term - c_off;
                    let t = off - y;
                    c_off = (t - off) + y;
                    off = t;
                }
                set_r(j, i, stride, off / diag);
            }
        } else {
            for (var i: u32 = j + 1u; i < k; i = i + 1u) {
                set_r(j, i, stride, 0.0);
            }
        }
        for (var i: u32 = 0u; i < j; i = i + 1u) {
            set_r(j, i, stride, 0.0);
        }
    }

    // Invert the upper-triangular factor (Gauss-Jordan style).
    for (var j: u32 = 0u; j < k; j = j + 1u) {
        let diag = r_at(j, j, stride);
        let inv_diag = select(0.0, 1.0 / diag, diag > EPS);
        set_r_inv(j, j, stride, inv_diag);

        var row = i32(j) - 1;
        loop {
            if (row < 0) {
                break;
            }
            let row_u = u32(row);
            var sum = 0.0;
            var c_sum = 0.0;
            for (var p: u32 = row_u + 1u; p <= j; p = p + 1u) {
                let term = r_at(row_u, p, stride) * r_inv_at(p, j, stride);
                let y = term - c_sum;
                let t = sum + y;
                c_sum = (t - sum) - y;
                sum = t;
            }
            let diag_row = max(r_at(row_u, row_u, stride), EPS);
            set_r_inv(row_u, j, stride, -sum / diag_row);
            row = row - 1;
        }
    }

    // Zero lower triangles explicitly for determinism.
    for (var col: u32 = 0u; col < k; col = col + 1u) {
        for (var row: u32 = col + 1u; row < k; row = row + 1u) {
            set_r_inv(row, col, stride, 0.0);
            set_r(row, col, stride, 0.0);
        }
    }
}
"#;
