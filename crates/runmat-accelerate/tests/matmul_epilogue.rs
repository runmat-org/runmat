#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{HostTensorView, MatmulEpilogue, ScaleOp};

#[cfg(feature = "wgpu")]
fn cpu_matmul(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    assert_eq!(ac, br);
    let mut out = vec![0.0; ar * bc];
    for j in 0..bc {
        for i in 0..ar {
            let mut sum = 0.0;
            for k in 0..ac {
                sum += a[i + k * ar] * b[k + j * br];
            }
            out[i + j * ar] = sum;
        }
    }
    out
}

#[cfg(feature = "wgpu")]
#[test]
fn matmul_epilogue_row_col_alpha_beta() {
    // Force f64 unless device requires f32; we only use provider APIs
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    // A: 3x2, B: 2x4 -> C: 3x4 (column-major)
    let ar = 3usize;
    let ac = 2usize;
    let br = 2usize;
    let bc = 4usize;
    let a: Vec<f64> = vec![
        1.0, 2.0, 3.0, // col 0
        4.0, 5.0, 6.0, // col 1
    ];
    let b: Vec<f64> = vec![
        1.0, 3.0, 5.0, 7.0, // col 0
        2.0, 4.0, 6.0, 8.0, // col 1
    ];

    let ha = p
        .upload(&HostTensorView {
            data: &a,
            shape: &[ar, ac],
        })
        .expect("upload A");
    let hb = p
        .upload(&HostTensorView {
            data: &b,
            shape: &[br, bc],
        })
        .expect("upload B");

    // Row/col scales
    let row_scale: Vec<f64> = vec![2.0, 0.5, 1.0];
    let col_scale: Vec<f64> = vec![1.0, 2.0, 0.25, 1.5];
    let hrow = p
        .upload(&HostTensorView {
            data: &row_scale,
            shape: &[ar, 1],
        })
        .expect("row scale");
    let hcol = p
        .upload(&HostTensorView {
            data: &col_scale,
            shape: &[1, bc],
        })
        .expect("col scale");

    let mut ep = MatmulEpilogue::noop();
    ep.alpha = 1.25;
    ep.beta = -0.5;
    ep.row_scale = Some(hrow.clone());
    ep.col_scale = Some(hcol.clone());
    ep.row_op = ScaleOp::Multiply;
    ep.col_op = ScaleOp::Multiply;

    let hc = p.matmul_epilogue(&ha, &hb, &ep).expect("matmul_epilogue");
    let host = p.download(&hc).expect("download");
    assert_eq!(host.shape, vec![ar, bc]);

    // CPU reference: (alpha * (A*B) + beta) .* row .* col
    let base = cpu_matmul(&a, ar, ac, &b, br, bc);
    let mut expected = vec![0.0; ar * bc];
    for (j, col) in col_scale.iter().enumerate().take(bc) {
        for (i, row) in row_scale.iter().enumerate().take(ar) {
            let idx = i + j * ar;
            let v = base[idx] * ep.alpha + ep.beta;
            expected[idx] = v * row * col;
        }
    }

    for (idx, (got, want)) in host.data.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff < 1e-9,
            "mismatch at {}: got={} want={} diff={}",
            idx,
            got,
            want,
            diff
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn matmul_epilogue_col_divide() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    // A: 2x2, B: 2x2
    let ar = 2usize;
    let ac = 2usize;
    let br = 2usize;
    let bc = 2usize;
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
    let ha = p
        .upload(&HostTensorView {
            data: &a,
            shape: &[ar, ac],
        })
        .expect("upload A");
    let hb = p
        .upload(&HostTensorView {
            data: &b,
            shape: &[br, bc],
        })
        .expect("upload B");

    // col denom
    let denom: Vec<f64> = vec![2.0, 4.0];
    let hcol = p
        .upload(&HostTensorView {
            data: &denom,
            shape: &[1, bc],
        })
        .expect("col denom");

    let mut ep = MatmulEpilogue::noop();
    ep.col_scale = Some(hcol.clone());
    ep.col_op = runmat_accelerate_api::ScaleOp::Divide;

    let hc = p.matmul_epilogue(&ha, &hb, &ep).expect("matmul_epilogue");
    let host = p.download(&hc).expect("download");
    // Expected: identity matmul gives A, then divide each column by denom
    let expected = [a[0] / 2.0, a[1] / 2.0, a[2] / 4.0, a[3] / 4.0];
    for (got, want) in host.data.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-9);
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn matmul_epilogue_clamp_pow() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    // Simple 2x2 matmul with known output
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let b: Vec<f64> = vec![2.0, 0.0, 0.0, 2.0]; // 2x2 diagonal scale
    let ha = p
        .upload(&HostTensorView {
            data: &a,
            shape: &[2, 2],
        })
        .expect("upload A");
    let hb = p
        .upload(&HostTensorView {
            data: &b,
            shape: &[2, 2],
        })
        .expect("upload B");

    let mut ep = MatmulEpilogue::noop();
    ep.alpha = 1.0;
    ep.beta = 0.0;
    ep.clamp_min = Some(4.0);
    ep.clamp_max = Some(10.0);
    ep.pow_exponent = Some(2.0);

    let hc = p.matmul_epilogue(&ha, &hb, &ep).expect("matmul_epilogue");
    let host = p.download(&hc).expect("download");
    assert_eq!(host.shape, vec![2, 2]);

    // CPU reference: pow(clamp(matmul, [4,10]), 2)
    let base = cpu_matmul(&a, 2, 2, &b, 2, 2);
    let mut expected = vec![0.0; base.len()];
    for (idx, val) in base.iter().enumerate() {
        expected[idx] = val.clamp(4.0, 10.0).powf(2.0);
    }
    for (idx, (got, want)) in host.data.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff < 5e-5,
            "clamp+pow mismatch at {}: got={} want={} diff={}",
            idx,
            got,
            want,
            diff
        );
    }
}
