#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider::{
    WgpuPrecisionPolicy, WgpuProvider, WgpuProviderOptions,
};
use runmat_accelerate_api::{
    AccelProvider, DenseReductionOperation, DenseReductionRequest, GpuTensorHandle, HostTensorView,
    ProviderCancellationToken,
};
use runmat_execution::{identity::AttemptId, Digest};

// Guard tests to avoid provider state races
static TEST_MUTEX: once_cell::sync::Lazy<tokio::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| tokio::sync::Mutex::new(()));

fn upload_matrix(
    provider: &WgpuProvider,
    rows: usize,
    cols: usize,
    data: &[f64],
) -> GpuTensorHandle {
    assert_eq!(data.len(), rows * cols);
    provider
        .upload(&HostTensorView {
            data,
            shape: &[rows, cols],
        })
        .expect("upload")
}

#[tokio::test]
async fn reduce_sum_dim_semantics() {
    let _guard = TEST_MUTEX.lock().await;
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("create provider");
    // 4x3 matrix with simple pattern: M[r,c] = r + 10*c
    let rows = 4usize;
    let cols = 3usize;
    let mut host = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            host[r + c * rows] = r as f64 + 10.0 * (c as f64);
        }
    }
    let m = upload_matrix(&provider, rows, cols, &host);

    // Sum over dim=0 (rows) → shape [1, cols]
    let sum_dim0 = provider.reduce_sum_dim(&m, 0).await.expect("sum dim0");
    let sum_dim0_host = AccelProvider::download(&provider, &sum_dim0)
        .await
        .expect("dl dim0");
    let expected_dim0: Vec<f64> = (0..cols)
        .map(|c| (0..rows).map(|r| host[r + c * rows]).sum::<f64>())
        .collect();
    assert_eq!(sum_dim0_host.shape, vec![1, cols]);
    for (c, (got, exp)) in sum_dim0_host
        .data
        .iter()
        .zip(expected_dim0.iter())
        .enumerate()
        .take(cols)
    {
        let got = *got;
        let exp = *exp;
        assert!((got - exp).abs() < 1e-6, "dim0 c={c} got={got} exp={exp}");
    }

    // Sum over dim=1 (cols) → shape [rows, 1]
    let sum_dim1 = provider.reduce_sum_dim(&m, 1).await.expect("sum dim1");
    let sum_dim1_host = AccelProvider::download(&provider, &sum_dim1)
        .await
        .expect("dl dim1");
    let expected_dim1: Vec<f64> = (0..rows)
        .map(|r| (0..cols).map(|c| host[r + c * rows]).sum::<f64>())
        .collect();
    assert_eq!(sum_dim1_host.shape, vec![rows, 1]);
    for (r, (got, exp)) in sum_dim1_host
        .data
        .iter()
        .zip(expected_dim1.iter())
        .enumerate()
        .take(rows)
    {
        let got = *got;
        let exp = *exp;
        assert!((got - exp).abs() < 1e-6, "dim1 r={r} got={got} exp={exp}");
    }
}

#[tokio::test]
async fn governed_resident_reduction_binds_attempt_and_reports_no_host_fallback() {
    let _guard = TEST_MUTEX.lock().await;
    let provider = match WgpuProvider::new(WgpuProviderOptions {
        precision: WgpuPrecisionPolicy::RequireF64,
        ..WgpuProviderOptions::default()
    }) {
        Ok(provider) => provider,
        Err(error)
            if error
                .to_string()
                .contains("selected adapter does not expose SHADER_F64") =>
        {
            if std::env::var_os("RUNMAT_REQUIRE_WGPU_F64").is_some() {
                panic!("required exact-f64 WGPU adapter is unavailable: {error}");
            }
            eprintln!("exact-f64 governed reduction qualification: {error}");
            return;
        }
        Err(error) => {
            if std::env::var_os("RUNMAT_REQUIRE_WGPU_F64").is_some() {
                panic!(
                    "required exact-f64 WGPU qualification could not inspect an adapter: {error}"
                );
            }
            eprintln!("exact-f64 governed reduction qualification: {error}");
            return;
        }
    };
    let values = [
        1.0, 7.0, -2.0, // entity 0
        5.0, 3.0, 8.0, // entity 1
        5.0, 9.0, 4.0, // entity 2
        -1.0, 2.0, 8.0, // entity 3
    ];
    let input = upload_matrix(&provider, 3, 4, &values);
    let capability = provider.capability_snapshot();
    let cancellation = ProviderCancellationToken::new();
    let request = DenseReductionRequest {
        input: &input,
        operation: DenseReductionOperation::Maximum { dimension: 1 },
        attempt_id: AttemptId::derive(&[b"governed-reduction-test"]),
        input_identity: Digest::sha256(b"canonical-field-partition"),
        provider_capability_schema_version: capability.schema_version,
        provider_capability_revision: capability.revision,
        cancellation: &cancellation,
    };
    let result = provider
        .execute_dense_reduction(&request)
        .await
        .expect("governed reduction");
    result.validate(request, &capability).unwrap();
    assert!(!result.evidence.used_host_fallback);
    assert_eq!(result.evidence.host_upload_bytes, 0);
    assert_eq!(result.evidence.host_download_bytes, 0);
    assert_eq!(result.evidence.host_synchronizations, 0);

    let reduced = provider.download(&result.values).await.expect("values");
    let indices = provider
        .download(result.indices.as_ref().expect("indices"))
        .await
        .expect("indices");
    assert_eq!(reduced.data, vec![5.0, 9.0, 8.0]);
    assert_eq!(indices.data, vec![2.0, 3.0, 2.0]);
}

#[tokio::test]
async fn elementwise_broadcast_pxc_times_1xc() {
    let _guard = TEST_MUTEX.lock().await;
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("create provider");
    let rows = 4usize;
    let cols = 3usize;
    // X[r,c] = r + 1
    let mut xh = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            xh[r + c * rows] = (r as f64) + 1.0;
        }
    }
    let x = upload_matrix(&provider, rows, cols, &xh);
    // S[1,c] = 2*c
    let mut sh = vec![0.0f64; cols];
    for (c, value) in sh.iter_mut().enumerate().take(cols) {
        *value = (2 * c) as f64;
    }
    let s = provider
        .upload(&HostTensorView {
            data: &sh,
            shape: &[1usize, cols],
        })
        .expect("upload S");

    // Y = X .* S (expect broadcast over rows)
    let y = provider.elem_mul(&x, &s).await.expect("mul");
    let yh = AccelProvider::download(&provider, &y).await.expect("dl y");
    assert_eq!(yh.shape, vec![rows, cols]);
    for c in 0..cols {
        for r in 0..rows {
            let got = yh.data[r + c * rows];
            let exp = xh[r + c * rows] * sh[c];
            assert!((got - exp).abs() < 1e-6, "r={r} c={c} got={got} exp={exp}");
        }
    }
}

#[tokio::test]
async fn fused_dot_per_column_matches_manual() {
    let _guard = TEST_MUTEX.lock().await;
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("create provider");
    let rows = 5usize;
    let cols = 4usize;
    // X[r,c] = r+1, W[r,c] = (c+1)
    let mut xh = vec![0.0f64; rows * cols];
    let mut wh = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            xh[r + c * rows] = (r as f64) + 1.0;
            wh[r + c * rows] = (c as f64) + 1.0;
        }
    }
    let x = upload_matrix(&provider, rows, cols, &xh);
    let w = upload_matrix(&provider, rows, cols, &wh);
    let prod = provider.elem_mul(&x, &w).await.expect("mul");
    let y = provider
        .reduce_sum_dim(&prod, 0)
        .await
        .expect("sum per col");
    let yh = AccelProvider::download(&provider, &y).await.expect("dl y");
    // Manual per-column dot
    for c in 0..cols {
        let mut acc = 0.0;
        for r in 0..rows {
            acc += xh[r + c * rows] * wh[r + c * rows];
        }
        let got = yh.data[c];
        assert!(
            (got - acc).abs() < 1e-6,
            "col={c} got={got} exp={acc} rows={rows}"
        );
    }
}
