#[cfg(feature = "wgpu")]
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{
    AccelProvider, CorrcoefOptions, HostTensorView, ProviderQrOptions, ProviderQrPivot,
};
#[cfg(feature = "wgpu")]
use std::sync::Mutex;

#[cfg(feature = "wgpu")]
static TELEMETRY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(feature = "wgpu")]
fn register_provider() -> &'static dyn AccelProvider {
    // Registration is idempotent; ignore duplicate registrations from other tests.
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default());
    runmat_accelerate_api::provider().expect("wgpu provider registered")
}

#[cfg(feature = "wgpu")]
#[test]
fn telemetry_records_basic_dispatches() {
    let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
    let provider = register_provider();
    provider.reset_telemetry();

    let elem_shape = [8usize];
    let elem_len = elem_shape[0];
    let elem_data: Vec<f64> = (0..elem_len).map(|i| i as f64).collect();
    let elem = provider
        .upload(&HostTensorView {
            data: &elem_data,
            shape: &elem_shape,
        })
        .expect("upload elementwise tensor");

    // Simple copy shader adapted from workgroup smoke test to ensure GPU execution.
    std::env::set_var("RUNMAT_WG", "64");
    let shader = r#"
struct Tensor { data: array<f32> };
struct Params { len: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage,read> input0: Tensor;
@group(0) @binding(1) var<storage,read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  output.data[i] = input0.data[i] + 1.0;
}
"#;
    let _ = provider
        .fused_elementwise(shader, std::slice::from_ref(&elem), &elem_shape, elem_len)
        .expect("fused_elementwise");

    let mat_shape = [2usize, 2usize];
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let a = provider
        .upload(&HostTensorView {
            data: &a_data,
            shape: &mat_shape,
        })
        .expect("upload a");
    let b = provider
        .upload(&HostTensorView {
            data: &b_data,
            shape: &mat_shape,
        })
        .expect("upload b");

    let _prod = block_on(provider.matmul(&a, &b)).expect("matmul");

    let telemetry = provider.telemetry_snapshot();
    assert!(telemetry.upload_bytes > 0, "expected upload bytes > 0");
    assert!(
        telemetry.fused_elementwise.count > 0,
        "expected fused elementwise dispatch count > 0"
    );
    assert!(
        telemetry.matmul.count > 0,
        "expected matmul dispatch count > 0"
    );
    assert!(
        !telemetry.kernel_launches.is_empty(),
        "expected at least one kernel launch to be recorded"
    );

    provider.reset_telemetry();
    let reset = provider.telemetry_snapshot();
    assert_eq!(reset.upload_bytes, 0);
    assert_eq!(reset.fused_elementwise.count, 0);
    assert_eq!(reset.matmul.count, 0);
    assert!(reset.kernel_launches.is_empty());
}

#[cfg(feature = "wgpu")]
#[test]
fn telemetry_records_chunked_matmul_activity() {
    let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
    let provider = register_provider();
    provider.reset_telemetry();

    let m = 32usize;
    let n = 32usize;
    let k = 131_072usize; // ensure chunked path (K_CHUNK_SWITCH = 65536)

    let a = provider.zeros(&[m, k]).expect("zeros a");
    let b = provider.zeros(&[k, n]).expect("zeros b");
    let c = block_on(provider.matmul(&a, &b)).expect("matmul");

    let telemetry = provider.telemetry_snapshot();
    assert!(
        telemetry.matmul.count > 0,
        "expected matmul dispatch count > 0 for chunked matmul, got {}",
        telemetry.matmul.count
    );
    let bind_group_cache_activity =
        telemetry.bind_group_cache_hits + telemetry.bind_group_cache_misses;
    assert!(
        bind_group_cache_activity > 0,
        "expected bind group cache activity > 0 for chunked matmul, got hits={} misses={}",
        telemetry.bind_group_cache_hits,
        telemetry.bind_group_cache_misses
    );
    assert!(
        telemetry.kernel_launches.iter().any(|launch| {
            launch.kernel == "matmul"
                && launch
                    .tuning
                    .iter()
                    .any(|attr| attr.key == "chunked" && attr.value == 1)
        }),
        "expected chunked matmul kernel launch telemetry, got {:?}",
        telemetry.kernel_launches
    );

    let _ = provider.free(&a);
    let _ = provider.free(&b);
    let _ = provider.free(&c);
}

#[cfg(feature = "wgpu")]
#[test]
fn corrcoef_device_path_avoids_host_downloads() {
    let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
    let provider = register_provider();

    let rows = 64usize;
    let cols = 4usize;
    let mut data = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            let x = (r + 1) as f64;
            data[r + c * rows] = x + (c as f64) * 0.25 + x.powi((c as i32) + 1) * 1.0e-4;
        }
    }
    let input = provider
        .upload(&HostTensorView {
            data: &data,
            shape: &[rows, cols],
        })
        .expect("upload corrcoef input");

    provider.reset_telemetry();
    let options = CorrcoefOptions::default();
    let out = block_on(provider.corrcoef(&input, &options)).expect("corrcoef");
    let telemetry = provider.telemetry_snapshot();
    assert_eq!(
        telemetry.download_bytes, 0,
        "corrcoef device path should not download tensors to host"
    );

    let _ = provider.free(&input);
    let _ = provider.free(&out);
}

#[cfg(feature = "wgpu")]
#[test]
fn qr_power_iter_device_path_avoids_host_downloads() {
    let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    std::env::remove_var("RUNMAT_DEBUG_QR");
    std::env::remove_var("RUNMAT_DEBUG_QR_ZEROHOST");

    let provider = register_provider();

    let n = 16usize;
    let k = 4usize;
    let mut lhs = vec![0.0f64; n * n];
    for i in 0..n {
        lhs[i + i * n] = 1.0;
    }
    let mut q0 = vec![0.0f64; n * k];
    for c in 0..k {
        for r in 0..n {
            q0[r + c * n] = ((r + 1 + c) as f64).sin() + (c as f64) * 0.05;
        }
    }

    let lhs_handle = provider
        .upload(&HostTensorView {
            data: &lhs,
            shape: &[n, n],
        })
        .expect("upload lhs");
    let q_handle = provider
        .upload(&HostTensorView {
            data: &q0,
            shape: &[n, k],
        })
        .expect("upload q");
    let product = block_on(provider.matmul(&lhs_handle, &q_handle)).expect("matmul");

    provider.reset_telemetry();
    let options = ProviderQrOptions {
        economy: true,
        pivot: ProviderQrPivot::Matrix,
    };
    let qr = block_on(provider.qr_power_iter(&product, Some(&lhs_handle), &q_handle, &options))
        .expect("qr_power_iter call")
        .expect("device qr result");
    let telemetry = provider.telemetry_snapshot();
    assert_eq!(
        telemetry.download_bytes, 0,
        "qr_power_iter device path should not download tensors to host"
    );
    assert_eq!(qr.q.shape, vec![n, k], "unexpected Q shape");
    assert_eq!(qr.r.shape, vec![k, k], "unexpected R shape");

    let _ = provider.free(&lhs_handle);
    let _ = provider.free(&q_handle);
    let _ = provider.free(&qr.q);
    let _ = provider.free(&qr.r);
    let _ = provider.free(&qr.perm_matrix);
    let _ = provider.free(&qr.perm_vector);
}

#[cfg(feature = "wgpu")]
#[test]
fn qr_power_iter_zero_product_path_remains_stable() {
    let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    std::env::remove_var("RUNMAT_DEBUG_QR");
    std::env::remove_var("RUNMAT_DEBUG_QR_ZEROHOST");

    let provider = register_provider();

    let n = 16usize;
    let k = 4usize;
    let mut lhs = vec![0.0f64; n * n];
    for i in 0..n {
        lhs[i + i * n] = 1.0;
    }
    let q0 = vec![0.0f64; n * k];

    let lhs_handle = provider
        .upload(&HostTensorView {
            data: &lhs,
            shape: &[n, n],
        })
        .expect("upload lhs");
    let q_handle = provider
        .upload(&HostTensorView {
            data: &q0,
            shape: &[n, k],
        })
        .expect("upload q");
    let product = block_on(provider.matmul(&lhs_handle, &q_handle)).expect("matmul");

    provider.reset_telemetry();
    let options = ProviderQrOptions {
        economy: true,
        pivot: ProviderQrPivot::Matrix,
    };
    let qr = block_on(provider.qr_power_iter(&product, Some(&lhs_handle), &q_handle, &options))
        .expect("qr_power_iter call")
        .expect("qr result");
    let telemetry = provider.telemetry_snapshot();
    assert!(
        telemetry.download_bytes > 0,
        "expected host fallback download for zero-product qr_power_iter edge path"
    );

    let q_host = block_on(provider.download(&qr.q)).expect("download q");
    let r_host = block_on(provider.download(&qr.r)).expect("download r");
    assert!(
        q_host.data.iter().all(|v| v.is_finite()),
        "Q contained non-finite values in zero-product edge path"
    );
    assert!(
        r_host.data.iter().all(|v| v.is_finite()),
        "R contained non-finite values in zero-product edge path"
    );

    let _ = provider.free(&lhs_handle);
    let _ = provider.free(&q_handle);
    let _ = provider.free(&qr.q);
    let _ = provider.free(&qr.r);
    let _ = provider.free(&qr.perm_matrix);
    let _ = provider.free(&qr.perm_vector);
}

#[cfg(feature = "wgpu")]
#[test]
fn qr_power_iter_rank_deficient_path_remains_finite() {
    let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    std::env::remove_var("RUNMAT_DEBUG_QR");
    std::env::remove_var("RUNMAT_DEBUG_QR_ZEROHOST");

    let provider = register_provider();

    let n = 16usize;
    let k = 4usize;
    let mut lhs = vec![0.0f64; n * n];
    for i in 0..n {
        lhs[i + i * n] = 1.0;
    }
    let mut q0 = vec![0.0f64; n * k];
    for r in 0..n {
        let base = ((r + 1) as f64) * 0.01;
        q0[r] = base;
        q0[r + n] = base;
        q0[r + 2 * n] = base * 2.0;
        q0[r + 3 * n] = base * 3.0;
    }

    let lhs_handle = provider
        .upload(&HostTensorView {
            data: &lhs,
            shape: &[n, n],
        })
        .expect("upload lhs");
    let q_handle = provider
        .upload(&HostTensorView {
            data: &q0,
            shape: &[n, k],
        })
        .expect("upload q");
    let product = block_on(provider.matmul(&lhs_handle, &q_handle)).expect("matmul");

    provider.reset_telemetry();
    let options = ProviderQrOptions {
        economy: true,
        pivot: ProviderQrPivot::Matrix,
    };
    let qr = block_on(provider.qr_power_iter(&product, Some(&lhs_handle), &q_handle, &options))
        .expect("qr_power_iter call")
        .expect("qr result");

    let telemetry = provider.telemetry_snapshot();
    assert_eq!(
        telemetry.download_bytes, 0,
        "rank-deficient qr_power_iter should stay on the device path for this case"
    );

    let q_host = block_on(provider.download(&qr.q)).expect("download q");
    let r_host = block_on(provider.download(&qr.r)).expect("download r");
    assert!(
        q_host.data.iter().all(|v| v.is_finite()),
        "Q contained non-finite values in rank-deficient edge path"
    );
    assert!(
        r_host.data.iter().all(|v| v.is_finite()),
        "R contained non-finite values in rank-deficient edge path"
    );

    let _ = provider.free(&lhs_handle);
    let _ = provider.free(&q_handle);
    let _ = provider.free(&qr.q);
    let _ = provider.free(&qr.r);
    let _ = provider.free(&qr.perm_matrix);
    let _ = provider.free(&qr.perm_vector);
}

#[cfg(feature = "wgpu")]
#[test]
fn qr_device_column_limit_boundary_uses_expected_path() {
    let _guard = TELEMETRY_TEST_LOCK.lock().expect("telemetry test lock");
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    let provider = register_provider();

    let options = ProviderQrOptions {
        economy: true,
        pivot: ProviderQrPivot::Matrix,
    };

    let rows = 80usize;
    let cols_device = 64usize;
    let mut data_64 = vec![0.0f64; rows * cols_device];
    for c in 0..cols_device {
        for r in 0..rows {
            data_64[r + c * rows] = ((r + 1 + c) as f64).sin() + (c as f64) * 0.01;
        }
    }
    let h64 = provider
        .upload(&HostTensorView {
            data: &data_64,
            shape: &[rows, cols_device],
        })
        .expect("upload 64-col matrix");
    provider.reset_telemetry();
    let qr64 = block_on(provider.qr(&h64, options)).expect("qr 64-col");
    let telem64 = provider.telemetry_snapshot();
    assert_eq!(
        telem64.download_bytes, 0,
        "cols=64 should use device QR path without host download"
    );
    assert_eq!(qr64.q.shape, vec![rows, cols_device]);
    assert_eq!(qr64.r.shape, vec![cols_device, cols_device]);

    let cols_host = 65usize;
    let mut data_65 = vec![0.0f64; rows * cols_host];
    for c in 0..cols_host {
        for r in 0..rows {
            data_65[r + c * rows] = ((r + 1 + c) as f64).cos() + (c as f64) * 0.01;
        }
    }
    let h65 = provider
        .upload(&HostTensorView {
            data: &data_65,
            shape: &[rows, cols_host],
        })
        .expect("upload 65-col matrix");
    provider.reset_telemetry();
    let qr65 = block_on(provider.qr(&h65, options)).expect("qr 65-col");
    let telem65 = provider.telemetry_snapshot();
    assert!(
        telem65.download_bytes > 0,
        "cols=65 should bypass device QR kernel and use host fallback path"
    );
    assert_eq!(qr65.q.shape, vec![rows, cols_host]);
    assert_eq!(qr65.r.shape, vec![cols_host, cols_host]);

    let _ = provider.free(&h64);
    let _ = provider.free(&h65);
    let _ = provider.free(&qr64.q);
    let _ = provider.free(&qr64.r);
    let _ = provider.free(&qr64.perm_matrix);
    let _ = provider.free(&qr64.perm_vector);
    let _ = provider.free(&qr65.q);
    let _ = provider.free(&qr65.r);
    let _ = provider.free(&qr65.perm_matrix);
    let _ = provider.free(&qr65.perm_vector);
}
