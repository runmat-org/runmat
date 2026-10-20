use super::*;
use crate::context::RuntimeContext;
use crate::execution::{
    CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError, RuntimeExecutionService,
};
use futures::executor::block_on;
use image::io::Reader as ImageReader;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostLogicalTensorView, HostNumericTensorView, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::IntegerStorage;
use std::cell::Cell;
use std::fs;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tempfile::tempdir;

struct CountingDownloadProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
    pending: AtomicBool,
    abandoned: AtomicUsize,
}

struct PendingRead<'a>(&'a AtomicUsize);

impl Drop for PendingRead<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl AccelProvider for CountingDownloadProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_logical_raw(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            if self.pending.load(Ordering::SeqCst) {
                let _pending = PendingRead(&self.abandoned);
                futures::future::pending::<()>().await;
                unreachable!()
            }
            self.inner.download_numeric(handle).await
        })
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
}

fn tensor(data: Vec<f64>, shape: Vec<usize>, dtype: NumericDType) -> Tensor {
    Tensor::new_with_dtype(data, shape, dtype).expect("tensor")
}

fn typed_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(storage, shape).expect("integer tensor")
}

fn call(args: Vec<Value>) -> BuiltinResult<Value> {
    block_on(imwrite_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

fn indexed_with_large_host_map(path: &std::path::Path) -> Vec<Value> {
    vec![
        Value::Tensor(tensor(vec![1.0], vec![1, 1], NumericDType::F64)),
        Value::Tensor(tensor(
            vec![0.5; 1500 * 3],
            vec![1500, 3],
            NumericDType::F64,
        )),
        Value::String(path.to_string_lossy().into_owned().into()),
    ]
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn host_preflight_keeps_exact_colormap_and_alpha_geometry_without_file_effects() {
    let image = Value::Tensor(tensor(vec![1.0], Vec::new(), NumericDType::F64));
    let row_map = Value::Tensor(tensor(vec![0.0, 0.5, 1.0], vec![1, 3], NumericDType::F64));
    preflight::preflight_resident_argument_roles(crate::call::FunctionArgumentView::from_slice(&[
        image.clone(),
        row_map,
    ]))
    .unwrap();
    let vector_map = Value::Tensor(tensor(vec![0.0, 0.5, 1.0], vec![3], NumericDType::F64));
    assert_eq!(
        preflight::preflight_resident_argument_roles(
            crate::call::FunctionArgumentView::from_slice(&[image.clone(), vector_map])
        )
        .unwrap_err()
        .identifier(),
        IMWRITE_ERROR_INVALID_COLORMAP.identifier
    );

    let options = |alpha| {
        vec![
            image.clone(),
            Value::String("unused.png".into()),
            Value::String("Alpha".into()),
            alpha,
        ]
    };
    let scalar_alpha = Value::Tensor(tensor(vec![0.5], vec![1, 1], NumericDType::F64));
    preflight::preflight_option_text(
        crate::call::FunctionArgumentView::from_slice(&options(scalar_alpha)),
        2,
        "unused.png",
        false,
    )
    .unwrap();
    let rank_zero_alpha = Value::Tensor(tensor(vec![0.5], Vec::new(), NumericDType::F64));
    assert_eq!(
        preflight::preflight_option_text(
            crate::call::FunctionArgumentView::from_slice(&options(rank_zero_alpha)),
            2,
            "unused.png",
            false
        )
        .unwrap_err()
        .identifier(),
        IMWRITE_ERROR_INVALID_OPTION.identifier
    );
}

#[test]
fn known_invalid_option_precedes_host_map_scan_and_cancellation() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("invalid-option.png");
    let mut args = indexed_with_large_host_map(&path);
    args.extend([Value::String("BitDepth".into()), Value::Num(8.0)]);
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let context = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let error = block_on(imwrite_builtin(
        Some(context),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_OPTION.identifier);
    assert!(!path.exists());
}

#[test]
fn host_map_scan_reports_cooperation_failure_before_io() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("turn-failed.png");
    let args = indexed_with_large_host_map(&path);
    preflight::preflight_resident_argument_roles(crate::call::FunctionArgumentView::from_slice(
        &args,
    ))
    .unwrap();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let cancellation = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        move || {
            counted.set(counted.get() + 1);
            Box::pin(async {
                Err(ExecutionServiceError::Infrastructure(
                    "test turn failed".into(),
                ))
            })
        },
    );
    let mut cooperation = ImageCooperation::with_budget(budget, BUILTIN_NAME);
    let error = block_on(preflight::preflight_known_host_pixels(
        crate::call::FunctionArgumentView::from_slice(&args),
        &mut cooperation,
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:parallel:InfrastructureFailure")
    );
    assert_eq!(turns.get(), 1);
    assert!(!path.exists());
}

#[test]
fn builtin_context_cancellation_stops_before_file_publication() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("cancelled.png");
    let args = indexed_with_large_host_map(&path);
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let context = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let error = block_on(imwrite_builtin(
        Some(context),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(!path.exists());
}

#[test]
fn invalid_filename_text_is_rejected_before_image_gather_or_write() {
    let image = Value::Tensor(tensor(vec![0.0], vec![1, 1], NumericDType::F64));
    let error = call(vec![image.clone(), Value::Num(42.0)]).unwrap_err();
    assert_eq!(
        error.identifier(),
        IMWRITE_ERROR_INVALID_ARGUMENT.identifier
    );
    let error = call(vec![image, Value::Num(1.0), Value::Bool(true)]).unwrap_err();
    assert_eq!(
        error.identifier(),
        IMWRITE_ERROR_INVALID_FILENAME.identifier
    );
}

#[test]
fn unsupported_explicit_format_fails_before_writing() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("unsupported.dat");
    let image = Value::Tensor(tensor(vec![0.0], vec![1, 1], NumericDType::F64));
    let error = call(vec![
        image,
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::String("not-an-image-format".into()),
    ])
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_FORMAT.identifier);
    assert!(!path.exists());
}

#[test]
fn recognized_but_unsupported_option_fails_before_writing() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("unsupported.png");
    let image = Value::Tensor(tensor(vec![0.0], vec![1, 1], NumericDType::F64));
    let error = call(vec![
        image,
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::String("Bit_Depth".into()),
        Value::Num(8.0),
    ])
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_OPTION.identifier);
    assert!(!path.exists());
}

#[test]
fn known_invalid_geometry_and_controls_never_download_resident_image() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(CountingDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
        pending: AtomicBool::new(false),
        abandoned: AtomicUsize::new(0),
    }));
    let _thread = ThreadProviderGuard::set(Some(owner));
    let dir = tempdir().unwrap();
    let path = Value::String(
        dir.path()
            .join("never.png")
            .to_string_lossy()
            .into_owned()
            .into(),
    );

    let invalid_channels = tensor(vec![0.0; 8], vec![2, 2, 2], NumericDType::F64);
    let resident =
        crate::builtins::common::gpu_helpers::upload_tensor(owner, &invalid_channels).unwrap();
    let error = call(vec![Value::GpuTensor(resident), path.clone()]).unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);

    let valid = tensor(vec![0.0; 4], vec![2, 2], NumericDType::F64);
    let resident = crate::builtins::common::gpu_helpers::upload_tensor(owner, &valid).unwrap();
    let invalid_map = Value::Tensor(tensor(vec![0.0; 4], vec![2, 2], NumericDType::F64));
    let error = call(vec![
        Value::GpuTensor(resident.clone()),
        invalid_map,
        path.clone(),
    ])
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        IMWRITE_ERROR_INVALID_COLORMAP.identifier
    );
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);

    let error = call(vec![
        Value::GpuTensor(resident.clone()),
        path.clone(),
        Value::String("Quality".into()),
        Value::String("invalid".into()),
    ])
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_OPTION.identifier);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
    for (name, value) in [
        ("LoopCount", Value::Num(0.5)),
        ("LoopCount", Value::Num(65536.0)),
        ("DelayTime", Value::Num(1e300)),
        ("WriteMode", Value::String("invalid".into())),
    ] {
        let error = call(vec![
            Value::GpuTensor(resident.clone()),
            Value::String(
                dir.path()
                    .join("never.gif")
                    .to_string_lossy()
                    .into_owned()
                    .into(),
            ),
            Value::String(name.into()),
            value,
        ])
        .unwrap_err();
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_OPTION.identifier);
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        assert!(!dir.path().join("never.gif").exists());
    }
    let png_path = dir.path().join("last_append.png");
    let error = call(vec![
        Value::GpuTensor(resident.clone()),
        Value::String(png_path.to_string_lossy().into_owned().into()),
        Value::String("WriteMode".into()),
        Value::String("overwrite".into()),
        Value::String("WriteMode".into()),
        Value::String("append".into()),
    ])
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_OPTION.identifier);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
    assert!(!png_path.exists());
    let invalid_values = Value::Tensor(tensor(
        vec![f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![2, 3],
        NumericDType::F64,
    ));
    let error = call(vec![
        Value::GpuTensor(resident.clone()),
        invalid_values,
        path.clone(),
    ])
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        IMWRITE_ERROR_INVALID_COLORMAP.identifier
    );
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
    let resident_map = tensor(vec![0.0; 6], vec![2, 3], NumericDType::F64);
    let resident_map =
        crate::builtins::common::gpu_helpers::upload_tensor(owner, &resident_map).unwrap();
    for bad_index in [f64::NAN, -1.0, 3.0] {
        let indexed = Value::Tensor(tensor(vec![bad_index], vec![1, 1], NumericDType::F64));
        let error = call(vec![
            indexed,
            Value::GpuTensor(resident_map.clone()),
            path.clone(),
        ])
        .unwrap_err();
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
    }
    let logical_alpha = owner
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0, 0, 1],
            shape: &[2, 2],
        })
        .unwrap();
    let error = call(vec![
        Value::GpuTensor(resident.clone()),
        path.clone(),
        Value::String("Alpha".into()),
        Value::GpuTensor(logical_alpha.clone()),
    ])
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_OPTION.identifier);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
    let rgb = tensor(vec![0.0; 12], vec![2, 2, 3], NumericDType::F64);
    let rgb = crate::builtins::common::gpu_helpers::upload_tensor(owner, &rgb).unwrap();
    let error = call(vec![
        Value::GpuTensor(rgb),
        Value::String(
            dir.path()
                .join("rgb.gif")
                .to_string_lossy()
                .into_owned()
                .into(),
        ),
    ])
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);

    let sixteen = typed_tensor(IntegerStorage::U16(vec![0; 4]), vec![2, 2]);
    let sixteen = crate::builtins::common::gpu_helpers::upload_tensor(owner, &sixteen).unwrap();
    for extension in ["jpg", "bmp"] {
        let file = dir.path().join(format!("depth.{extension}"));
        let error = call(vec![
            Value::GpuTensor(sixteen.clone()),
            Value::String(file.to_string_lossy().into_owned().into()),
        ])
        .unwrap_err();
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        assert!(!file.exists());
    }

    let four = typed_tensor(IntegerStorage::U8(vec![1, 2, 3, 4]), vec![1, 1, 4]);
    let four = crate::builtins::common::gpu_helpers::upload_tensor(owner, &four).unwrap();
    let error = call(vec![
        Value::GpuTensor(four),
        Value::String(
            dir.path()
                .join("cmky.tif")
                .to_string_lossy()
                .into_owned()
                .into(),
        ),
    ])
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);

    let alpha = Value::Tensor(tensor(vec![0.5; 4], vec![2, 2], NumericDType::F64));
    for extension in ["jpg", "bmp"] {
        let file = dir.path().join(format!("alpha.{extension}"));
        let error = call(vec![
            Value::GpuTensor(resident.clone()),
            Value::String(file.to_string_lossy().into_owned().into()),
            Value::String("Alpha".into()),
            alpha.clone(),
        ])
        .unwrap_err();
        assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_OPTION.identifier);
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        assert!(!file.exists());
    }
    // Liveness probes deliberately download. Keep them after every zero-readback
    // assertion so the fixture's own probes cannot pollute admission counters.
    assert!(block_on(owner.download_numeric(&logical_alpha)).is_ok());
    assert!(block_on(owner.download_numeric(&resident)).is_ok());

    let reads = owner.downloads.load(Ordering::SeqCst);
    let unpolled = imwrite_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(resident.clone()),
            path.clone(),
        ]),
    );
    drop(unpolled);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), reads);
    assert!(!dir.path().join("never.png").exists());

    owner.pending.store(true, Ordering::SeqCst);
    let mut future = Box::pin(imwrite_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(resident.clone()),
            path,
        ]),
    ));
    let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
    assert!(std::future::Future::poll(future.as_mut(), &mut context).is_pending());
    assert_eq!(owner.downloads.load(Ordering::SeqCst), reads + 1);
    drop(future);
    assert_eq!(owner.abandoned.load(Ordering::SeqCst), 1);
    assert!(!dir.path().join("never.png").exists());
    owner.pending.store(false, Ordering::SeqCst);
    assert!(block_on(owner.download_numeric(&resident)).is_ok());
}

mod encoding;
