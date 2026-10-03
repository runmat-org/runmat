use super::*;
use futures::executor::block_on;
use image::io::Reader as ImageReader;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostLogicalTensorView, HostNumericTensorView, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_value::IntegerStorage;
use std::fs;
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
    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> anyhow::Result<GpuTensorHandle> {
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
    block_on(imwrite_builtin(args))
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
        Value::String(path.to_string_lossy().into_owned()),
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
        Value::String(path.to_string_lossy().into_owned()),
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
    let path = Value::String(dir.path().join("never.png").to_string_lossy().into_owned());

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
            Value::String(dir.path().join("never.gif").to_string_lossy().into_owned()),
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
        Value::String(png_path.to_string_lossy().into_owned()),
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
        Value::String(dir.path().join("rgb.gif").to_string_lossy().into_owned()),
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
            Value::String(file.to_string_lossy().into_owned()),
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
        Value::String(dir.path().join("cmky.tif").to_string_lossy().into_owned()),
    ])
    .unwrap_err();
    assert_eq!(error.identifier(), IMWRITE_ERROR_INVALID_IMAGE.identifier);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);

    let alpha = Value::Tensor(tensor(vec![0.5; 4], vec![2, 2], NumericDType::F64));
    for extension in ["jpg", "bmp"] {
        let file = dir.path().join(format!("alpha.{extension}"));
        let error = call(vec![
            Value::GpuTensor(resident.clone()),
            Value::String(file.to_string_lossy().into_owned()),
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
    let unpolled = imwrite_builtin(vec![Value::GpuTensor(resident.clone()), path.clone()]);
    drop(unpolled);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), reads);
    assert!(!dir.path().join("never.png").exists());

    owner.pending.store(true, Ordering::SeqCst);
    let mut future = Box::pin(imwrite_builtin(vec![
        Value::GpuTensor(resident.clone()),
        path,
    ]));
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
