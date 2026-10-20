use super::*;
use runmat_value::CellArray;

#[test]
fn filewrite_is_gated_before_filesystem_side_effects() {
    let target = unique_path("filewrite_strict_gate");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(filewrite_builtin(
        Value::from(target.to_string_lossy().to_string()),
        Value::from("must not be written"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("strict compatibility mode must reject RunMat-native filewrite");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FilewriteNativeExtension")
    );
    assert!(!target.exists());
}

#[test]
fn filewrite_integer_metadata_covers_every_native_class() {
    let capability = &FILEWRITE_INTEGER_CAPABILITIES[0];
    assert_eq!(capability.inputs[0].classes.len(), 8);
    assert_eq!(
        capability.backend,
        BuiltinIntegerBackendRule::GatherFallback
    );
    assert_eq!(FILEWRITE_EXTENSIONS[0].id, "filewrite-runmat-native");
}

#[test]
fn filewrite_gathers_resident_integer_bytes_before_host_io() {
    test_support::with_test_provider(|provider| {
        let values = [0_u64, 127, 255];
        let handle = provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
                shape: &[1, 3],
            })
            .expect("upload integer bytes");
        let target = unique_path("filewrite_resident_integer");
        let result = run_filewrite(
            Value::from(target.to_string_lossy().to_string()),
            Value::GpuTensor(handle),
            Vec::new(),
        )
        .expect("resident integer filewrite");
        assert_eq!(result, Value::Num(3.0));
        assert_eq!(test_support::fs::read(&target).unwrap(), vec![0, 127, 255]);
        test_support::fs::remove_file(target).unwrap();
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FILEWRITE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"count = filewrite(filename, data)"));
    assert!(labels.contains(&"count = filewrite(filename, data, encoding)"));
    assert!(labels.contains(&"count = filewrite(filename, data, \"Encoding\", encoding)"));
    assert!(labels.contains(&"count = filewrite(filename, data, ..., \"WriteMode\", mode)"));
}

#[test]
fn invalid_host_path_options_and_nested_data_precede_resident_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident data");
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let resident = Value::GpuTensor(allocation.handle.clone());
    let bad_path = run_filewrite(Value::Num(1.0), resident.clone(), Vec::new())
        .expect_err("invalid path before data gather");
    assert_eq!(
        bad_path.identifier(),
        FILEWRITE_ERROR_INVALID_INPUT.identifier
    );

    let target = unique_path("filewrite_preflight");
    let bad_option = run_filewrite(
        Value::from(target.to_string_lossy().to_string()),
        resident.clone(),
        vec![Value::Num(23.0)],
    )
    .expect_err("invalid option before data gather");
    assert_eq!(
        bad_option.identifier(),
        FILEWRITE_ERROR_INVALID_OPTION.identifier
    );
    assert!(!target.exists());

    let cell = CellArray::new(vec![resident], 1, 1).expect("nested resident data");
    let bad_data = run_filewrite(
        Value::from(target.to_string_lossy().to_string()),
        Value::Cell(cell),
        Vec::new(),
    )
    .expect_err("cell payload before recursive gather");
    assert_eq!(
        bad_data.identifier(),
        FILEWRITE_ERROR_INVALID_DATA.identifier
    );
    assert_eq!(allocation.probe.retirements(), 0);
    assert!(!target.exists());
}

#[test]
fn invalid_scalar_bytes_preserve_the_invalid_data_envelope() {
    let target = unique_path("filewrite_invalid_scalar");
    for invalid in [
        Value::Num(256.0),
        Value::Int(runmat_value::IntValue::I64(-1)),
    ] {
        let error = run_filewrite(
            Value::from(target.to_string_lossy().to_string()),
            invalid,
            Vec::new(),
        )
        .expect_err("invalid scalar byte");
        assert_eq!(error.identifier(), FILEWRITE_ERROR_INVALID_DATA.identifier);
        assert!(!target.exists());
    }
}

struct PausedByteDownload {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    released: std::sync::atomic::AtomicBool,
    downloads: std::sync::atomic::AtomicUsize,
}

impl runmat_accelerate_api::AccelProvider for PausedByteDownload {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "filewrite owner-capture fixture".into()
    }

    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected double download") })
    }

    fn download_numeric<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        use std::sync::atomic::Ordering;
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            futures::future::poll_fn(|_| {
                if self.released.load(Ordering::SeqCst) {
                    std::task::Poll::Ready(())
                } else {
                    std::task::Poll::Pending
                }
            })
            .await;
            Ok(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::U8(vec![42]),
                shape: vec![1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
        })
    }
}

#[test]
fn pending_download_keeps_captured_filesystem_owner_and_cwd() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    // Filesystem replacement is process-wide. Execute this witness in its own
    // test process so unrelated parallel tests retain their filesystem owner.
    #[cfg(not(target_arch = "wasm32"))]
    {
        const CHILD_MARKER: &str = "RUNMAT_FILEWRITE_OWNER_CAPTURE_TEST_CHILD";
        if std::env::var_os(CHILD_MARKER).is_none() {
            let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "builtins::io::filetext::filewrite::tests::controls::pending_download_keeps_captured_filesystem_owner_and_cwd",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD_MARKER, "1")
            .output()
            .expect("isolated filesystem owner witness");
            assert!(
                output.status.success(),
                "owner witness failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
            return;
        }
    }

    let _fs_lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/selected",
    ));
    let replacement = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/replacement",
    ));
    let _selected_guard = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir("/selected").expect("selected cwd");
    let provider = Box::leak(Box::new(PausedByteDownload {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
        released: AtomicBool::new(false),
        downloads: AtomicUsize::new(0),
    }));
    let _gpu_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::U8,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident byte");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut pending = Box::pin(filewrite_builtin(
        Value::from("owner.bin"),
        Value::GpuTensor(resident.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    assert!(matches!(
        std::future::Future::poll(
            pending.as_mut(),
            &mut std::task::Context::from_waker(futures::task::noop_waker_ref()),
        ),
        std::task::Poll::Pending
    ));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    let _replacement_guard = runmat_filesystem::replace_provider(replacement.clone());
    runmat_filesystem::set_current_dir("/replacement").expect("replacement cwd");
    provider.released.store(true, Ordering::SeqCst);
    assert_eq!(
        futures::executor::block_on(pending).expect("write on selected owner"),
        Value::Num(1.0)
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(
        selected.read_project_path("/selected/owner.bin").unwrap(),
        vec![42]
    );
    assert!(replacement
        .read_project_path("/replacement/owner.bin")
        .is_err());
    assert_eq!(resident.probe.retirements(), 0);
}
