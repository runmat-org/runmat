use super::*;

struct CountedAccel {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl AccelProvider for CountedAccel {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn advertised_revoked_lease_rejects_read_and_write_before_gather_or_effect() {
    let _accel = crate::builtins::common::test_support::accel_test_lock();
    let _filesystem = fs_test_guard();
    let selected = Arc::new(RevocableProvider {
        memory: runmat_filesystem::MemoryFsProvider::with_current_dir(selected_dir()),
        valid: Arc::new(AtomicBool::new(false)),
        opens: Arc::new(AtomicUsize::new(0)),
        reads: Arc::new(AtomicUsize::new(0)),
        writes: Arc::new(AtomicUsize::new(0)),
    });
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");

    assert!(XmlFsPath::capture(selected_path("revoked.xml")).is_err());
    assert!(XmlPathOwner::capture().is_err());

    let accelerator = Box::leak(Box::new(CountedAccel {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(accelerator));
    let resident = Value::GpuTensor(
        accelerator
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .expect("resident fixture"),
    );

    let read_error = futures::executor::block_on(xmlread_builtin(None, vec![resident.clone()]))
        .expect_err("revoked read source");
    assert_eq!(read_error.identifier(), Some("RunMat:xmlread:IoFailed"));
    assert!(std::error::Error::source(&read_error).is_some());

    let struct_error = futures::executor::block_on(readstruct_builtin(
        None,
        vec![
            Value::from("revoked.xml"),
            Value::from("AttributeSuffix"),
            resident.clone(),
        ],
    ))
    .expect_err("revoked readstruct source");
    assert_eq!(
        struct_error.identifier(),
        Some("RunMat:readstruct:IoFailed")
    );
    assert!(std::error::Error::source(&struct_error).is_some());

    let write_error = futures::executor::block_on(xmlwrite_builtin(
        None,
        vec![Value::from("revoked.xml"), resident],
    ))
    .expect_err("revoked destination");
    assert_eq!(write_error.identifier(), Some("RunMat:xmlwrite:IoFailed"));
    assert!(std::error::Error::source(&write_error).is_some());

    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(selected.opens.load(Ordering::SeqCst), 0);
    assert_eq!(selected.reads.load(Ordering::SeqCst), 0);
    assert_eq!(selected.writes.load(Ordering::SeqCst), 0);
}
