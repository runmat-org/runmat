use super::*;

#[derive(Clone, Copy)]
enum Completion {
    Fresh,
    Alias,
    Error,
    #[cfg(panic = "unwind")]
    Panic,
}
struct Mutating {
    inner: Legacy,
    borrowed: GpuTensorHandle,
    completion: Completion,
}
impl AccelProvider for Mutating {
    fn device_id(&self) -> u32 {
        self.inner.device
    }
    fn device_info(&self) -> String {
        "mutating upload fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.inner.free(handle)
    }
    fn upload_numeric_raw(&self, _: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        clear_handle_metadata(&self.borrowed);
        set_handle_logical(&self.borrowed, true);
        set_handle_class_identity(&self.borrowed, standard::CHAR.owned());
        record_handle_transpose(&self.borrowed, 7, 8);
        match self.completion {
            Completion::Fresh => Ok(self.inner.returned.clone()),
            Completion::Alias => Ok(self.borrowed.clone()),
            Completion::Error => anyhow::bail!("injected raw upload failure"),
            #[cfg(panic = "unwind")]
            Completion::Panic => panic!("injected raw upload unwind"),
        }
    }
}

fn run_mutation(completion: Completion, initially_logical: bool) {
    let inner = Legacy::new();
    let mut borrowed = inner.returned.clone();
    borrowed.buffer_id = 99;
    mark_handle_explicit(&mut borrowed);
    if initially_logical {
        set_handle_logical(&borrowed, true);
        set_handle_logical(&borrowed, true);
        record_handle_transpose(&borrowed, 1, 1);
    }
    let snapshot = snapshot_handle_metadata(&borrowed);
    let provider = Mutating {
        inner,
        borrowed: borrowed.clone(),
        completion,
    };
    let call = || upload_numeric_protecting(&provider, &double(), &[&borrowed, &borrowed]);
    match completion {
        Completion::Fresh => {
            assert!(call().is_ok());
        }
        Completion::Alias | Completion::Error => {
            assert!(call().is_err());
        }
        #[cfg(panic = "unwind")]
        Completion::Panic => {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).is_err());
        }
    }
    assert_eq!(snapshot_handle_metadata(&borrowed), snapshot);
    assert_eq!(
        handle_provenance(&borrowed),
        Some(GpuHandleProvenance::Explicit)
    );
    assert!(!provider
        .inner
        .freed
        .lock()
        .unwrap()
        .contains(&(borrowed.device_id, borrowed.buffer_id)));
    clear_handle_metadata(&borrowed);
}

#[test]
fn numeric_upload_restores_every_borrowed_annotation_on_success_alias_and_error() {
    for completion in [Completion::Fresh, Completion::Alias, Completion::Error] {
        for logical in [false, true] {
            run_mutation(completion, logical);
        }
    }
}

#[test]
#[cfg(panic = "unwind")]
fn numeric_upload_restores_borrowed_annotations_during_unwind() {
    for logical in [false, true] {
        run_mutation(Completion::Panic, logical);
    }
}

#[test]
fn numeric_upload_rejects_conflicting_fresh_annotations_and_cleans_them() {
    enum Annotation {
        Logical,
        Class,
        Transpose,
    }
    for annotation in [
        Annotation::Logical,
        Annotation::Class,
        Annotation::Transpose,
    ] {
        let provider = Legacy::new();
        match annotation {
            Annotation::Logical => set_handle_logical(&provider.returned, true),
            Annotation::Class => {
                set_handle_class_identity(&provider.returned, standard::SINGLE.owned())
            }
            Annotation::Transpose => record_handle_transpose(&provider.returned, 1, 1),
        }
        let error = upload_numeric_protecting(&provider, &double(), &[]).unwrap_err();
        assert!(is_invalid_numeric_upload_result(&error));
        assert_eq!(provider.freed.lock().unwrap().len(), 1);
        assert!(!handle_is_logical(&provider.returned));
        assert!(!handle_is_transposed(&provider.returned));
        assert_eq!(
            handle_class_identity(&provider.returned),
            Some(standard::DOUBLE.owned())
        );
    }
}
