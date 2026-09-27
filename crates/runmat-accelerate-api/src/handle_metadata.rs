//! Scoped preservation of mutable annotations on borrowed resident handles.
use crate::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuHandleMetadataSnapshot {
    descriptor: GpuTensorDescriptor,
    logical: bool,
    logical_hits: Option<u64>,
    transpose: Option<TransposeInfo>,
    class_annotation: Option<ClassIdentity>,
}

pub fn snapshot_handle_metadata(handle: &GpuTensorHandle) -> GpuHandleMetadataSnapshot {
    GpuHandleMetadataSnapshot {
        descriptor: handle.descriptor,
        logical: handle_is_logical(handle),
        logical_hits: handle_logical_hits_for_handle(handle),
        transpose: handle_transpose_info(handle),
        class_annotation: HANDLE_CLASS_IDENTITIES
            .read()
            .ok()
            .and_then(|classes| classes.get(&handle_identity(handle)).cloned()),
    }
}

pub fn restore_handle_metadata(handle: &GpuTensorHandle, snapshot: &GpuHandleMetadataSnapshot) {
    debug_assert_eq!(handle.descriptor, snapshot.descriptor);
    let identity = handle_identity(handle);
    if let Ok(mut logical) = LOGICAL_HANDLES.write() {
        if snapshot.logical {
            logical.insert(identity);
        } else {
            logical.remove(&identity);
        }
    }
    if let Ok(mut hits) = LOGICAL_HANDLE_HITS.write() {
        if let Some(count) = snapshot.logical_hits {
            hits.insert(identity, count);
        } else {
            hits.remove(&identity);
        }
    }
    match snapshot.transpose {
        Some(info) => record_handle_transpose(handle, info.base_rows, info.base_cols),
        None => clear_handle_transpose(handle),
    }
    match &snapshot.class_annotation {
        Some(class) => set_handle_class_identity(handle, class.clone()),
        None => clear_handle_class_identity(handle),
    }
    mark_residency(handle);
}

/// Restores globally keyed annotations even when a provider call fails or unwinds.
pub struct HandleMetadataRestoreGuard<'a> {
    handle: &'a GpuTensorHandle,
    snapshot: GpuHandleMetadataSnapshot,
}
impl<'a> HandleMetadataRestoreGuard<'a> {
    pub fn original_descriptor(&self) -> &GpuTensorDescriptor {
        &self.snapshot.descriptor
    }

    pub fn was_logical(&self) -> bool {
        self.snapshot.logical
    }

    pub fn new(handle: &'a GpuTensorHandle) -> Self {
        Self {
            handle,
            snapshot: snapshot_handle_metadata(handle),
        }
    }
}
impl Drop for HandleMetadataRestoreGuard<'_> {
    fn drop(&mut self) {
        restore_handle_metadata(self.handle, &self.snapshot);
    }
}
