use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{GpuTensorDescriptor, GpuTensorHandle, GpuTensorStorage, NumericElementType};
mod export;
pub use export::{export_native_device_buffer, OwnedNativeDeviceBuffer};

/// Native device ABI exposed by an acceleration provider to compatible
/// extension code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeDeviceApi {
    Cuda,
}

impl NativeDeviceApi {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
        }
    }
}

impl fmt::Display for NativeDeviceApi {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Access requested for a provider-owned native buffer export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDeviceAccess {
    ReadOnly,
    ReadWrite,
}

/// Initialization requested for a newly allocated native device buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDeviceInitialization {
    Uninitialized,
    Zeroed,
}

/// Admitted native ABI class, not a claim about values in uninitialized memory.
/// Logical native arrays are byte-addressable; callers cannot supply a
/// contradictory independent element type and logical flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDeviceClass {
    Numeric(NumericElementType),
    Logical,
}
impl NativeDeviceClass {
    pub const fn element_type(self) -> NumericElementType {
        match self {
            Self::Numeric(element) => element,
            Self::Logical => NumericElementType::U8,
        }
    }
    pub const fn is_logical(self) -> bool {
        matches!(self, Self::Logical)
    }
    pub fn validate_storage(self, storage: GpuTensorStorage) -> anyhow::Result<()> {
        if self.is_logical() && storage != GpuTensorStorage::Real {
            anyhow::bail!("native logical arrays require real byte storage");
        }
        Ok(())
    }
}

/// Component selected from an interleaved complex native buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDeviceComponent {
    Real,
    Imaginary,
}

/// Stable identity of the provider device, context, and stream used by a
/// native extension invocation.
///
/// The context and stream fields are opaque identities. They are suitable for
/// affinity checks and diagnostics, but consumers must not reinterpret them as
/// callable host pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDeviceContext {
    pub api: NativeDeviceApi,
    pub provider_device_id: u32,
    pub device_ordinal: u32,
    pub context_identity: u64,
    pub stream_identity: u64,
}

/// Provider-owned native buffer made available to a compatible extension.
///
/// `device_address` is meaningful only while the originating handle and the
/// invocation's context guard remain alive. The provider remains the allocator
/// and synchronization authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDeviceBuffer {
    pub context: NativeDeviceContext,
    pub device_address: u64,
    pub byte_length: usize,
    pub descriptor: GpuTensorDescriptor,
}

/// Scope guard returned after a provider has made its native context current.
///
/// The guard is intentionally not transferable between threads: entering and
/// leaving a native device context must occur on the same execution lane.
pub struct NativeDeviceContextGuard {
    release: Option<Box<dyn FnOnce() -> anyhow::Result<()>>>,
    _thread_affinity: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl NativeDeviceContextGuard {
    pub fn new(release: impl FnOnce() -> anyhow::Result<()> + 'static) -> Self {
        Self {
            release: Some(Box::new(release)),
            _thread_affinity: std::marker::PhantomData,
        }
    }

    pub fn inert() -> Self {
        Self::new(|| Ok(()))
    }

    /// Leave the native context and return any teardown error.
    ///
    /// Drop remains a best-effort safety net. Native ABI boundaries should
    /// call this method so a context-pop failure becomes part of the
    /// invocation result.
    pub fn leave(mut self) -> anyhow::Result<()> {
        self.release.take().map_or(Ok(()), |release| release())
    }
}

impl fmt::Debug for NativeDeviceContextGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeDeviceContextGuard")
            .finish_non_exhaustive()
    }
}

impl Drop for NativeDeviceContextGuard {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release();
        }
    }
}

/// Validate a provider result before exposing it as native device memory.
pub fn validate_native_device_buffer(
    handle: &GpuTensorHandle,
    buffer: &NativeDeviceBuffer,
    api: NativeDeviceApi,
) -> anyhow::Result<()> {
    if buffer.context.api != api {
        anyhow::bail!(
            "provider exported {} memory for a {} request",
            buffer.context.api,
            api
        );
    }
    if buffer.context.provider_device_id != handle.device_id() {
        anyhow::bail!(
            "provider exported device {} for handle device {}",
            buffer.context.provider_device_id,
            handle.device_id()
        );
    }
    if buffer.device_address == 0 && buffer.byte_length != 0 {
        anyhow::bail!("provider exported a null address for a non-empty native buffer");
    }
    if buffer.descriptor.element_type != Some(handle.physical().element_type())
        || buffer.descriptor.storage != Some(handle.physical().storage())
    {
        anyhow::bail!("provider native-buffer descriptor does not match its handle");
    }
    let element_type = handle.physical().element_type();
    let storage = handle.physical().storage();
    let expected = native_buffer_byte_length(handle.shape(), element_type, storage)?;
    if buffer.byte_length < expected {
        anyhow::bail!(
            "provider native buffer has {} bytes; shape {:?} requires at least {}",
            buffer.byte_length,
            handle.shape(),
            expected
        );
    }
    Ok(())
}

/// Validate a provider-created numeric handle before another subsystem takes
/// ownership of it.
///
/// Native extension adapters use this after allocation and device-side copy
/// operations. A provider remains responsible for the storage, but the caller
/// must not accept a handle whose identity or numeric layout differs from the
/// operation it requested.
pub fn validate_native_device_result(
    handle: &GpuTensorHandle,
    provider_device_id: u32,
    shape: &[usize],
    class: NativeDeviceClass,
    storage: GpuTensorStorage,
) -> anyhow::Result<()> {
    class.validate_storage(storage)?;
    let element_type = class.element_type();
    if handle.device_id() != provider_device_id {
        anyhow::bail!(
            "provider returned device {} for an operation on device {}",
            handle.device_id(),
            provider_device_id
        );
    }
    if handle.shape() != shape {
        anyhow::bail!(
            "provider returned shape {:?}; requested shape was {:?}",
            handle.shape(),
            shape
        );
    }
    if handle.physical().element_type() != element_type
        || handle.physical().storage() != storage
        || handle.view().is_logical() != class.is_logical()
    {
        anyhow::bail!(
            "provider returned numeric layout {:?}; requested {:?} {:?}",
            handle.descriptor(),
            element_type,
            storage
        );
    }
    native_buffer_byte_length(shape, element_type, storage)?;
    Ok(())
}

pub fn native_buffer_byte_length(
    shape: &[usize],
    element_type: NumericElementType,
    storage: GpuTensorStorage,
) -> anyhow::Result<usize> {
    let logical_elements = if shape.contains(&0) {
        Some(0)
    } else {
        shape
            .iter()
            .try_fold(1usize, |length, dimension| length.checked_mul(*dimension))
    };
    let logical_elements =
        logical_elements.ok_or_else(|| anyhow::anyhow!("native device shape overflows usize"))?;
    let lanes = match storage {
        GpuTensorStorage::Real => logical_elements,
        GpuTensorStorage::ComplexInterleaved => logical_elements
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("native complex device length overflows usize"))?,
    };
    lanes
        .checked_mul(element_type.element_size())
        .ok_or_else(|| anyhow::anyhow!("native device byte length overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle() -> GpuTensorHandle {
        crate::test_support::handle(
            41,
            vec![2, 3],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
    }

    #[test]
    fn provider_result_validation_rejects_identity_and_layout_drift() {
        let expected = handle();
        validate_native_device_result(
            &expected,
            41,
            &[2, 3],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
        )
        .unwrap();

        let wrong_device = crate::test_support::handle(
            42,
            vec![2, 3],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        );
        assert!(validate_native_device_result(
            &wrong_device,
            41,
            &[2, 3],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
        )
        .is_err());

        let wrong_shape = expected
            .with_view(crate::GpuTensorView::contiguous(vec![3, 2]))
            .unwrap();
        assert!(validate_native_device_result(
            &wrong_shape,
            41,
            &[2, 3],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
        )
        .is_err());

        let wrong_layout = crate::test_support::handle(
            41,
            vec![2, 3],
            NumericElementType::F32,
            GpuTensorStorage::ComplexInterleaved,
        );
        assert!(validate_native_device_result(
            &wrong_layout,
            41,
            &[2, 3],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
        )
        .is_err());
    }

    #[test]
    fn explicit_context_leave_reports_teardown_failures_once() {
        let attempts = std::rc::Rc::new(std::cell::Cell::new(0));
        let observed = std::rc::Rc::clone(&attempts);
        let guard = NativeDeviceContextGuard::new(move || {
            observed.set(observed.get() + 1);
            anyhow::bail!("fixture context-pop failure")
        });
        let error = guard.leave().unwrap_err();
        assert_eq!(error.to_string(), "fixture context-pop failure");
        assert_eq!(attempts.get(), 1);
    }

    #[test]
    fn logical_native_class_preserves_byte_encoding_and_rejects_numeric_relabeling() {
        assert_eq!(
            NativeDeviceClass::Logical.element_type(),
            NumericElementType::U8
        );
        assert!(NativeDeviceClass::Logical
            .validate_storage(GpuTensorStorage::ComplexInterleaved)
            .is_err());
        let (logical, _) = crate::test_support::owned(
            41,
            vec![2],
            NumericElementType::U8,
            GpuTensorStorage::Real,
            true,
        );
        validate_native_device_result(
            &logical,
            41,
            &[2],
            NativeDeviceClass::Logical,
            GpuTensorStorage::Real,
        )
        .unwrap();
        let numeric = crate::test_support::handle(
            41,
            vec![2],
            NumericElementType::U8,
            GpuTensorStorage::Real,
        );
        assert!(validate_native_device_result(
            &numeric,
            41,
            &[2],
            NativeDeviceClass::Logical,
            GpuTensorStorage::Real
        )
        .is_err());
        assert!(validate_native_device_result(
            &logical,
            41,
            &[2],
            NativeDeviceClass::Numeric(NumericElementType::U8),
            GpuTensorStorage::Real
        )
        .is_err());
    }
}
