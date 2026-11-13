//! Shared owner-authoritative class admission for resident feed-forward inputs.
use super::*;

pub(super) struct NativeRealInputAdmission {
    pub(super) logical_producer: bool,
    physical_class_compatible: bool,
}

impl NativeRealInputAdmission {
    pub(super) fn is_ready(&self) -> bool {
        !self.logical_producer && self.physical_class_compatible
    }
}

pub(super) fn classify_native_real_input(
    provider: &WgpuProvider,
    input: &GpuTensorHandle,
    entry: &super::super::backend_types::BufferEntry,
) -> Result<NativeRealInputAdmission> {
    Ok(NativeRealInputAdmission {
        logical_producer: provider.allocations.requires_logical_encoding(input)?,
        physical_class_compatible: provider.supports_native_real_element_type(entry.element_type),
    })
}
