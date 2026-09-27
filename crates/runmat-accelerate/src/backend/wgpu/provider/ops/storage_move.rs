//! Bit-preserving storage moves use the allocation's physical element type,
//! independently of the provider's arithmetic precision.
use super::*;
mod permutation;

impl WgpuProvider {
    pub(super) fn get_entry_for_bitwise_move(
        &self,
        handle: &GpuTensorHandle,
        operation: &str,
    ) -> Result<BufferEntry> {
        let entry = self.get_entry_raw(handle)?;
        ensure!(
            handle.shape == entry.shape
                && handle.descriptor.element_type == Some(entry.element_type)
                && handle.descriptor.storage == Some(entry.storage),
            "{operation}: storage move descriptor does not match allocation"
        );
        Ok(entry)
    }

    pub(super) fn register_storage_move_alias(
        &self,
        entry: &BufferEntry,
        shape: Vec<usize>,
    ) -> Result<GpuTensorHandle> {
        self.register_storage_move_buffer(
            entry.buffer.clone(),
            entry,
            shape,
            entry.len,
            entry.allocated_bytes,
        )
    }

    pub(super) fn register_storage_move_buffer(
        &self,
        buffer: Arc<wgpu::Buffer>,
        source: &BufferEntry,
        shape: Vec<usize>,
        len: usize,
        allocated_bytes: u64,
    ) -> Result<GpuTensorHandle> {
        self.register_numeric_buffer(
            buffer,
            NumericBufferRegistration {
                shape,
                len,
                physical_element_type: source.element_type,
                storage: source.storage,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        )
    }
}
