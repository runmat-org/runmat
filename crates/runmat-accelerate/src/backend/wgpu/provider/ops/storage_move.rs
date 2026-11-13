//! Bit-preserving storage moves use the allocation's physical element type,
//! independently of the provider's arithmetic precision.
use super::*;
#[path = "storage_move/flip.rs"]
mod flip;
#[path = "storage_move/permutation.rs"]
mod permutation;

impl WgpuProvider {
    pub(super) fn get_entry_for_bitwise_move(
        &self,
        handle: &GpuTensorHandle,
        operation: &str,
    ) -> Result<BufferEntry> {
        let mut entry = self.get_entry_raw(handle)?;
        ensure!(
            handle.shape() == entry.shape
                && handle.physical().element_type() == entry.element_type
                && handle.physical().storage() == entry.storage,
            "{operation}: storage move descriptor does not match allocation"
        );
        // Bit-preserving moves retain the allocation's encoding authority, not
        // a facade's logical/numeric interpretation of those same bytes.
        entry.logical = self.allocations.requires_logical_encoding(handle)?;
        Ok(entry)
    }

    pub(super) fn register_storage_move_buffer(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        source: &BufferEntry,
        shape: Vec<usize>,
        len: usize,
        allocated_bytes: u64,
    ) -> Result<GpuTensorHandle> {
        self.register_storage_move_buffer_admitted(
            buffer,
            source,
            shape,
            len,
            allocated_bytes,
            None,
        )
    }

    pub(super) fn register_storage_move_buffer_admitted(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        source: &BufferEntry,
        shape: Vec<usize>,
        len: usize,
        allocated_bytes: u64,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        self.register_numeric_buffer_produced_admitted(
            buffer,
            NumericBufferRegistration {
                shape,
                len,
                physical_element_type: source.element_type,
                storage: source.storage,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
            source.logical,
            admission,
        )
    }
}
