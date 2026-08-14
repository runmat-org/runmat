use std::collections::{BTreeMap, HashSet};

use runmat_accelerate_api::{
    handle_integer_type, handle_precision, handle_storage, GpuHandleIdentity, GpuTensorHandle,
    GpuTensorStorage, ProviderPrecision,
};
use runmat_value::{IntValue, Value};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Authoritative-copy state for a value considered by local placement.
pub enum CoherencyState {
    /// Only the host copy is current.
    Host,
    /// Only the copy on `device_id` is current.
    Device { device_id: u32 },
    /// Host and device copies agree after a host-to-device transfer.
    MirroredHostClean { device_id: u32 },
    /// Host and device copies agree after a device-to-host transfer.
    MirroredDeviceClean { device_id: u32 },
    /// The authoritative copy cannot be proven.
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Coherency state plus the bytes required to move the complete value.
pub struct CoherencyRecord {
    pub state: CoherencyState,
    pub bytes: u64,
}

impl CoherencyRecord {
    /// Create a host-authoritative record.
    pub fn host(bytes: u64) -> Self {
        Self {
            state: CoherencyState::Host,
            bytes,
        }
    }

    /// Record a completed host-to-device transfer.
    pub fn uploaded(mut self, device_id: u32) -> Self {
        self.state = CoherencyState::MirroredHostClean { device_id };
        self
    }

    /// Invalidate any device copy after host mutation.
    pub fn host_mutated(mut self) -> Self {
        self.state = CoherencyState::Host;
        self
    }

    /// Invalidate any host or other-device copy after device mutation.
    pub fn device_mutated(mut self, device_id: u32) -> Self {
        self.state = CoherencyState::Device { device_id };
        self
    }

    /// Record a completed device-to-host transfer.
    pub fn downloaded(mut self, device_id: u32) -> Self {
        self.state = CoherencyState::MirroredDeviceClean { device_id };
        self
    }

    /// Discard coherency knowledge after an opaque mutation or alias escape.
    pub fn invalidate(mut self) -> Self {
        self.state = CoherencyState::Unknown;
        self
    }

    /// Bytes that must move before execution on `device_id`.
    pub fn upload_bytes(self, device_id: u32) -> u64 {
        match self.state {
            CoherencyState::Device {
                device_id: resident,
            }
            | CoherencyState::MirroredHostClean {
                device_id: resident,
            }
            | CoherencyState::MirroredDeviceClean {
                device_id: resident,
            } if resident == device_id => 0,
            _ => self.bytes,
        }
    }

    /// Bytes that must move before host execution.
    pub fn download_bytes(self) -> u64 {
        match self.state {
            CoherencyState::Host
            | CoherencyState::MirroredHostClean { .. }
            | CoherencyState::MirroredDeviceClean { .. } => 0,
            _ => self.bytes,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidencySummary {
    pub(crate) host_bytes: u64,
    pub(crate) provider_bytes: BTreeMap<u32, u64>,
    pub(crate) unknown_values: u64,
}

impl ResidencySummary {
    pub(crate) fn required_upload_bytes(&self, device_id: u32) -> u64 {
        if self
            .provider_bytes
            .keys()
            .all(|resident| *resident == device_id)
        {
            self.host_bytes
        } else {
            self.host_bytes.saturating_add(
                self.provider_bytes
                    .iter()
                    .filter(|(resident, _)| **resident != device_id)
                    .map(|(_, bytes)| *bytes)
                    .fold(0_u64, u64::saturating_add),
            )
        }
    }
}

pub(crate) fn summarize_values(values: &[&Value]) -> ResidencySummary {
    let mut summary = ResidencySummary::default();
    let mut visited_gpu = HashSet::new();
    let mut visited_handles = HashSet::new();
    for value in values {
        summarize_value(value, &mut summary, &mut visited_gpu, &mut visited_handles);
    }
    summary
}

fn summarize_value(
    value: &Value,
    summary: &mut ResidencySummary,
    visited_gpu: &mut HashSet<GpuHandleIdentity>,
    visited_handles: &mut HashSet<usize>,
) {
    match value {
        Value::Int(value) => add_host(summary, int_byte_size(value)),
        Value::Num(_) => add_host(summary, 8),
        Value::Complex(_, _) => add_host(summary, 16),
        Value::Bool(_) => add_host(summary, 1),
        Value::LogicalArray(array) => add_host(summary, array.len()),
        Value::Tensor(tensor) => add_host(
            summary,
            tensor
                .len()
                .saturating_mul(tensor.numeric_dtype().byte_size()),
        ),
        Value::SparseTensor(tensor) => {
            let index_bytes = tensor
                .col_ptrs
                .len()
                .saturating_add(tensor.row_indices.len())
                .saturating_mul(std::mem::size_of::<usize>());
            let value_bytes = tensor.nnz().saturating_mul(tensor.value_byte_size());
            add_host(summary, index_bytes.saturating_add(value_bytes));
        }
        Value::ComplexTensor(tensor) => add_host(
            summary,
            tensor
                .len()
                .saturating_mul(tensor.numeric_dtype().byte_size())
                .saturating_mul(2),
        ),
        Value::GpuTensor(handle) => {
            let identity = runmat_accelerate_api::handle_identity(handle);
            if visited_gpu.insert(identity) {
                let bytes = gpu_handle_bytes(handle).unwrap_or_else(|| {
                    summary.unknown_values = summary.unknown_values.saturating_add(1);
                    0
                });
                let entry = summary.provider_bytes.entry(handle.device_id).or_default();
                *entry = entry.saturating_add(bytes);
            }
        }
        Value::Cell(cell) => {
            for element in &cell.data {
                summarize_value(element, summary, visited_gpu, visited_handles);
            }
        }
        Value::Struct(value) => {
            for element in value.fields.values() {
                summarize_value(element, summary, visited_gpu, visited_handles);
            }
        }
        Value::Object(value) => {
            for element in value.properties.values() {
                summarize_value(element, summary, visited_gpu, visited_handles);
            }
        }
        Value::ObjectArray(array) => {
            for element in array.data() {
                summarize_value(element, summary, visited_gpu, visited_handles);
            }
        }
        Value::Closure(closure) => {
            for element in &closure.captures {
                summarize_value(element, summary, visited_gpu, visited_handles);
            }
        }
        Value::OutputList(values) => {
            for element in values {
                summarize_value(element, summary, visited_gpu, visited_handles);
            }
        }
        Value::HandleObject(handle) => {
            let address = runmat_gc::gc_handle_addr(&handle.target);
            if visited_handles.insert(address)
                && runmat_gc::gc_with_value(&handle.target, |target| {
                    summarize_value(target, summary, visited_gpu, visited_handles);
                })
                .is_err()
            {
                summary.unknown_values = summary.unknown_values.saturating_add(1);
            }
        }
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Foreign(_) => {
            summary.unknown_values = summary.unknown_values.saturating_add(1);
        }
    }
}

fn add_host(summary: &mut ResidencySummary, bytes: usize) {
    summary.host_bytes = summary
        .host_bytes
        .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
}

fn int_byte_size(value: &IntValue) -> usize {
    match value {
        IntValue::I8(_) | IntValue::U8(_) => 1,
        IntValue::I16(_) | IntValue::U16(_) => 2,
        IntValue::I32(_) | IntValue::U32(_) => 4,
        IntValue::I64(_) | IntValue::U64(_) => 8,
    }
}

fn gpu_handle_bytes(handle: &GpuTensorHandle) -> Option<u64> {
    let elements = handle.shape.iter().try_fold(1_u64, |total, dimension| {
        total.checked_mul(u64::try_from(*dimension).ok()?)
    })?;
    let width = handle_integer_type(handle)
        .and_then(|kind| u64::try_from(kind.element_size()).ok())
        .or_else(|| {
            handle_precision(handle).map(|precision| match precision {
                ProviderPrecision::F32 => 4,
                ProviderPrecision::F64 => 8,
            })
        })?;
    let lanes = match handle_storage(handle) {
        GpuTensorStorage::Real => 1,
        GpuTensorStorage::ComplexInterleaved => 2,
    };
    elements.checked_mul(width)?.checked_mul(lanes)
}

#[cfg(test)]
mod tests {
    use runmat_accelerate_api::{set_handle_precision, GpuTensorHandle, ProviderPrecision};
    use runmat_value::{CellArray, Tensor};

    use super::*;

    #[test]
    fn mutation_invalidates_the_other_copy() {
        let record = CoherencyRecord::host(128).uploaded(7);
        assert_eq!(record.upload_bytes(7), 0);
        assert_eq!(record.host_mutated().upload_bytes(7), 128);
        assert_eq!(record.device_mutated(7).download_bytes(), 128);
        assert_eq!(record.device_mutated(7).downloaded(7).download_bytes(), 0);
        assert_eq!(record.invalidate().state, CoherencyState::Unknown);
    }

    #[test]
    fn nested_values_are_counted_and_aliased_gpu_handles_are_deduplicated() {
        let host = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let gpu = GpuTensorHandle {
            shape: vec![2, 2],
            device_id: 9,
            buffer_id: 41,
        };
        set_handle_precision(&gpu, ProviderPrecision::F32);
        let cell = Value::Cell(
            CellArray::new(
                vec![host, Value::GpuTensor(gpu.clone()), Value::GpuTensor(gpu)],
                1,
                3,
            )
            .unwrap(),
        );
        let summary = summarize_values(&[&cell]);
        assert_eq!(summary.host_bytes, 16);
        assert_eq!(summary.provider_bytes.get(&9), Some(&16));
        assert_eq!(summary.required_upload_bytes(9), 16);
    }
}
