use serde::ser::{Serialize, SerializeMap, Serializer};
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

/// One-device summaries need no tree node. A second device promotes to the
/// ordinary sorted map, while serialization retains the original map bytes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProviderByteSummary {
    one: Option<(u32, u64)>,
    many: BTreeMap<u32, u64>,
}

impl ProviderByteSummary {
    pub(crate) fn insert(&mut self, device: u32, bytes: u64) -> Option<u64> {
        if let Some((old_device, old_bytes)) = self.one {
            if old_device == device {
                self.one = Some((device, bytes));
                return Some(old_bytes);
            }
            self.many.insert(old_device, old_bytes);
            self.one = None;
        }
        if self.many.is_empty() {
            self.one = Some((device, bytes));
            None
        } else {
            self.many.insert(device, bytes)
        }
    }

    pub(crate) fn add(&mut self, device: u32, bytes: u64) {
        let old = self.get(&device).copied().unwrap_or(0);
        self.insert(device, old.saturating_add(bytes));
    }

    pub(crate) fn get(&self, device: &u32) -> Option<&u64> {
        self.one
            .as_ref()
            .filter(|(stored, _)| stored == device)
            .map(|(_, bytes)| bytes)
            .or_else(|| self.many.get(device))
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u32, &u64)> {
        self.one
            .iter()
            .map(|(device, bytes)| (device, bytes))
            .chain(self.many.iter())
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &u32> {
        self.iter().map(|(device, _)| device)
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &u64> {
        self.iter().map(|(_, bytes)| bytes)
    }
}

impl Serialize for ProviderByteSummary {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map =
            serializer.serialize_map(Some(self.one.is_some() as usize + self.many.len()))?;
        for (device, bytes) in self.iter() {
            map.serialize_entry(device, bytes)?;
        }
        map.end()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidencySummary {
    pub(crate) host_bytes: u64,
    pub(crate) provider_bytes: ProviderByteSummary,
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
    if let [Value::GpuTensor(handle)] = values {
        let mut summary = ResidencySummary::default();
        let bytes = gpu_handle_bytes(handle).unwrap_or_else(|| {
            summary.unknown_values = 1;
            0
        });
        summary.provider_bytes.insert(handle.device_id(), bytes);
        return summary;
    }
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
                summary.provider_bytes.add(handle.device_id(), bytes);
            }
        }
        Value::Cell(cell) => {
            for element in cell.data() {
                summarize_value(element, summary, visited_gpu, visited_handles);
            }
        }
        Value::Struct(value) => {
            for element in value.fields.values() {
                summarize_value(element, summary, visited_gpu, visited_handles);
            }
        }
        Value::StructArray(array) => {
            array.for_each_value(|element| {
                summarize_value(element, summary, visited_gpu, visited_handles)
            });
        }
        Value::Object(value) => {
            for element in value.properties().values() {
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
        | Value::BoundFunctionHandle(_)
        | Value::ScopedFunctionHandle(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
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
    let elements = handle.shape().iter().try_fold(1_u64, |total, dimension| {
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
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };
    use runmat_value::{CellArray, StructArray, Tensor};

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
    fn inline_provider_bytes_keep_sorted_map_serialization() {
        let mut summary = ProviderByteSummary::default();
        let mut original = BTreeMap::new();
        for (device, bytes) in [(9, 40), (2, 16)] {
            summary.insert(device, bytes);
            original.insert(device, bytes);
            assert_eq!(
                serde_json::to_vec(&summary).unwrap(),
                serde_json::to_vec(&original).unwrap(),
            );
        }
        assert_eq!(summary.get(&2), Some(&16));
    }

    #[test]
    fn nested_values_are_counted_and_aliased_gpu_handles_are_deduplicated() {
        let host = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let gpu = allocation.handle;
        let device = gpu.device_id();
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
        assert_eq!(summary.provider_bytes.get(&device), Some(&16));
        assert_eq!(summary.required_upload_bytes(device), 16);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(cell);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn nd_structure_array_residency_visits_field_columns_and_empty_schema() {
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let gpu = allocation.handle;
        let device = gpu.device_id();
        let element = |host: Value, resident: Value| {
            let mut value = runmat_value::UnboundedStructValueBuilder::with_capacity(2);
            value.insert("host", host);
            value.insert("resident", resident);
            value.finish()
        };
        let array = StructArray::with_fields(
            vec!["host".into(), "resident".into()],
            vec![
                element(Value::Num(1.0), Value::GpuTensor(gpu.clone())),
                element(
                    Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap()),
                    Value::GpuTensor(gpu),
                ),
            ],
            vec![1, 1, 2],
        )
        .expect("N-D structure array");
        let value = Value::StructArray(array);
        let empty = Value::StructArray(
            StructArray::with_fields(
                vec!["host".into(), "resident".into()],
                Vec::new(),
                vec![0, 2],
            )
            .expect("empty structure array"),
        );

        let summary = summarize_values(&[&value, &empty]);
        assert_eq!(summary.host_bytes, 24);
        assert_eq!(summary.provider_bytes.get(&device), Some(&16));
        assert_eq!(summary.required_upload_bytes(device), 24);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(value);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
