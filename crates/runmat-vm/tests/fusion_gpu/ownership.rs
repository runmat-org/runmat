//! The deterministic provider's payload map is also its lease authority.
use super::TestProvider;
use runmat_accelerate_api::allocation::{
    AllocationAcquisitionError, AllocationIdentity, AllocationOwner, AllocationRegistrationError,
    AllocationRetirementOutcome, WeakAllocationLease,
};
use runmat_accelerate_api::{
    AccelProvider, GpuPhysicalDescriptor, GpuTensorHandle, GpuTensorStorage, GpuTensorView,
    NumericElementType,
};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

pub(super) fn element_for(
    class: runmat_accelerate_api::ProviderElementType,
) -> anyhow::Result<NumericElementType> {
    match class {
        runmat_accelerate_api::ProviderElementType::F32 => Ok(NumericElementType::F32),
        runmat_accelerate_api::ProviderElementType::F64 => Ok(NumericElementType::F64),
        other => anyhow::bail!("test fusion provider does not support {other:?} output"),
    }
}

pub(super) type BufferStore = HashMap<u64, BufferRecord>;

#[derive(Clone)]
pub(super) enum BufferPayload {
    F32(Vec<f32>),
    F64(Vec<f64>),
    ComplexF32(Vec<(f32, f32)>),
    ComplexF64(Vec<(f64, f64)>),
    U8(Vec<u8>),
}

impl BufferPayload {
    // Fusion expressions are interpreted in f64 by this deterministic fixture;
    // conversion here tests native output storage/class, not f32 shader arithmetic.
    pub fn from_values(data: Vec<f64>, element: NumericElementType) -> Self {
        match element {
            NumericElementType::F32 => {
                Self::F32(data.into_iter().map(|value| value as f32).collect())
            }
            NumericElementType::F64 => Self::F64(data),
            NumericElementType::U8 => Self::U8(data.into_iter().map(|value| value as u8).collect()),
            _ => panic!("fixture only supports floating-point and logical-byte payloads"),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::F64(data) => data.len(),
            Self::ComplexF32(data) => data.len(),
            Self::ComplexF64(data) => data.len(),
            Self::U8(data) => data.len(),
        }
    }
    pub fn to_f64(&self) -> Option<Vec<f64>> {
        match self {
            Self::F32(data) => Some(data.iter().copied().map(f64::from).collect()),
            Self::F64(data) => Some(data.clone()),
            Self::ComplexF32(_) | Self::ComplexF64(_) => None,
            Self::U8(data) => Some(data.iter().copied().map(f64::from).collect()),
        }
    }
    pub fn write(&mut self, index: usize, value: f64) -> Result<(), &'static str> {
        match self {
            Self::F32(data) => data[index] = value as f32,
            Self::F64(data) => data[index] = value,
            Self::ComplexF32(_) | Self::ComplexF64(_) => {
                return Err("real scalar write cannot change complex fixture storage");
            }
            Self::U8(data) => data[index] = value as u8,
        }
        Ok(())
    }
    pub fn exact_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => {
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) && a.len() == b.len()
            }
            (Self::F64(a), Self::F64(b)) => {
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) && a.len() == b.len()
            }
            (Self::ComplexF32(a), Self::ComplexF32(b)) => {
                a.iter()
                    .zip(b)
                    .all(|(x, y)| x.0.to_bits() == y.0.to_bits() && x.1.to_bits() == y.1.to_bits())
                    && a.len() == b.len()
            }
            (Self::ComplexF64(a), Self::ComplexF64(b)) => {
                a.iter()
                    .zip(b)
                    .all(|(x, y)| x.0.to_bits() == y.0.to_bits() && x.1.to_bits() == y.1.to_bits())
                    && a.len() == b.len()
            }
            (Self::U8(a), Self::U8(b)) => a == b,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub(super) struct BufferRecord {
    pub data: BufferPayload,
    pub shape: Vec<usize>,
    physical: GpuPhysicalDescriptor,
    logical: bool,
    lease: Option<WeakAllocationLease>,
    phase: Phase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Pending,
    Live,
    RetainedFailure,
    Uncertain,
}

impl BufferRecord {
    pub fn new(
        data: BufferPayload,
        shape: Vec<usize>,
        physical: GpuPhysicalDescriptor,
        logical: bool,
    ) -> Self {
        assert_eq!(
            shape.iter().try_fold(1usize, |n, &dim| n.checked_mul(dim)),
            Some(data.len())
        );
        Self {
            data,
            shape,
            physical,
            logical,
            lease: None,
            phase: Phase::Pending,
        }
    }

    pub fn is_live(&self) -> bool {
        self.phase == Phase::Live
    }

    pub fn owns(&self, handle: &GpuTensorHandle) -> bool {
        self.is_live()
            && self
                .lease
                .as_ref()
                .and_then(WeakAllocationLease::acquire)
                .is_some_and(|lease| lease.shares_allocation_with(handle.allocation_lease()))
    }

    pub fn exact_snapshot_eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.physical == other.physical
            && self.logical == other.logical
            && self.phase == other.phase
            && self.data.exact_eq(&other.data)
    }
}

pub(super) fn physical_for(len: usize, element: NumericElementType) -> GpuPhysicalDescriptor {
    let bytes = len
        .checked_mul(element.element_size())
        .expect("fixture byte length");
    GpuPhysicalDescriptor::new(
        element,
        GpuTensorStorage::Real,
        len as u64,
        bytes as u64,
        bytes as u64,
    )
    .expect("fixture physical descriptor")
}

pub(super) fn complex_physical_for(
    len: usize,
    element: NumericElementType,
) -> GpuPhysicalDescriptor {
    let bytes = len
        .checked_mul(element.element_size())
        .and_then(|bytes| bytes.checked_mul(2))
        .expect("fixture complex byte length");
    GpuPhysicalDescriptor::new(
        element,
        GpuTensorStorage::ComplexInterleaved,
        len as u64,
        bytes as u64,
        bytes as u64,
    )
    .expect("fixture complex physical descriptor")
}

impl AllocationOwner for TestProvider {
    fn register_lease(
        &self,
        identity: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<GpuPhysicalDescriptor, AllocationRegistrationError> {
        if identity.device_id() != 0 {
            return Err(AllocationRegistrationError::WrongOwner);
        }
        if identity.generation() != 1 {
            return Err(AllocationRegistrationError::StaleGeneration);
        }
        let mut buffers = self
            .buffers
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let record = buffers
            .get_mut(&identity.allocation_id())
            .ok_or(AllocationRegistrationError::UnknownAllocation)?;
        if record.lease.is_some() {
            return Err(AllocationRegistrationError::AlreadyRegistered);
        }
        if record.phase != Phase::Pending {
            return Err(AllocationRegistrationError::NotLive);
        }
        let physical = record.physical;
        record.lease = Some(lease);
        record.phase = Phase::Live;
        Ok(physical)
    }

    fn validate_live(
        &self,
        identity: AllocationIdentity,
    ) -> Result<(), AllocationAcquisitionError> {
        if identity.device_id() != 0 {
            return Err(AllocationAcquisitionError::WrongOwner);
        }
        if identity.generation() != 1 {
            return Err(AllocationAcquisitionError::StaleGeneration);
        }
        let buffers = self
            .buffers
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if buffers
            .get(&identity.allocation_id())
            .is_some_and(BufferRecord::is_live)
        {
            Ok(())
        } else {
            Err(AllocationAcquisitionError::NotLive)
        }
    }

    fn validate_view(
        &self,
        identity: AllocationIdentity,
        view: &GpuTensorView,
    ) -> Result<(), AllocationAcquisitionError> {
        self.validate_live(identity)?;
        if view.is_logical()
            != self
                .buffers
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .get(&identity.allocation_id())
                .is_some_and(|record| record.logical)
        {
            return Err(AllocationAcquisitionError::Unsupported);
        }
        Ok(())
    }

    fn retire(&self, identity: AllocationIdentity) -> AllocationRetirementOutcome {
        if identity.device_id() != 0 || identity.generation() != 1 {
            return AllocationRetirementOutcome::Uncertain {
                detail: "fixture owner identity mismatch".into(),
            };
        }
        let id = identity.allocation_id();
        let failure = self.fail_free_buffer.load(Ordering::Relaxed) == id;
        {
            let mut buffers = self
                .buffers
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let Some(record) = buffers.get_mut(&id) else {
                return AllocationRetirementOutcome::Uncertain {
                    detail: "fixture allocation missing at retirement".into(),
                };
            };
            if record.phase != Phase::Live {
                return AllocationRetirementOutcome::Uncertain {
                    detail: "fixture allocation not live at retirement".into(),
                };
            }
            record.phase = if failure {
                Phase::RetainedFailure
            } else {
                Phase::Uncertain
            };
        }
        // Exercise the GC lock-order witness after dropping the payload-map lock.
        if self.inspect_roots_on_free.load(Ordering::Relaxed) {
            assert!(
                runmat_gc::gc_visit_retained_values(&[], &mut |_| Some(())).is_some(),
                "provider release must run outside root registration locks"
            );
        }
        if failure {
            return AllocationRetirementOutcome::RetainedFailure {
                detail: "injected release failure for allocated buffer".into(),
            };
        }
        *self
            .frees
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(id)
            .or_default() += 1;
        self.buffers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&id);
        AllocationRetirementOutcome::Retired
    }

    fn record_retirement_panic(&self, identity: AllocationIdentity) {
        let mut buffers = self
            .buffers
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(record) = buffers.get_mut(&identity.allocation_id()) {
            record.phase = Phase::Uncertain;
        }
    }
}

#[test]
fn typed_payload_reshape_retains_one_physical_allocation_until_final_view() {
    runmat_gc::gc_test_context(|| {
        let _lock = super::accel_test_lock();
        super::ensure_provider_registered();
        let provider = super::registered_provider().unwrap();
        let original = provider.push_typed(vec![1.25, 2.5], vec![1, 2], NumericElementType::F32);
        let id = original.buffer_id();
        let reshaped = provider.reshape(&original, &[2, 1]).unwrap();
        assert!(original.shares_allocation_with(&reshaped));
        assert_eq!(original.physical().element_type(), NumericElementType::F32);
        assert_eq!(
            provider.pull(&original).unwrap(),
            (vec![1.25, 2.5], vec![1, 2])
        );
        assert_eq!(
            provider.pull(&reshaped).unwrap(),
            (vec![1.25, 2.5], vec![2, 1])
        );
        drop(reshaped);
        assert_eq!(provider.frees.lock().unwrap().get(&id), None);
        assert_eq!(
            original.release().unwrap(),
            runmat_accelerate_api::allocation::AllocationReleaseOutcome::Retired
        );
        assert_eq!(provider.frees.lock().unwrap().get(&id), Some(&1));
        assert!(!provider.buffers.lock().unwrap().contains_key(&id));
    });
}
