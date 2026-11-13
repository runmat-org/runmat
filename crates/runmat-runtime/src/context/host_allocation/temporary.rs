use num_complex::Complex64;
use std::alloc::{alloc, dealloc, Layout};
use std::fmt;
use std::sync::Arc;

use runmat_accelerate_api::HostNumericDataView;
use runmat_value::{
    exact_capacity_vec, ComplexElement, ExactHostAllocationClaim, ExactHostAllocationPermit,
    ExactHostAllocationReservation, ExactHostNumericAllocation, ExactHostNumericAllocationError,
    NumericDType, NumericScalar,
};

use super::receipt::AllocationFence;
use super::HostAllocationError;
use crate::context::host_effect::{
    HostEffectLiveAllocationClaim, HostEffectLiveAllocationKind, HostEffectStageReservation,
    ManagedHostEffectStageReservation,
};

pub(super) fn scratch_claim_bytes() -> usize {
    std::mem::size_of::<ScratchClaim>()
        + ManagedHostEffectStageReservation::exact_temporary_claim_metadata_bytes()
}

struct ScratchClaim {
    _live: HostEffectLiveAllocationClaim,
}

impl fmt::Debug for ScratchClaim {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScratchClaim")
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationReservation for ScratchClaim {
    fn retire_owned(self: Box<Self>) {
        let pointer = Box::into_raw(self);
        // SAFETY: consuming the unique Box makes the retained live claim
        // uniquely owned until it is dropped below.
        let live = unsafe { std::ptr::read(std::ptr::addr_of!((*pointer)._live)) };
        // SAFETY: `pointer` came from a Box<ScratchClaim> with this layout.
        unsafe { dealloc(pointer.cast::<u8>(), Layout::new::<Self>()) };
        drop(live);
    }
}

pub(super) struct ScratchPermit {
    stage: Arc<ManagedHostEffectStageReservation>,
    fence: AllocationFence,
}

impl ScratchPermit {
    pub(super) fn new(
        stage: Arc<ManagedHostEffectStageReservation>,
        fence: AllocationFence,
    ) -> Self {
        Self { stage, fence }
    }
}

impl fmt::Debug for ScratchPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScratchPermit")
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationPermit for ScratchPermit {
    fn reserve_exact(
        &self,
        _bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        // This private permit is consumed by ExactHostNumericAllocation, whose
        // owned-claim path avoids a second shared allocation owner.
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    }

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let charged = bytes
            .checked_add(std::mem::size_of::<ScratchClaim>())
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        let live = self
            .stage
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::TemporaryResult, charged)
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let layout = Layout::new::<ScratchClaim>();
        // SAFETY: this exact-layout block is initialized once and transferred
        // immediately to Box; on failure the live claim is dropped/refunded.
        let pointer = unsafe { alloc(layout) }.cast::<ScratchClaim>();
        if pointer.is_null() {
            return Err(ExactHostNumericAllocationError::AllocationFailed);
        }
        unsafe {
            pointer.write(ScratchClaim { _live: live });
            Ok(ExactHostAllocationClaim::owned(Box::from_raw(pointer)))
        }
    }
}

/// Aligned numeric workspace whose physical bytes remain in the stage's
/// scratch envelope until the workspace itself drops. No output conversion is
/// exposed, so retained Values must use the receipt's output permit instead.
pub struct HostNumericWorkspace {
    allocation: ExactHostNumericAllocation,
    dtype: NumericDType,
}

/// Safe complex64 view of one exactly claimed interleaved F64 workspace.
/// No alternate complex vector or byte reinterpretation is created.
pub struct HostComplexWorkspace {
    numeric: HostNumericWorkspace,
    len: usize,
}

mod typed_element {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for usize {}
    impl Sealed for (usize, usize) {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for runmat_value::ComplexElement<f32> {}
    impl Sealed for runmat_value::ComplexElement<f64> {}
}

/// A scratch vector whose exact physical backing and live stage claim remain
/// one owner. The sealed element set excludes `Value` and other language
/// payloads; this owner cannot be converted into a published result.
pub trait HostTypedElement: typed_element::Sealed + Copy + Default {}
impl<T: typed_element::Sealed + Copy + Default> HostTypedElement for T {}

pub struct HostTypedWorkspace<T: HostTypedElement> {
    values: Vec<T>,
    _claim: HostEffectLiveAllocationClaim,
}

impl<T: HostTypedElement> HostTypedWorkspace<T> {
    /// Exact backing and claim metadata to include in a stage peak plan.
    pub fn quoted_bytes(len: usize) -> Result<u64, HostAllocationError> {
        let layout = Layout::array::<T>(len).map_err(|_| HostAllocationError::Overflow)?;
        let bytes = layout
            .size()
            .checked_add(ManagedHostEffectStageReservation::exact_temporary_claim_metadata_bytes())
            .ok_or(HostAllocationError::Overflow)?;
        u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)
    }

    pub(super) fn new(
        len: usize,
        stage: Arc<ManagedHostEffectStageReservation>,
        fence: AllocationFence,
    ) -> Result<Self, HostAllocationError> {
        fence.check()?;
        let layout = Layout::array::<T>(len).map_err(|_| HostAllocationError::Overflow)?;
        let claim = stage.reserve_live_exact_claim(
            HostEffectLiveAllocationKind::TemporaryResult,
            layout.size(),
        )?;
        fence.check()?;
        let mut values = exact_capacity_vec(len).map_err(HostAllocationError::Numeric)?;
        values.resize(len, T::default());
        fence.check()?;
        Ok(Self {
            values,
            _claim: claim,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }
}

impl HostComplexWorkspace {
    pub(super) fn new(
        len: usize,
        stage: Arc<ManagedHostEffectStageReservation>,
        fence: AllocationFence,
    ) -> Result<Self, HostAllocationError> {
        let lanes = len.checked_mul(2).ok_or(HostAllocationError::Overflow)?;
        Ok(Self {
            numeric: HostNumericWorkspace::new(NumericDType::F64, lanes, stage, fence)?,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, index: usize) -> Option<Complex64> {
        let lanes = self.numeric.as_f64_slice()?;
        let start = index.checked_mul(2)?;
        Some(Complex64::new(*lanes.get(start)?, *lanes.get(start + 1)?))
    }

    pub fn set(&mut self, index: usize, value: Complex64) -> Option<()> {
        let lanes = self.numeric.as_f64_slice_mut()?;
        let start = index.checked_mul(2)?;
        let end = start.checked_add(2)?;
        let pair = lanes.get_mut(start..end)?;
        pair[0] = value.re;
        pair[1] = value.im;
        Some(())
    }
}

impl HostNumericWorkspace {
    pub(super) fn new(
        dtype: NumericDType,
        len: usize,
        stage: Arc<ManagedHostEffectStageReservation>,
        fence: AllocationFence,
    ) -> Result<Self, HostAllocationError> {
        let permit = ScratchPermit { stage, fence };
        let allocation = ExactHostNumericAllocation::new(dtype, len, &permit)?;
        Ok(Self { allocation, dtype })
    }

    pub fn set_value(&mut self, index: usize, value: NumericScalar) -> Result<(), String> {
        self.allocation.set_value(index, value)
    }

    pub fn data_view(&self) -> HostNumericDataView<'_> {
        match self.dtype {
            NumericDType::F64 => HostNumericDataView::F64(self.allocation.as_f64_slice().unwrap()),
            NumericDType::F32 => HostNumericDataView::F32(self.allocation.as_f32_slice().unwrap()),
            NumericDType::I8 => HostNumericDataView::I8(self.allocation.as_i8_slice().unwrap()),
            NumericDType::I16 => HostNumericDataView::I16(self.allocation.as_i16_slice().unwrap()),
            NumericDType::I32 => HostNumericDataView::I32(self.allocation.as_i32_slice().unwrap()),
            NumericDType::I64 => HostNumericDataView::I64(self.allocation.as_i64_slice().unwrap()),
            NumericDType::U8 => HostNumericDataView::U8(self.allocation.as_u8_slice().unwrap()),
            NumericDType::U16 => HostNumericDataView::U16(self.allocation.as_u16_slice().unwrap()),
            NumericDType::U32 => HostNumericDataView::U32(self.allocation.as_u32_slice().unwrap()),
            NumericDType::U64 => HostNumericDataView::U64(self.allocation.as_u64_slice().unwrap()),
        }
    }

    pub fn as_f64_slice_mut(&mut self) -> Option<&mut [f64]> {
        self.allocation.as_f64_slice_mut()
    }

    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        self.allocation.as_f64_slice()
    }

    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        self.allocation.as_f32_slice_mut()
    }

    /// View one exactly admitted F32 scratch buffer as interleaved complex
    /// values. The owner retains the original backing and claim throughout
    /// the returned borrow; no second vector or payload allocation is made.
    pub fn as_complex_f32_slice_mut(&mut self) -> Option<&mut [ComplexElement<f32>]> {
        complex_f32_view(self.allocation.as_f32_slice_mut()?)
    }

    /// F64 counterpart of `as_complex_f32_slice_mut`.
    pub fn as_complex_f64_slice_mut(&mut self) -> Option<&mut [ComplexElement<f64>]> {
        complex_f64_view(self.allocation.as_f64_slice_mut()?)
    }

    pub fn as_u64_slice_mut(&mut self) -> Option<&mut [u64]> {
        self.allocation.as_u64_slice_mut()
    }

    pub fn as_u64_slice(&self) -> Option<&[u64]> {
        self.allocation.as_u64_slice()
    }
}

macro_rules! complex_workspace_view {
    ($name:ident, $lane:ty) => {
        fn $name(lanes: &mut [$lane]) -> Option<&mut [ComplexElement<$lane>]> {
            if lanes.len() % 2 != 0 {
                return None;
            }
            // The exact owner allocated initialized, aligned native lanes;
            // repr(C) ComplexElement has precisely two of these lanes.
            assert_eq!(
                std::mem::size_of::<ComplexElement<$lane>>(),
                2 * std::mem::size_of::<$lane>()
            );
            assert_eq!(
                std::mem::align_of::<ComplexElement<$lane>>(),
                std::mem::align_of::<$lane>()
            );
            Some(unsafe {
                std::slice::from_raw_parts_mut(lanes.as_mut_ptr().cast(), lanes.len() / 2)
            })
        }
    };
}

complex_workspace_view!(complex_f32_view, f32);
complex_workspace_view!(complex_f64_view, f64);
