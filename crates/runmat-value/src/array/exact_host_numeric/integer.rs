//! Exact, zero-initialized integer block with one retained quota claim.

use super::*;

pub(super) struct ExactIntegerStorage {
    pub(super) values: IntegerStorage,
    reservation: ExactHostAllocationClaim,
    capacity_bytes: usize,
}

impl ExactIntegerStorage {
    pub(super) fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub(super) fn into_parts(self) -> (IntegerStorage, ExactHostAllocationClaim) {
        (self.values, self.reservation)
    }
}

pub(super) fn integer_layout(
    dtype: NumericDType,
    len: usize,
) -> Result<Layout, std::alloc::LayoutError> {
    match dtype {
        NumericDType::I8 => Layout::array::<i8>(len),
        NumericDType::I16 => Layout::array::<i16>(len),
        NumericDType::I32 => Layout::array::<i32>(len),
        NumericDType::I64 => Layout::array::<i64>(len),
        NumericDType::U8 => Layout::array::<u8>(len),
        NumericDType::U16 => Layout::array::<u16>(len),
        NumericDType::U32 => Layout::array::<u32>(len),
        NumericDType::U64 => Layout::array::<u64>(len),
        NumericDType::F32 | NumericDType::F64 => {
            unreachable!("floating layout is handled by caller")
        }
    }
}

pub(super) fn allocate_integer(
    dtype: NumericDType,
    len: usize,
    layout: Layout,
    reservation: ExactHostAllocationClaim,
) -> Result<ExactIntegerStorage, ExactHostNumericAllocationError> {
    macro_rules! allocate {
        ($variant:ident, $type:ty) => {{
            let pointer = if layout.size() == 0 {
                NonNull::<$type>::dangling().as_ptr()
            } else {
                // SAFETY: the layout was built for exactly `len` elements of
                // this branch's type, with the global Rust allocator.
                let pointer = unsafe { alloc_zeroed(layout) }.cast::<$type>();
                if pointer.is_null() {
                    return Err(ExactHostNumericAllocationError::AllocationFailed);
                }
                pointer
            };
            // SAFETY: pointer and layout have the same element type/length,
            // are initialized to zero, and ownership transfers exactly once
            // to Vec, which uses the same global allocator.
            IntegerStorage::$variant(unsafe { Vec::from_raw_parts(pointer, len, len) })
        }};
    }
    let values = match dtype {
        NumericDType::I8 => allocate!(I8, i8),
        NumericDType::I16 => allocate!(I16, i16),
        NumericDType::I32 => allocate!(I32, i32),
        NumericDType::I64 => allocate!(I64, i64),
        NumericDType::U8 => allocate!(U8, u8),
        NumericDType::U16 => allocate!(U16, u16),
        NumericDType::U32 => allocate!(U32, u32),
        NumericDType::U64 => allocate!(U64, u64),
        NumericDType::F32 | NumericDType::F64 => {
            unreachable!("floating allocation is handled by caller")
        }
    };
    Ok(ExactIntegerStorage {
        values,
        reservation,
        capacity_bytes: layout.size(),
    })
}
