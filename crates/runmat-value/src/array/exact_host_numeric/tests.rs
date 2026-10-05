use super::*;
use crate::Tensor;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct Permit {
    limit: usize,
    retained: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct Reservation {
    retained: Arc<AtomicUsize>,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.retained.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

impl ExactHostAllocationReservation for Reservation {}

impl ExactHostAllocationPermit for Permit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.retained
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        Ok(Arc::new(Reservation {
            retained: self.retained.clone(),
            bytes,
        }))
    }
}

fn permit(limit: usize) -> Permit {
    Permit {
        limit,
        retained: Arc::new(AtomicUsize::new(0)),
    }
}

#[test]
fn exact_float32_allocation_moves_into_tensor_without_payload_copy() {
    let permit = permit(12);
    let mut owned = ExactHostNumericAllocation::new(NumericDType::F32, 3, &permit).unwrap();
    assert_eq!(permit.retained.load(Ordering::SeqCst), 12);
    assert_eq!(owned.capacity_bytes(), 12);
    owned
        .as_f32_slice_mut()
        .unwrap()
        .copy_from_slice(&[0.25, -0.5, 1.5]);
    assert!(owned.as_f64_slice_mut().is_none());
    let pointer = match &owned.allocation {
        ExactAllocation::Floating(allocation) => allocation.pointer().as_ptr(),
        ExactAllocation::Integer(_) => unreachable!("F32 allocation is floating"),
    };
    let tensor = Tensor::from_host_buffer(owned.into_host_buffer(), vec![3, 1]).unwrap();
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    assert_eq!(tensor.as_f32_slice(), Some([0.25, -0.5, 1.5].as_slice()));
    assert_eq!(
        tensor.as_f32_slice().unwrap().as_ptr().cast::<u8>(),
        pointer
    );
    drop(tensor);
    assert_eq!(permit.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn all_eight_integer_classes_move_exact_extrema_into_the_returned_tensor() {
    macro_rules! check {
        ($dtype:ident, $type:ty, $view:ident, $variant:ident, $low:expr, $high:expr) => {{
            let bytes = 2 * std::mem::size_of::<$type>();
            let quota = permit(bytes);
            let mut allocation = ExactHostNumericAllocation::new(NumericDType::$dtype, 2, &quota)
                .expect("exact integer allocation");
            assert_eq!(allocation.capacity_bytes(), bytes);
            let slice = allocation.$view().expect("matching typed view");
            slice.copy_from_slice(&[$low, $high]);
            let pointer = slice.as_ptr();
            let tensor = Tensor::from_host_buffer(allocation.into_host_buffer(), vec![1, 2])
                .expect("exact shape");
            assert_eq!(tensor.numeric_dtype(), NumericDType::$dtype);
            assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::$variant(vec![$low, $high]))
            );
            assert_eq!(
                unsafe { tensor.host_buffer().foreign_data_pointer() }.cast::<$type>(),
                pointer.cast_mut(),
                "integer block moved without a payload copy"
            );
            assert_eq!(quota.retained.load(Ordering::SeqCst), bytes);
            drop(tensor);
            assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
        }};
    }
    check!(I8, i8, as_i8_slice_mut, I8, i8::MIN, i8::MAX);
    check!(I16, i16, as_i16_slice_mut, I16, i16::MIN, i16::MAX);
    check!(I32, i32, as_i32_slice_mut, I32, i32::MIN, i32::MAX);
    check!(I64, i64, as_i64_slice_mut, I64, i64::MIN, i64::MAX);
    check!(U8, u8, as_u8_slice_mut, U8, u8::MIN, u8::MAX);
    check!(U16, u16, as_u16_slice_mut, U16, u16::MIN, u16::MAX);
    check!(U32, u32, as_u32_slice_mut, U32, u32::MIN, u32::MAX);
    check!(
        U64,
        u64,
        as_u64_slice_mut,
        U64,
        9_007_199_254_740_993,
        u64::MAX
    );
}

#[test]
fn integer_permit_rejection_overflow_and_shape_failure_release_exact_claim() {
    let quota = permit(1);
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::U16, 1, &quota),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::U64, usize::MAX, &quota),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    ));
    let quota = permit(8);
    let owned = ExactHostNumericAllocation::new(NumericDType::I64, 1, &quota).unwrap();
    assert!(Tensor::from_host_buffer(owned.into_host_buffer(), vec![2, 1]).is_err());
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn integer_copy_on_write_keeps_only_the_original_block_charged() {
    let quota = permit(16);
    let mut owned = ExactHostNumericAllocation::new(NumericDType::U64, 2, &quota).unwrap();
    owned
        .as_u64_slice_mut()
        .unwrap()
        .copy_from_slice(&[9_007_199_254_740_993, u64::MAX]);
    let original = Tensor::from_host_buffer(owned.into_host_buffer(), vec![1, 2]).unwrap();
    let mut clone = original.clone();
    assert_eq!(quota.retained.load(Ordering::SeqCst), 16);
    clone.set2(0, 0, 7.0).unwrap();
    assert_eq!(
        original.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
    assert_eq!(
        clone.integer_storage(),
        Some(&IntegerStorage::U64(vec![7, u64::MAX]))
    );
    drop(clone);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 16);
    drop(original);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn sole_owner_integer_mutation_keeps_block_and_materialization_releases_claim() {
    let quota = permit(16);
    let mut owned = ExactHostNumericAllocation::new(NumericDType::I64, 2, &quota).unwrap();
    owned
        .as_i64_slice_mut()
        .unwrap()
        .copy_from_slice(&[i64::MIN, i64::MAX]);
    let mut buffer = owned.into_host_buffer();
    let pointer = unsafe { buffer.foreign_data_pointer() };
    buffer
        .set_value(0, crate::NumericScalar::I64(-9_007_199_254_740_993))
        .unwrap();
    assert_eq!(unsafe { buffer.foreign_data_pointer_mut() }, pointer);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 16);
    let detached = buffer.into_numeric_storage();
    assert_eq!(
        detached,
        crate::NumericStorage::I64(vec![-9_007_199_254_740_993, i64::MAX])
    );
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_double_and_empty_allocations_have_exact_reported_capacity() {
    let double_permit = permit(16);
    let mut double = ExactHostNumericAllocation::new(NumericDType::F64, 2, &double_permit).unwrap();
    double
        .as_f64_slice_mut()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0]);
    let tensor = Tensor::from_host_buffer(double.into_host_buffer(), vec![1, 2]).unwrap();
    assert_eq!(tensor.as_f64_slice(), Some([1.0, 2.0].as_slice()));

    let empty = ExactHostNumericAllocation::new(NumericDType::F32, 0, &permit(0)).unwrap();
    assert_eq!(empty.capacity_bytes(), 0);
    let tensor = Tensor::from_host_buffer(empty.into_host_buffer(), vec![0, 1]).unwrap();
    assert!(tensor.is_empty());
}

#[test]
fn rejected_shape_and_unfinished_owner_release_reservation() {
    let quota = permit(8);
    let owned = ExactHostNumericAllocation::new(NumericDType::F64, 1, &quota).unwrap();
    assert_eq!(quota.retained.load(Ordering::SeqCst), 8);
    assert!(Tensor::from_host_buffer(owned.into_host_buffer(), vec![2, 1]).is_err());
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);

    let owned = ExactHostNumericAllocation::new(NumericDType::F64, 1, &quota).unwrap();
    drop(owned);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn shared_clone_retains_reservation_until_original_allocation_drops() {
    let quota = permit(8);
    let mut owned = ExactHostNumericAllocation::new(NumericDType::F64, 1, &quota).unwrap();
    owned.as_f64_slice_mut().unwrap()[0] = 3.0;
    let original = Tensor::from_host_buffer(owned.into_host_buffer(), vec![1, 1]).unwrap();
    let mut clone = original.clone();
    assert_eq!(quota.retained.load(Ordering::SeqCst), 8);
    clone.set2(0, 0, 4.0).unwrap();
    // Copy-on-write creates a new allocation outside the original call. Its
    // resource admission belongs to that later mutation, not this permit.
    assert_eq!(clone.as_f64_slice(), Some([4.0].as_slice()));
    assert_eq!(original.as_f64_slice(), Some([3.0].as_slice()));
    assert_eq!(quota.retained.load(Ordering::SeqCst), 8);
    drop(original);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_allocation_rejects_over_budget_and_overflow_without_leaking() {
    let permit = permit(11);
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::F32, 3, &permit),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::F64, usize::MAX, &permit),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    ));
    let integer = ExactHostNumericAllocation::new(NumericDType::I16, 2, &permit)
        .expect("I16 is now an exact retained class");
    assert_eq!(permit.retained.load(Ordering::SeqCst), 4);
    drop(integer);
    assert_eq!(permit.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn simultaneous_reservations_are_charged_until_each_owner_drops() {
    let permit = permit(16);
    let first = ExactHostNumericAllocation::new(NumericDType::F64, 1, &permit).unwrap();
    let second = ExactHostNumericAllocation::new(NumericDType::F32, 2, &permit).unwrap();
    assert_eq!(permit.retained.load(Ordering::SeqCst), 16);
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::F32, 1, &permit),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    drop(first);
    assert_eq!(permit.retained.load(Ordering::SeqCst), 8);
    drop(second);
    assert_eq!(permit.retained.load(Ordering::SeqCst), 0);
}
