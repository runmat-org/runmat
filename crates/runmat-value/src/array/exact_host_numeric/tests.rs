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

fn floating_charge(payload_and_shape: usize) -> usize {
    payload_and_shape + std::mem::size_of::<RustExactRelease>()
}

#[test]
fn exact_two_dimensional_owner_charges_shape_before_payload_and_retains_both() {
    let shape_bytes = 2 * std::mem::size_of::<usize>();
    let denied = permit(16 + shape_bytes - 1);
    assert!(matches!(
        ExactHostTensor2dAllocation::new(NumericDType::U64, 1, 2, &denied),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(denied.retained.load(Ordering::SeqCst), 0);

    let quota = permit(16 + shape_bytes);
    let mut allocation = ExactHostTensor2dAllocation::new(NumericDType::U64, 1, 2, &quota).unwrap();
    allocation
        .as_u64_slice_mut()
        .unwrap()
        .copy_from_slice(&[9_007_199_254_740_993, u64::MAX]);
    let original = allocation.as_u64_slice_mut().unwrap().as_ptr();
    assert_eq!(quota.retained.load(Ordering::SeqCst), 16 + shape_bytes);
    let tensor = allocation.into_tensor().expect("exact geometry");
    assert_eq!(tensor.shape, [1, 2]);
    assert_eq!(tensor.shape.capacity(), 2);
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
    assert_eq!(
        unsafe { tensor.host_buffer().foreign_data_pointer() }.cast::<u64>(),
        original.cast_mut()
    );
    assert_eq!(quota.retained.load(Ordering::SeqCst), 16 + shape_bytes);
    drop(tensor);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_two_dimensional_shape_rejects_count_overflow_without_a_claim() {
    let quota = permit(usize::MAX);
    assert!(matches!(
        ExactHostTensor2dAllocation::new(NumericDType::F32, usize::MAX, 2, &quota),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    ));
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_two_dimensional_float_retains_single_storage_and_shape_charge() {
    let total = floating_charge(3 * std::mem::size_of::<f32>() + 2 * std::mem::size_of::<usize>());
    let quota = permit(total);
    let mut allocation = ExactHostTensor2dAllocation::new(NumericDType::F32, 3, 1, &quota).unwrap();
    allocation
        .as_f32_slice_mut()
        .unwrap()
        .copy_from_slice(&[0.25, -0.5, 1.5]);
    let original = allocation.as_f32_slice_mut().unwrap().as_ptr();
    let tensor = allocation.into_tensor().expect("exact shape");
    assert_eq!(tensor.shape.as_slice(), [3, 1]);
    assert_eq!(tensor.as_f32_slice(), Some([0.25, -0.5, 1.5].as_slice()));
    assert_eq!(tensor.as_f32_slice().unwrap().as_ptr(), original);
    assert_eq!(quota.retained.load(Ordering::SeqCst), total);
    drop(tensor);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_float_release_owner_is_precharged_before_payload_allocation() {
    let payload_and_shape = 8 + 2 * std::mem::size_of::<usize>();
    let denied = permit(floating_charge(payload_and_shape) - 1);
    assert!(matches!(
        ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 1, &denied),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(denied.retained.load(Ordering::SeqCst), 0);
    let admitted = permit(floating_charge(payload_and_shape));
    let output = ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 1, &admitted)
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_eq!(
        admitted.retained.load(Ordering::SeqCst),
        floating_charge(payload_and_shape)
    );
    drop(output);
    assert_eq!(admitted.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_parser_store_preserves_float_and_wide_integer_bits_without_reallocation() {
    let float_quota = permit(floating_charge(16 + 2 * std::mem::size_of::<usize>()));
    let mut floats = ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 2, &float_quota)
        .expect("exact floating owner");
    let float_pointer = floats.as_f64_slice_mut().unwrap().as_ptr();
    floats
        .set_value(0, crate::NumericScalar::F64(-0.0))
        .unwrap();
    floats
        .set_value(1, crate::NumericScalar::F64(f64::INFINITY))
        .unwrap();
    assert!(floats.set_value(2, crate::NumericScalar::F64(1.0)).is_err());
    let tensor = floats.into_tensor().unwrap();
    assert_eq!(tensor.as_f64_slice().unwrap().as_ptr(), float_pointer);
    assert_eq!(
        tensor.as_f64_slice().unwrap()[0].to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(tensor.as_f64_slice().unwrap()[1], f64::INFINITY);
    drop(tensor);
    assert_eq!(float_quota.retained.load(Ordering::SeqCst), 0);

    let single_quota = permit(floating_charge(4 + 2 * std::mem::size_of::<usize>()));
    let mut singles = ExactHostTensor2dAllocation::new(NumericDType::F32, 1, 1, &single_quota)
        .expect("exact single owner");
    let single_pointer = singles.as_f32_slice_mut().unwrap().as_ptr();
    singles
        .set_value(0, crate::NumericScalar::F32(-0.25))
        .unwrap();
    let tensor = singles.into_tensor().unwrap();
    assert_eq!(tensor.as_f32_slice().unwrap().as_ptr(), single_pointer);
    assert_eq!(tensor.as_f32_slice().unwrap(), &[-0.25]);
    drop(tensor);
    assert_eq!(single_quota.retained.load(Ordering::SeqCst), 0);

    let integer_quota = permit(16 + 2 * std::mem::size_of::<usize>());
    let mut integers = ExactHostTensor2dAllocation::new(NumericDType::U64, 1, 2, &integer_quota)
        .expect("exact integer owner");
    let integer_pointer = integers.as_u64_slice_mut().unwrap().as_ptr();
    integers
        .set_value(0, crate::NumericScalar::U64(9_007_199_254_740_993))
        .unwrap();
    integers
        .set_value(1, crate::NumericScalar::U64(u64::MAX))
        .unwrap();
    assert!(integers
        .set_value(0, crate::NumericScalar::F64(1.0))
        .is_err());
    assert!(integers
        .set_value(1, crate::NumericScalar::I64(-1))
        .is_err());
    let tensor = integers.into_tensor().unwrap();
    assert_eq!(
        unsafe { tensor.host_buffer().foreign_data_pointer() }.cast::<u64>(),
        integer_pointer.cast_mut()
    );
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
    drop(tensor);
    assert_eq!(integer_quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_float32_allocation_moves_into_tensor_without_payload_copy() {
    let permit = permit(floating_charge(12));
    let mut owned = ExactHostNumericAllocation::new(NumericDType::F32, 3, &permit).unwrap();
    assert_eq!(permit.retained.load(Ordering::SeqCst), floating_charge(12));
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
    let double_permit = permit(floating_charge(16));
    let mut double = ExactHostNumericAllocation::new(NumericDType::F64, 2, &double_permit).unwrap();
    double
        .as_f64_slice_mut()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0]);
    let tensor = Tensor::from_host_buffer(double.into_host_buffer(), vec![1, 2]).unwrap();
    assert_eq!(tensor.as_f64_slice(), Some([1.0, 2.0].as_slice()));

    let empty =
        ExactHostNumericAllocation::new(NumericDType::F32, 0, &permit(floating_charge(0))).unwrap();
    assert_eq!(empty.capacity_bytes(), 0);
    let tensor = Tensor::from_host_buffer(empty.into_host_buffer(), vec![0, 1]).unwrap();
    assert!(tensor.is_empty());
}

#[test]
fn rejected_shape_and_unfinished_owner_release_reservation() {
    let quota = permit(floating_charge(8));
    let owned = ExactHostNumericAllocation::new(NumericDType::F64, 1, &quota).unwrap();
    assert_eq!(quota.retained.load(Ordering::SeqCst), floating_charge(8));
    assert!(Tensor::from_host_buffer(owned.into_host_buffer(), vec![2, 1]).is_err());
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);

    let owned = ExactHostNumericAllocation::new(NumericDType::F64, 1, &quota).unwrap();
    drop(owned);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn shared_clone_retains_reservation_until_original_allocation_drops() {
    let quota = permit(floating_charge(8));
    let mut owned = ExactHostNumericAllocation::new(NumericDType::F64, 1, &quota).unwrap();
    owned.as_f64_slice_mut().unwrap()[0] = 3.0;
    let original = Tensor::from_host_buffer(owned.into_host_buffer(), vec![1, 1]).unwrap();
    let mut clone = original.clone();
    assert_eq!(quota.retained.load(Ordering::SeqCst), floating_charge(8));
    clone.set2(0, 0, 4.0).unwrap();
    // Copy-on-write creates a new allocation outside the original call. Its
    // resource admission belongs to that later mutation, not this permit.
    assert_eq!(clone.as_f64_slice(), Some([4.0].as_slice()));
    assert_eq!(original.as_f64_slice(), Some([3.0].as_slice()));
    assert_eq!(quota.retained.load(Ordering::SeqCst), floating_charge(8));
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
    let permit = permit(floating_charge(8) * 2);
    let first = ExactHostNumericAllocation::new(NumericDType::F64, 1, &permit).unwrap();
    let second = ExactHostNumericAllocation::new(NumericDType::F32, 2, &permit).unwrap();
    assert_eq!(
        permit.retained.load(Ordering::SeqCst),
        floating_charge(8) * 2
    );
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::F32, 1, &permit),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    drop(first);
    assert_eq!(permit.retained.load(Ordering::SeqCst), floating_charge(8));
    drop(second);
    assert_eq!(permit.retained.load(Ordering::SeqCst), 0);
}
