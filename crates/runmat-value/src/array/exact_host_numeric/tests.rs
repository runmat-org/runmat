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
    let pointer = owned.allocation.pointer().as_ptr();
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
fn exact_allocation_rejects_over_budget_overflow_and_unadopted_integer_class() {
    let permit = permit(11);
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::F32, 3, &permit),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::F64, usize::MAX, &permit),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    ));
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::I16, 2, &permit),
        Err(ExactHostNumericAllocationError::UnsupportedClass)
    ));
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
