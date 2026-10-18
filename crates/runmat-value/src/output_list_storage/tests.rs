use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::*;
use crate::ExactHostAllocationReservation;

#[derive(Debug)]
struct Permit {
    limit: usize,
    live: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct Claim {
    bytes: usize,
    live: Arc<AtomicUsize>,
}

impl ExactHostAllocationReservation for Claim {}

impl Drop for Claim {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

impl ExactHostAllocationPermit for Permit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.live
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.limit)
            })
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        Ok(Arc::new(Claim {
            bytes,
            live: self.live.clone(),
        }))
    }
}

#[test]
fn shared_output_backing_and_consumption_preserve_one_physical_claim() {
    let bytes = OutputListStorage::requested_bytes(3).unwrap();
    let permit = Permit {
        limit: bytes,
        live: Arc::new(AtomicUsize::new(0)),
    };
    let list = OutputListStorage::try_initialize(3, &permit, |index| {
        Ok::<_, ()>(Value::Num(index as f64))
    })
    .unwrap();
    assert!(list.has_retained_claim());
    assert_eq!(list.payload.as_ref().unwrap().get().values.capacity(), 3);
    let pointer = list.as_slice().as_ptr();
    let alias = list.clone();
    assert_eq!(pointer, alias.as_slice().as_ptr());
    assert_eq!(permit.live.load(Ordering::SeqCst), bytes);
    let list = list.try_into_owned().unwrap_err();
    assert_eq!(pointer, list.as_slice().as_ptr());
    drop(alias);
    let owned = list.try_into_owned().unwrap();
    assert_eq!(pointer, owned.as_slice().as_ptr());
    let mut values = owned.into_iter();
    assert_eq!(values.next(), Some(Value::Num(0.0)));
    assert_eq!(values.next_back(), Some(Value::Num(2.0)));
    assert_eq!(values.len(), 1);
    assert_eq!(permit.live.load(Ordering::SeqCst), bytes);
    drop(values);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
}

#[test]
fn initialization_refuses_before_elements_and_refunds_failed_construction() {
    let bytes = OutputListStorage::requested_bytes(2).unwrap();
    let permit = Permit {
        limit: bytes - 1,
        live: Arc::new(AtomicUsize::new(0)),
    };
    let mut calls = 0;
    let denied = OutputListStorage::try_initialize(2, &permit, |_| {
        calls += 1;
        Ok::<_, ()>(Value::Num(1.0))
    });
    assert!(matches!(
        denied,
        Err(OutputListInitializationError::Allocation(
            ExactHostNumericAllocationError::ExceedsPermit
        ))
    ));
    assert_eq!(calls, 0);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
    let permit = Permit {
        limit: bytes,
        live: permit.live,
    };
    let failed = OutputListStorage::try_initialize(2, &permit, |index| {
        if index == 0 {
            Ok(Value::Num(1.0))
        } else {
            Err(17)
        }
    });
    assert!(matches!(
        failed,
        Err(OutputListInitializationError::Element(17))
    ));
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
    let empty = OutputListStorage::try_initialize(0, &permit, |_| Err::<Value, _>(()))
        .expect("empty output does not initialize elements");
    assert!(empty.is_empty());
    assert_eq!(empty, OutputListStorage::unbounded(Vec::new()));
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
}
