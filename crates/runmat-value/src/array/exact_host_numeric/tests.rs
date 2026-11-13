use super::*;
use crate::{ComplexStorage, Tensor};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
struct Permit {
    limit: usize,
    retained: Arc<AtomicUsize>,
    claims: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct Reservation {
    retained: Arc<AtomicUsize>,
    claims: Arc<AtomicUsize>,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.retained.fetch_sub(self.bytes, Ordering::SeqCst);
        self.claims.fetch_sub(1, Ordering::SeqCst);
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
        self.claims.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(Reservation {
            retained: self.retained.clone(),
            claims: self.claims.clone(),
            bytes,
        }))
    }
}

fn permit(limit: usize) -> Permit {
    Permit {
        limit,
        retained: Arc::new(AtomicUsize::new(0)),
        claims: Arc::new(AtomicUsize::new(0)),
    }
}

fn floating_charge(payload_and_shape: usize) -> usize {
    payload_and_shape
        + std::mem::size_of::<RustExactRelease>()
        + HostNumericBuffer::exact_metadata_bytes()
}

fn integer_charge(payload_and_shape: usize) -> usize {
    payload_and_shape + HostNumericBuffer::exact_metadata_bytes()
}

#[test]
fn exact_sparse_csc_keeps_original_claims_until_final_value_drop() {
    use std::num::NonZeroUsize;

    for len in [0, 3] {
        let bytes = ExactHostIndexBufferAllocation::requested_bytes(len).unwrap();
        let quota = permit(bytes);
        let mut output = ExactHostIndexBufferAllocation::new(len, &quota).unwrap();
        let pointer = output.as_mut_slice().as_ptr();
        for (index, value) in output.as_mut_slice().iter_mut().enumerate() {
            *value = index;
        }
        let output = output.into_buffer().unwrap();
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(output.len(), len);
        assert_eq!(quota.retained.load(Ordering::SeqCst), bytes);
        assert_eq!(quota.claims.load(Ordering::SeqCst), 1);
        let shared = output.clone();
        assert!(shared.shares_allocation_with(&output));
        drop(output);
        assert_eq!(quota.retained.load(Ordering::SeqCst), bytes);
        drop(shared);
        assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
        assert_eq!(quota.claims.load(Ordering::SeqCst), 0);
        let denied = permit(bytes - 1);
        assert!(matches!(
            ExactHostIndexBufferAllocation::new(len, &denied),
            Err(ExactHostNumericAllocationError::ExceedsPermit)
        ));
        assert_eq!(denied.retained.load(Ordering::SeqCst), 0);
    }
    assert_eq!(
        ExactHostIndexBufferAllocation::requested_bytes(usize::MAX),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    );

    for class in [
        ExactHostSparseClass::Logical,
        ExactHostSparseClass::Numeric(NumericDType::F32),
        ExactHostSparseClass::Numeric(NumericDType::F64),
        ExactHostSparseClass::Numeric(NumericDType::I8),
        ExactHostSparseClass::Numeric(NumericDType::I16),
        ExactHostSparseClass::Numeric(NumericDType::I32),
        ExactHostSparseClass::Numeric(NumericDType::I64),
        ExactHostSparseClass::Numeric(NumericDType::U8),
        ExactHostSparseClass::Numeric(NumericDType::U16),
        ExactHostSparseClass::Numeric(NumericDType::U32),
        ExactHostSparseClass::Numeric(NumericDType::U64),
        ExactHostSparseClass::Complex(NumericDType::F32),
        ExactHostSparseClass::Complex(NumericDType::F64),
    ] {
        let charged =
            ExactHostSparseTensorAllocation::requested_bytes_for_dimensions(class, 2, 1, 1)
                .expect("CSC quote");
        let quota = permit(charged);
        let mut allocation =
            ExactHostSparseTensorAllocation::new(class, 2, 1, 1, &quota).expect("admitted CSC");
        assert_eq!(quota.retained.load(Ordering::SeqCst), charged);
        assert_eq!(
            quota.claims.load(Ordering::SeqCst),
            class.retained_claim_count()
        );
        allocation.set_col_ptr(0, 0).unwrap();
        allocation.set_col_ptr(1, 1).unwrap();
        allocation.set_row_index(0, 1).unwrap();
        let expected_numeric = match class {
            ExactHostSparseClass::Logical => None,
            ExactHostSparseClass::Numeric(dtype) => {
                let scalar = match dtype {
                    NumericDType::F32 => NumericScalar::F32(1.25),
                    NumericDType::F64 => NumericScalar::F64(1.25),
                    NumericDType::I8 => NumericScalar::I8(i8::MIN),
                    NumericDType::I16 => NumericScalar::I16(i16::MIN),
                    NumericDType::I32 => NumericScalar::I32(i32::MIN),
                    NumericDType::I64 => NumericScalar::I64(i64::MIN),
                    NumericDType::U8 => NumericScalar::U8(u8::MAX),
                    NumericDType::U16 => NumericScalar::U16(u16::MAX),
                    NumericDType::U32 => NumericScalar::U32(u32::MAX),
                    NumericDType::U64 => NumericScalar::U64(u64::MAX),
                };
                allocation.set_numeric_value(0, scalar).unwrap();
                Some(scalar)
            }
            ExactHostSparseClass::Complex(_) => {
                allocation.set_complex_value(0, 2.0, -3.0).unwrap();
                None
            }
        };
        let mut validation = allocation.into_validation().expect("owned CSC validation");
        while !validation.advance(NonZeroUsize::new(1).unwrap()).unwrap() {}
        let sparse = validation.into_tensor().expect("validated CSC");
        assert_eq!((sparse.rows, sparse.cols), (2, 1));
        assert_eq!(&sparse.col_ptrs[..], &[0, 1]);
        assert_eq!(&sparse.row_indices[..], &[1]);
        if let Some(expected) = expected_numeric {
            assert_eq!(sparse.numeric_value_at(0), Some(expected));
        } else if matches!(class, ExactHostSparseClass::Complex(_)) {
            assert_eq!(
                sparse.complex_value_at(0),
                Some(crate::ComplexElement(2.0, -3.0))
            );
        }
        let alias = sparse.clone();
        assert_eq!(sparse.col_ptrs.as_ptr(), alias.col_ptrs.as_ptr());
        assert_eq!(sparse.row_indices.as_ptr(), alias.row_indices.as_ptr());
        assert_eq!(quota.retained.load(Ordering::SeqCst), charged);
        drop(sparse);
        assert_eq!(quota.retained.load(Ordering::SeqCst), charged);
        drop(alias);
        assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
        assert_eq!(quota.claims.load(Ordering::SeqCst), 0);
        let denied = permit(charged - 1);
        assert!(ExactHostSparseTensorAllocation::new(class, 2, 1, 1, &denied).is_err());
        assert_eq!(denied.retained.load(Ordering::SeqCst), 0);
    }
    assert!(
        ExactHostSparseTensorAllocation::requested_bytes_for_dimensions(
            ExactHostSparseClass::Logical,
            1,
            usize::MAX,
            0
        )
        .is_err()
    );

    // CSC physical storage depends on the short pointer array and nnz, not
    // on the mathematical dense rows × cols product.
    let class = ExactHostSparseClass::Logical;
    let charged =
        ExactHostSparseTensorAllocation::requested_bytes_for_dimensions(class, usize::MAX, 2, 0)
            .unwrap();
    let quota = permit(charged);
    let mut allocation =
        ExactHostSparseTensorAllocation::new(class, usize::MAX, 2, 0, &quota).unwrap();
    for column in 0..=2 {
        allocation.set_col_ptr(column, 0).unwrap();
    }
    let mut validation = allocation.into_validation().unwrap();
    while !validation.advance(NonZeroUsize::new(1).unwrap()).unwrap() {}
    let sparse = validation.into_tensor().unwrap();
    assert_eq!((sparse.rows, sparse.cols, sparse.nnz()), (usize::MAX, 2, 0));
    drop(sparse);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn failed_release_metadata_allocation_refunds_the_retained_claim() {
    let quota = permit(32);
    let reservation = quota.reserve_exact_claim(32).expect("owned claim");
    assert_eq!(quota.retained.load(Ordering::SeqCst), 32);
    // SAFETY: null never transfers a block into the release owner.
    let failed = unsafe {
        RustExactRelease::try_new_with(Layout::new::<f64>(), reservation, |_| std::ptr::null_mut())
    };
    assert!(matches!(
        failed,
        Err(ExactHostNumericAllocationError::AllocationFailed)
    ));
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
    let next = quota.reserve_exact_claim(32).expect("refunded capacity");
    drop(next);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn owned_release_header_retires_before_its_claim() {
    #[derive(Debug)]
    struct OrderedReservation {
        owner_released: Arc<AtomicBool>,
        claim_dropped: Arc<AtomicBool>,
    }

    impl Drop for OrderedReservation {
        fn drop(&mut self) {
            assert!(self.owner_released.load(Ordering::SeqCst));
            self.claim_dropped.store(true, Ordering::SeqCst);
        }
    }

    impl ExactHostAllocationReservation for OrderedReservation {
        fn retire_owned(self: Box<Self>) {
            let pointer = Box::into_raw(self);
            // SAFETY: the unique Box supplied the exact test allocation.
            let value = unsafe { std::ptr::read(pointer) };
            // SAFETY: the pointer came from Box with this exact layout.
            unsafe {
                dealloc(pointer.cast::<u8>(), Layout::new::<Self>());
            }
            drop(value);
        }
    }

    let owner_released = Arc::new(AtomicBool::new(false));
    let claim_dropped = Arc::new(AtomicBool::new(false));
    let claim = ExactHostAllocationClaim::owned(Box::new(OrderedReservation {
        owner_released: owner_released.clone(),
        claim_dropped: claim_dropped.clone(),
    }));
    let payload_layout = Layout::new::<f64>();
    let payload = NonNull::new(unsafe { alloc(payload_layout) }).expect("test payload allocation");
    let owner = RustExactRelease::try_new(payload_layout, claim).expect("release owner");
    // SAFETY: both test allocations came from the global allocator with the
    // exact layouts supplied to their respective deallocation callbacks.
    unsafe {
        RustExactRelease::release_owned_with(
            owner,
            payload,
            |pointer, layout| dealloc(pointer.as_ptr(), layout),
            |pointer, layout| {
                dealloc(pointer.as_ptr(), layout);
                owner_released.store(true, Ordering::SeqCst);
            },
        )
    };
    assert!(claim_dropped.load(Ordering::SeqCst));
}

#[test]
fn exact_float_provenance_uses_static_allocator_identity() {
    let quota = permit(floating_charge(std::mem::size_of::<f64>()));
    let allocation = ExactHostNumericAllocation::new(NumericDType::F64, 1, &quota).unwrap();
    let ExactAllocation::Floating(owner) = &allocation.allocation else {
        panic!("floating class must retain an adopted float owner");
    };
    assert!(matches!(
        &owner.provenance().allocator,
        Cow::Borrowed("runmat.exact-host-numeric.v1")
    ));
}

#[test]
fn canonical_requested_bytes_equal_the_retained_charge_for_every_numeric_class() {
    assert_eq!(ExactHostTensor2dAllocation::retained_claim_count(), 2);
    assert_eq!(ExactHostTensorAllocation::retained_claim_count(), 2);
    assert_eq!(ExactHostComplexTensorAllocation::retained_claim_count(), 2);
    assert_eq!(ExactHostLogicalTensorAllocation::retained_claim_count(), 2);
    let classes = [
        (NumericDType::I8, 1),
        (NumericDType::U8, 1),
        (NumericDType::I16, 2),
        (NumericDType::U16, 2),
        (NumericDType::I32, 4),
        (NumericDType::U32, 4),
        (NumericDType::I64, 8),
        (NumericDType::U64, 8),
        (NumericDType::F32, 4),
        (NumericDType::F64, 8),
    ];
    for (dtype, width) in classes {
        let requested = ExactHostTensor2dAllocation::requested_bytes(dtype, 2, 3).unwrap();
        let release = if dtype.integer_class().is_some() {
            0
        } else {
            std::mem::size_of::<RustExactRelease>()
        };
        assert_eq!(
            requested,
            6 * width
                + release
                + HostNumericBuffer::exact_metadata_bytes()
                + SharedShape::requested_bytes_for_rank(2).unwrap()
        );
        let quota = permit(requested);
        let output = ExactHostTensor2dAllocation::new(dtype, 2, 3, &quota).unwrap();
        assert_eq!(quota.retained.load(Ordering::SeqCst), requested);
        assert_eq!(quota.claims.load(Ordering::SeqCst), 2);
        let retained = output.into_tensor().unwrap();
        assert_eq!(quota.retained.load(Ordering::SeqCst), requested);
        drop(retained);
        assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
        assert_eq!(quota.claims.load(Ordering::SeqCst), 0);
    }
    assert_eq!(
        ExactHostTensor2dAllocation::requested_bytes(NumericDType::F32, usize::MAX, 2),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    );
}

#[test]
fn exact_nd_real_and_complex_outputs_retain_their_physical_claims() {
    let shape = [2, 1, 2];
    for dtype in [NumericDType::F32, NumericDType::F64] {
        let real_bytes = ExactHostTensorAllocation::requested_bytes(dtype, &shape).unwrap();
        assert_eq!(
            real_bytes,
            ExactHostTensorAllocation::requested_bytes_for_dimensions(dtype, 4, shape.len())
                .unwrap()
        );
        let real_quota = permit(real_bytes);
        let mut real = ExactHostTensorAllocation::new(dtype, &shape, &real_quota).unwrap();
        assert_eq!(real_quota.claims.load(Ordering::SeqCst), 2);
        if let Some(values) = real.as_f32_slice_mut() {
            values.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        } else {
            real.as_f64_slice_mut()
                .unwrap()
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        }
        let real = real.into_tensor().unwrap();
        assert_eq!(real.shape(), shape);
        assert_eq!(real.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(real_quota.retained.load(Ordering::SeqCst), real_bytes);
        drop(real);
        assert_eq!(real_quota.retained.load(Ordering::SeqCst), 0);
        assert_eq!(real_quota.claims.load(Ordering::SeqCst), 0);

        let complex_bytes =
            ExactHostComplexTensorAllocation::requested_bytes(dtype, &shape).unwrap();
        assert_eq!(
            complex_bytes,
            ExactHostComplexTensorAllocation::requested_bytes_for_dimensions(dtype, 4, shape.len())
                .unwrap()
        );
        let complex_quota = permit(complex_bytes);
        let mut complex =
            ExactHostComplexTensorAllocation::new(dtype, &shape, &complex_quota).unwrap();
        assert_eq!(complex_quota.claims.load(Ordering::SeqCst), 2);
        for index in 0..4 {
            complex.set(index, index as f64, -(index as f64)).unwrap();
        }
        let complex = complex.into_tensor().unwrap();
        assert_eq!(complex.shape(), shape);
        match dtype {
            NumericDType::F32 => assert_eq!(
                complex.as_f32_slice().unwrap()[3],
                crate::ComplexElement(3.0, -3.0)
            ),
            NumericDType::F64 => assert_eq!(
                complex.as_f64_slice().unwrap()[3],
                crate::ComplexElement(3.0, -3.0)
            ),
            _ => unreachable!(),
        }
        assert_eq!(complex_quota.retained.load(Ordering::SeqCst), complex_bytes);
        drop(complex);
        assert_eq!(complex_quota.retained.load(Ordering::SeqCst), 0);
        assert_eq!(complex_quota.claims.load(Ordering::SeqCst), 0);
    }
    assert_eq!(
        ExactHostComplexTensorAllocation::requested_bytes(NumericDType::F64, &[usize::MAX, 2]),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    );
}

#[test]
fn exact_logical_output_retains_independent_payload_and_shape_claims() {
    let shape = [2, 2];
    let requested = ExactHostLogicalTensorAllocation::requested_bytes(&shape).unwrap();
    let quota = permit(requested);
    let mut output = ExactHostLogicalTensorAllocation::new(&shape, &quota).unwrap();
    output.as_mut_slice().copy_from_slice(&[0, 1, 1, 0]);
    let pointer = output.as_mut_slice().as_ptr();
    assert_eq!(quota.retained.load(Ordering::SeqCst), requested);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 2);
    let output = output.into_array().unwrap();
    assert_eq!(&*output.shape, &shape);
    assert_eq!(output.data.as_slice(), &[0, 1, 1, 0]);
    assert_eq!(output.data.as_slice().as_ptr(), pointer);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 2);
    drop(output);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 0);
}

#[test]
fn retained_exact_claim_follows_last_original_owner_not_later_cow_copy() {
    let requested = ExactHostTensor2dAllocation::requested_bytes(NumericDType::U64, 1, 2).unwrap();
    let quota = permit(requested);
    let original = ExactHostTensor2dAllocation::new(NumericDType::U64, 1, 2, &quota)
        .unwrap()
        .into_tensor()
        .unwrap();
    let mut later_copy = original.clone();
    let shape_bytes = SharedShape::requested_bytes_for_rank(2).unwrap();
    assert_eq!(quota.retained.load(Ordering::SeqCst), requested);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 2);
    later_copy
        .set_numeric_assignment_at(0, NumericScalar::U64(7))
        .unwrap();
    assert_eq!(original.numeric_value_at(0), Some(NumericScalar::U64(0)));
    assert_eq!(later_copy.numeric_value_at(0), Some(NumericScalar::U64(7)));
    assert_eq!(quota.retained.load(Ordering::SeqCst), requested);
    drop(original);
    assert_eq!(quota.retained.load(Ordering::SeqCst), shape_bytes);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 1);
    // The ordinary payload COW belongs to the later mutation, while the
    // immutable shape remains shared under its independent initial claim.
    assert_eq!(later_copy.numeric_value_at(0), Some(NumericScalar::U64(7)));
    drop(later_copy);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 0);
}

#[test]
fn owned_reservation_releases_after_retained_exact_output() {
    #[derive(Debug)]
    struct OwnedPermit(Arc<AtomicUsize>);
    impl ExactHostAllocationPermit for OwnedPermit {
        fn reserve_exact(
            &self,
            _: usize,
        ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError>
        {
            panic!("exact route must use the owned claim")
        }

        fn reserve_exact_claim(
            &self,
            bytes: usize,
        ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
            self.0.fetch_add(bytes, Ordering::SeqCst);
            Ok(ExactHostAllocationClaim::owned(Box::new(Reservation {
                retained: self.0.clone(),
                claims: Arc::new(AtomicUsize::new(1)),
                bytes,
            })))
        }
    }
    let retained = Arc::new(AtomicUsize::new(0));
    let permit = OwnedPermit(retained.clone());
    let requested = ExactHostTensor2dAllocation::requested_bytes(NumericDType::U16, 1, 2)
        .expect("known owner layout");
    let output = ExactHostTensor2dAllocation::new(NumericDType::U16, 1, 2, &permit)
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_eq!(retained.load(Ordering::SeqCst), requested);
    drop(output);
    assert_eq!(retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_two_dimensional_owner_charges_shape_before_payload_and_retains_both() {
    let total = ExactHostTensor2dAllocation::requested_bytes(NumericDType::U64, 1, 2).unwrap();
    let denied = permit(total - 1);
    assert!(matches!(
        ExactHostTensor2dAllocation::new(NumericDType::U64, 1, 2, &denied),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(denied.retained.load(Ordering::SeqCst), 0);
    assert_eq!(denied.claims.load(Ordering::SeqCst), 0);

    let quota = permit(total);
    let mut allocation = ExactHostTensor2dAllocation::new(NumericDType::U64, 1, 2, &quota).unwrap();
    allocation
        .as_u64_slice_mut()
        .unwrap()
        .copy_from_slice(&[9_007_199_254_740_993, u64::MAX]);
    let original = allocation.as_u64_slice_mut().unwrap().as_ptr();
    assert_eq!(quota.retained.load(Ordering::SeqCst), total);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 2);
    let tensor = allocation.into_tensor().expect("exact geometry");
    assert_eq!(tensor.shape(), [1, 2]);
    assert_eq!(tensor.shape_capacity_for_testing(), 2);
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
    assert_eq!(
        unsafe { tensor.host_buffer().foreign_data_pointer() }.cast::<u64>(),
        original.cast_mut()
    );
    assert_eq!(quota.retained.load(Ordering::SeqCst), total);
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
    let total = ExactHostTensor2dAllocation::requested_bytes(NumericDType::F32, 3, 1).unwrap();
    let quota = permit(total);
    let mut allocation = ExactHostTensor2dAllocation::new(NumericDType::F32, 3, 1, &quota).unwrap();
    allocation
        .as_f32_slice_mut()
        .unwrap()
        .copy_from_slice(&[0.25, -0.5, 1.5]);
    let original = allocation.as_f32_slice_mut().unwrap().as_ptr();
    let tensor = allocation.into_tensor().expect("exact shape");
    assert_eq!(tensor.shape(), [3, 1]);
    assert_eq!(tensor.as_f32_slice(), Some([0.25, -0.5, 1.5].as_slice()));
    assert_eq!(tensor.as_f32_slice().unwrap().as_ptr(), original);
    assert_eq!(quota.retained.load(Ordering::SeqCst), total);
    drop(tensor);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_float_release_owner_is_precharged_before_payload_allocation() {
    let total = ExactHostTensor2dAllocation::requested_bytes(NumericDType::F64, 1, 1).unwrap();
    let denied = permit(total - 1);
    assert!(matches!(
        ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 1, &denied),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(denied.retained.load(Ordering::SeqCst), 0);
    let admitted = permit(total);
    let output = ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 1, &admitted)
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_eq!(admitted.retained.load(Ordering::SeqCst), total);
    drop(output);
    assert_eq!(admitted.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_parser_store_preserves_float_and_wide_integer_bits_without_reallocation() {
    let float_quota =
        permit(ExactHostTensor2dAllocation::requested_bytes(NumericDType::F64, 1, 2).unwrap());
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

    let single_quota =
        permit(ExactHostTensor2dAllocation::requested_bytes(NumericDType::F32, 1, 1).unwrap());
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

    let integer_quota =
        permit(ExactHostTensor2dAllocation::requested_bytes(NumericDType::U64, 1, 2).unwrap());
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
fn paired_integer_owner_retains_both_payloads_and_shape_through_aliases() {
    let bytes = ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
        NumericDType::U64,
        2,
        2,
    )
    .expect("paired quote");
    let quota = permit(bytes);
    let mut output =
        ExactHostIntegerComplexTensorAllocation::new(NumericDType::U64, &[1, 2], &quota)
            .expect("paired owner");
    output
        .set(
            0,
            NumericScalar::U64(9_007_199_254_740_993),
            NumericScalar::U64(u64::MAX),
        )
        .expect("first pair");
    output
        .set(1, NumericScalar::U64(u64::MAX), NumericScalar::U64(7))
        .expect("second pair");
    let tensor = output.into_tensor().expect("paired tensor");
    let storage = tensor.integer_storage().expect("paired storage");
    let real_pointer = unsafe { storage.real_buffer().foreign_data_pointer() };
    let imag_pointer = unsafe { storage.imag_buffer().foreign_data_pointer() };
    let mut alias = tensor.clone();
    assert_eq!(quota.claims.load(Ordering::SeqCst), 3);
    assert_eq!(quota.retained.load(Ordering::SeqCst), bytes);
    let alias_storage = alias.integer_storage().expect("paired alias");
    assert_eq!(
        unsafe { alias_storage.real_buffer().foreign_data_pointer() },
        real_pointer
    );
    assert_eq!(
        unsafe { alias_storage.imag_buffer().foreign_data_pointer() },
        imag_pointer
    );
    assert!(alias.set_f64_assignment_at(0, 3.0, 4.0).is_err());
    drop(tensor);
    assert_eq!(quota.retained.load(Ordering::SeqCst), bytes);
    alias
        .set_f64_assignment_at(0, 3.0, 4.0)
        .expect("unique retained pair mutates in place");
    assert!(alias.resize_zeroed_for_shape(vec![1, 3]).is_err());
    assert_eq!(alias.shape(), &[1, 2]);
    let (storage, shape) = alias.into_complex_storage_and_shape();
    let ComplexStorage::Integer(storage) = storage else {
        panic!("paired integer output");
    };
    let (real, imaginary) = storage.into_host_components();
    assert_eq!(unsafe { real.foreign_data_pointer() }, real_pointer);
    assert_eq!(unsafe { imaginary.foreign_data_pointer() }, imag_pointer);
    drop(imaginary);
    let projected = Tensor::from_shared_host_buffer(real, shape).expect("zero-copy component");
    assert_eq!(
        projected.integer_storage(),
        Some(&IntegerStorage::U64(vec![3, u64::MAX]))
    );
    assert_eq!(quota.claims.load(Ordering::SeqCst), 2);
    drop(projected);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 0);

    let denied = permit(bytes - 1);
    assert!(matches!(
        ExactHostIntegerComplexTensorAllocation::new(NumericDType::U64, &[1, 2], &denied,),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(denied.retained.load(Ordering::SeqCst), 0);
    assert_eq!(denied.claims.load(Ordering::SeqCst), 0);
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
            let quota = permit(integer_charge(bytes));
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
            assert_eq!(quota.retained.load(Ordering::SeqCst), integer_charge(bytes));
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
    let quota = permit(integer_charge(2) - 1);
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::U16, 1, &quota),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
    assert!(matches!(
        ExactHostNumericAllocation::new(NumericDType::U64, usize::MAX, &quota),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    ));
    let quota = permit(integer_charge(8));
    let owned = ExactHostNumericAllocation::new(NumericDType::I64, 1, &quota).unwrap();
    assert!(Tensor::from_host_buffer(owned.into_host_buffer(), vec![2, 1]).is_err());
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn integer_copy_on_write_keeps_only_the_original_block_charged() {
    let quota = permit(integer_charge(16));
    let mut owned = ExactHostNumericAllocation::new(NumericDType::U64, 2, &quota).unwrap();
    owned
        .as_u64_slice_mut()
        .unwrap()
        .copy_from_slice(&[9_007_199_254_740_993, u64::MAX]);
    let original = Tensor::from_host_buffer(owned.into_host_buffer(), vec![1, 2]).unwrap();
    let mut clone = original.clone();
    assert_eq!(quota.retained.load(Ordering::SeqCst), integer_charge(16));
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
    assert_eq!(quota.retained.load(Ordering::SeqCst), integer_charge(16));
    drop(original);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn sole_owner_integer_mutation_keeps_block_and_materialization_releases_claim() {
    let quota = permit(integer_charge(16));
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
    assert_eq!(quota.retained.load(Ordering::SeqCst), integer_charge(16));
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
    let permit = permit(integer_charge(4));
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
    assert_eq!(permit.retained.load(Ordering::SeqCst), integer_charge(4));
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

#[test]
fn adopted_rust_vecs_keep_payload_and_shape_backings_until_exact_claim_refund() {
    let mut f32_values = Vec::with_capacity(8);
    f32_values.extend([1.0_f32, 2.0]);
    let f32_pointer = f32_values.as_ptr();
    let mut f32_shape = Vec::with_capacity(4);
    f32_shape.extend([1_usize, 2]);
    let f32_shape_pointer = f32_shape.as_ptr();
    let f32_shape_capacity = f32_shape.capacity();
    let f32_charge = ExactHostTensorAllocation::adopted_requested_bytes(
        NumericDType::F32,
        f32_values.capacity(),
        f32_shape.capacity(),
    )
    .unwrap();

    let mut f64_values = Vec::with_capacity(6);
    f64_values.extend([3.0_f64, 4.0]);
    let f64_pointer = f64_values.as_ptr();
    let mut f64_shape = Vec::with_capacity(3);
    f64_shape.extend([2_usize, 1]);
    let f64_shape_pointer = f64_shape.as_ptr();
    let f64_shape_capacity = f64_shape.capacity();
    let f64_charge = ExactHostTensorAllocation::adopted_requested_bytes(
        NumericDType::F64,
        f64_values.capacity(),
        f64_shape.capacity(),
    )
    .unwrap();

    let quota = permit(f32_charge + f64_charge);
    let f32_tensor = ExactHostTensorAllocation::adopt_f32_vec(f32_values, f32_shape, &quota)
        .expect("adopt F32 Vec");
    let f64_tensor = ExactHostTensorAllocation::adopt_f64_vec(f64_values, f64_shape, &quota)
        .expect("adopt F64 Vec");
    assert_eq!(f32_tensor.as_f32_slice().unwrap().as_ptr(), f32_pointer);
    assert_eq!(f32_tensor.shape().as_ptr(), f32_shape_pointer);
    assert_eq!(f32_tensor.shape_capacity_for_testing(), f32_shape_capacity);
    assert_eq!(f64_tensor.as_f64_slice().unwrap().as_ptr(), f64_pointer);
    assert_eq!(f64_tensor.shape().as_ptr(), f64_shape_pointer);
    assert_eq!(f64_tensor.shape_capacity_for_testing(), f64_shape_capacity);
    assert_eq!(
        quota.retained.load(Ordering::SeqCst),
        f32_charge + f64_charge
    );
    assert_eq!(quota.claims.load(Ordering::SeqCst), 4);
    drop(f32_tensor);
    assert_eq!(quota.retained.load(Ordering::SeqCst), f64_charge);
    drop(f64_tensor);
    assert_eq!(quota.retained.load(Ordering::SeqCst), 0);
    assert_eq!(quota.claims.load(Ordering::SeqCst), 0);

    let empty_shape = vec![0_usize, 2];
    let empty_charge = ExactHostTensorAllocation::adopted_requested_bytes(
        NumericDType::F32,
        0,
        empty_shape.capacity(),
    )
    .unwrap();
    let empty_quota = permit(empty_charge);
    let empty = ExactHostTensorAllocation::adopt_f32_vec(Vec::new(), empty_shape, &empty_quota)
        .expect("zero-capacity payload adoption");
    assert!(empty.as_f32_slice().unwrap().is_empty());
    assert_eq!(empty_quota.retained.load(Ordering::SeqCst), empty_charge);
    drop(empty);
    assert_eq!(empty_quota.retained.load(Ordering::SeqCst), 0);

    let mut refused_values = Vec::with_capacity(5);
    refused_values.extend([7.0_f32, 8.0]);
    let refused_shape = vec![1_usize, 2];
    let refused_charge = ExactHostTensorAllocation::adopted_requested_bytes(
        NumericDType::F32,
        refused_values.capacity(),
        refused_shape.capacity(),
    )
    .unwrap();
    let refused = permit(refused_charge - 1);
    assert!(matches!(
        ExactHostTensorAllocation::adopt_f32_vec(refused_values, refused_shape, &refused),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(refused.retained.load(Ordering::SeqCst), 0);
    assert_eq!(refused.claims.load(Ordering::SeqCst), 0);

    let mut integers = Vec::with_capacity(5);
    integers.extend([1_u64 << 63, u64::MAX]);
    let integer_pointer = integers.as_ptr();
    let integer_shape = vec![1, 2];
    let integer_charge = ExactHostTensorAllocation::adopted_requested_bytes(
        NumericDType::U64,
        integers.capacity(),
        integer_shape.capacity(),
    )
    .unwrap();
    let integer_quota = permit(integer_charge);
    let integer = ExactHostTensorAllocation::adopt_u64_vec(integers, integer_shape, &integer_quota)
        .expect("adopt exact U64 Vec");
    let IntegerStorage::U64(integer_values) = integer.integer_storage().expect("integer") else {
        panic!("preserve U64 class");
    };
    assert_eq!(integer_values.as_ptr(), integer_pointer);
    assert_eq!(integer_values.as_slice(), &[1_u64 << 63, u64::MAX]);
    drop(integer);
    assert_eq!(integer_quota.retained.load(Ordering::SeqCst), 0);

    let mut logical_values = Vec::with_capacity(6);
    logical_values.extend([0_u8, 1]);
    let logical_pointer = logical_values.as_ptr();
    let logical_shape = vec![1, 2];
    let logical_charge = ExactHostLogicalTensorAllocation::adopted_requested_bytes(
        logical_values.capacity(),
        logical_shape.capacity(),
    )
    .unwrap();
    let logical_quota = permit(logical_charge);
    let logical = ExactHostLogicalTensorAllocation::adopt_u8_vec(
        logical_values,
        logical_shape,
        &logical_quota,
    )
    .expect("adopt logical Vec");
    assert_eq!(logical.data.as_slice().as_ptr(), logical_pointer);
    drop(logical);
    assert_eq!(logical_quota.retained.load(Ordering::SeqCst), 0);

    let mut interleaved = Vec::with_capacity(8);
    interleaved.extend([1.0_f32, -2.0, 3.0, 4.0]);
    let complex_pointer = interleaved.as_ptr();
    let complex_shape = vec![1, 2];
    let complex_charge = ExactHostComplexTensorAllocation::adopted_requested_bytes(
        NumericDType::F32,
        interleaved.capacity(),
        complex_shape.capacity(),
    )
    .unwrap();
    let complex_quota = permit(complex_charge);
    let complex = ExactHostComplexTensorAllocation::adopt_f32_interleaved_vec(
        interleaved,
        complex_shape,
        &complex_quota,
    )
    .expect("adopt interleaved complex Vec");
    assert_eq!(
        complex
            .as_f32_slice()
            .expect("complex F32")
            .as_ptr()
            .cast::<f32>(),
        complex_pointer
    );
    drop(complex);
    assert_eq!(complex_quota.retained.load(Ordering::SeqCst), 0);
}

#[test]
fn unmanaged_tensor_quotes_match_the_real_owner_layout_without_claim_headers() {
    let shape = [2, 3];
    let release_bytes = Layout::new::<RustExactRelease>().size();

    let numeric = ExactHostTensorAllocation::unmanaged_requested_bytes(NumericDType::F64, &shape)
        .expect("numeric unmanaged quote");
    assert_eq!(
        ExactHostTensorAllocation::requested_bytes(NumericDType::F64, &shape)
            .expect("numeric managed quote"),
        numeric + release_bytes
    );
    assert_eq!(
        ExactHostTensorAllocation::requested_bytes(NumericDType::I32, &shape)
            .expect("integer managed quote"),
        ExactHostTensorAllocation::unmanaged_requested_bytes(NumericDType::I32, &shape)
            .expect("integer unmanaged quote")
    );

    let complex =
        ExactHostComplexTensorAllocation::unmanaged_requested_bytes(NumericDType::F32, &shape)
            .expect("complex unmanaged quote");
    assert_eq!(
        ExactHostComplexTensorAllocation::requested_bytes(NumericDType::F32, &shape)
            .expect("complex managed quote"),
        complex + release_bytes
    );

    let logical = ExactHostLogicalTensorAllocation::unmanaged_requested_bytes(&shape)
        .expect("logical unmanaged quote");
    assert_eq!(
        ExactHostLogicalTensorAllocation::requested_bytes(&shape).expect("logical managed quote"),
        logical + release_bytes
    );
}
