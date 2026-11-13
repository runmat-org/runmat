use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::*;
use crate::{
    CellArray, CharArray, ExactHostAllocationReservation, SharedShape, StringArray, StringElement,
    Utf16CodeUnit, Utf8Text, ValueSequence, ValueSequenceError,
};

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
fn staged_exact_owner_keeps_claim_through_partial_and_shared_lifetimes() {
    #[derive(Clone, Debug)]
    struct Child {
        claim_live: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for Child {
        fn drop(&mut self) {
            assert!(self.claim_live.load(Ordering::SeqCst) > 0);
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    let bytes = ExactArrayStorage::<Child>::requested_bytes(2).unwrap();
    let live = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let permit = Permit {
        limit: bytes,
        live: live.clone(),
    };
    let mut pending = ExactArrayStorage::<Child>::try_begin(2, &permit).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), bytes);
    assert_eq!(pending.expected_len(), 2);
    assert!(pending.initialized().is_empty());
    pending
        .try_push(Child {
            claim_live: live.clone(),
            drops: drops.clone(),
        })
        .unwrap();
    let pending = pending.try_finish().unwrap_err();
    assert_eq!(pending.initialized().len(), 1);
    // Cancelling a partial producer retires its child before the claim.
    drop(pending);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let mut complete = ExactArrayStorage::<Child>::try_begin(2, &permit).unwrap();
    for _ in 0..2 {
        complete
            .try_push(Child {
                claim_live: live.clone(),
                drops: drops.clone(),
            })
            .unwrap();
    }
    let excess = complete.try_push(Child {
        claim_live: live.clone(),
        drops: drops.clone(),
    });
    assert!(excess.is_err());
    assert_eq!(complete.initialized().len(), 2);
    drop(excess);
    let owner = complete.try_finish().unwrap();
    assert!(owner.is_exact());
    assert_eq!(owner.capacity(), 2);
    let pointer = owner.as_slice().as_ptr();
    let alias = owner.clone();
    assert_eq!(alias.as_slice().as_ptr(), pointer);
    drop(owner);
    assert_eq!(live.load(Ordering::SeqCst), bytes);
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 4);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let mut empty = OutputListStorage::try_begin(0, &permit).unwrap();
    assert_eq!(empty.try_push(Value::Num(1.0)), Err(Value::Num(1.0)));
    assert!(empty.try_finish().unwrap().is_exact());
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn shared_ordinary_promotion_claims_one_original_owner_until_final_alias() {
    let values = vec![Value::Num(3.0), Value::Num(7.0)];
    let capacity = values.capacity();
    let pointer = values.as_ptr();
    let owner = OutputListStorage::shared_unbounded(values);
    let alias = owner.clone();
    let bytes = OutputListStorage::requested_bytes_for_capacity(capacity).unwrap();
    assert_eq!(owner.unclaimed_shared_bytes().unwrap(), Some(bytes));
    let live = Arc::new(AtomicUsize::new(0));
    let denied = Permit {
        limit: bytes - 1,
        live: live.clone(),
    };
    assert_eq!(
        owner.try_promote_shared_claim(&denied),
        Err(ExactArrayClaimPromotionError::Allocation(
            ExactHostNumericAllocationError::ExceedsPermit
        ))
    );
    assert_eq!(owner.as_ptr(), pointer);
    assert!(!alias.is_exact());
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let permit = Permit {
        limit: bytes,
        live: live.clone(),
    };
    owner.try_promote_shared_claim(&permit).unwrap();
    alias.try_promote_shared_claim(&denied).unwrap();
    assert!(owner.is_exact() && alias.is_exact());
    assert_eq!(owner.unclaimed_shared_bytes().unwrap(), None);
    assert_eq!(live.load(Ordering::SeqCst), bytes);
    drop(owner);
    assert_eq!(alias.as_ptr(), pointer);
    assert_eq!(live.load(Ordering::SeqCst), bytes);
    drop(alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn callable_generation_retires_before_aliased_capture_claim() {
    let closure = crate::Closure::new(
        "capture_owner".into(),
        None,
        runmat_types::FunctionSourceRole::Anonymous,
        None,
        vec![Value::Num(9.0)],
        None,
    );
    let capture_alias = closure.captures.clone();
    let capture_pointer = capture_alias.as_ptr();
    let capture_bytes = capture_alias.unclaimed_shared_bytes().unwrap().unwrap();
    let name_alias = closure.function_name.clone();
    let name_pointer = name_alias.as_str().as_ptr();
    let name_bytes = name_alias.unclaimed_bytes().unwrap().unwrap();
    let data_alias = closure.clone();
    let data_bytes = crate::Closure::resolved_header_bytes();
    let callable = Value::Closure(closure);
    let quote = crate::SharedValueGeneration::callable_quote(&callable)
        .unwrap()
        .unwrap();
    // Generation, immutable callable data, name text and capture slots are
    // four distinct original physical owners, not one carrier certificate.
    assert_eq!(
        quote.retained_claims,
        1 + 1 + usize::from(name_bytes != 0) + usize::from(capture_bytes != 0)
    );
    assert_eq!(
        quote.requested_bytes,
        crate::SharedValueGeneration::requested_bytes() + data_bytes + name_bytes + capture_bytes
    );
    let live = Arc::new(AtomicUsize::new(0));
    let permit = Permit {
        limit: quote.requested_bytes,
        live: live.clone(),
    };
    let mut input = crate::CallableGenerationInput::new(callable);
    let generation = crate::SharedValueGeneration::prepare_callable(&mut input, &permit).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), quote.requested_bytes);
    assert_eq!(capture_alias.as_ptr(), capture_pointer);
    drop(generation);
    assert_eq!(
        live.load(Ordering::SeqCst),
        data_bytes + name_bytes + capture_bytes
    );
    assert_eq!(data_alias.function_name.as_str().as_ptr(), name_pointer);
    assert_eq!(data_alias.captures.as_ptr(), capture_pointer);
    // Explicit data detachment clones only canonical child owners. The old
    // Data header retires while those original child claims remain live.
    let detached = data_alias.into_data();
    assert_eq!(live.load(Ordering::SeqCst), name_bytes + capture_bytes);
    assert_eq!(detached.function_name.as_str().as_ptr(), name_pointer);
    assert_eq!(detached.captures.as_ptr(), capture_pointer);
    drop(detached);
    assert_eq!(live.load(Ordering::SeqCst), name_bytes + capture_bytes);
    drop(name_alias);
    assert_eq!(live.load(Ordering::SeqCst), capture_bytes);
    assert!(capture_alias.is_exact());
    drop(capture_alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    // A decoder can prepare the original child owners first, then fund only
    // the new immutable header. Neither construction nor alias extraction
    // may detach those children or quietly allocate a second text owner.
    let name = Utf8Text::new("decoded_owner".into());
    let pointer = name.as_ptr();
    let named = crate::NamedFunctionHandle::from_text_owner(name.clone());
    assert_eq!(named.name().as_ptr(), pointer);
    let data = crate::ClosureData {
        function_name: name,
        binding: None,
        source_role: runmat_types::FunctionSourceRole::Anonymous,
        source_file: None,
        captures: OutputListStorage::empty(),
        lexical_environment: None,
    };
    let header = crate::Closure::try_from_data_with_exact_header(data, &permit).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), data_bytes);
    let alias = header.clone();
    let detached = header.into_data();
    assert_eq!(detached.function_name.as_ptr(), pointer);
    assert_eq!(live.load(Ordering::SeqCst), data_bytes);
    drop(alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert_eq!(detached.function_name.as_ptr(), pointer);
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
    assert!(list.is_unique());
    assert_eq!(list.capacity(), 3);
    let pointer = list.as_slice().as_ptr();
    let alias = list.clone();
    assert!(!list.is_unique());
    assert!(!alias.is_unique());
    assert_eq!(pointer, alias.as_slice().as_ptr());
    assert_eq!(permit.live.load(Ordering::SeqCst), bytes);
    let list = list.try_into_owned().unwrap_err();
    assert_eq!(pointer, list.as_slice().as_ptr());
    drop(alias);
    assert!(list.is_unique());
    let mut owned = list.try_into_owned().unwrap();
    assert_eq!(pointer, owned.as_slice().as_ptr());
    assert_eq!(owned.len(), 3);
    assert!(!owned.is_empty());
    assert_eq!(owned.try_push(Value::Num(99.0)), Err(Value::Num(99.0)));
    assert_eq!(owned.try_remove(usize::MAX), None);
    assert_eq!(owned.try_remove(2), Some(Value::Num(2.0)));
    owned.try_push(Value::Num(2.0)).unwrap();
    owned.as_mut_slice()[1] = Value::Num(42.0);
    let list = owned.try_into_storage().unwrap();
    assert!(list.is_exact());
    assert_eq!(pointer, list.as_ptr());
    assert_eq!(permit.live.load(Ordering::SeqCst), bytes);
    let owned = list.try_into_owned().unwrap();
    let mut values = owned.into_iter();
    assert_eq!(values.next(), Some(Value::Num(0.0)));
    assert_eq!(values.next_back(), Some(Value::Num(2.0)));
    assert_eq!(values.len(), 1);
    assert_eq!(values.next(), Some(Value::Num(42.0)));
    assert_eq!(permit.live.load(Ordering::SeqCst), bytes);
    drop(values);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);

    let chars_bytes = ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(3).unwrap();
    let chars_permit = Permit {
        limit: chars_bytes,
        live: Arc::new(AtomicUsize::new(0)),
    };
    let mut chars = ExactArrayStorage::<Utf16CodeUnit>::try_initialize(3, &chars_permit, |index| {
        Ok::<_, ()>(
            [
                Utf16CodeUnit::from_raw(b'a'.into()),
                Utf16CodeUnit::from_raw(0xd800),
                Utf16CodeUnit::from_raw(b'c'.into()),
            ][index],
        )
    })
    .unwrap();
    assert!(chars
        .try_push_unbounded(Utf16CodeUnit::from_raw(b'd'.into()))
        .is_err());
    chars.try_as_mut_slice_unique().unwrap()[1] = Utf16CodeUnit::from_raw(0xdc00);
    let alias = chars.clone();
    assert!(chars.try_as_mut_slice_unique().is_none());
    let mut copied_alias = alias.into_code_unit_iter();
    assert_eq!(
        copied_alias.next(),
        Some(Utf16CodeUnit::from_raw(b'a'.into()))
    );
    assert_eq!(
        chars.as_slice(),
        &[
            Utf16CodeUnit::from_raw(b'a'.into()),
            Utf16CodeUnit::from_raw(0xdc00),
            Utf16CodeUnit::from_raw(b'c'.into()),
        ]
    );
    drop(chars);
    assert_eq!(chars_permit.live.load(Ordering::SeqCst), chars_bytes);
    assert_eq!(
        copied_alias.map(Utf16CodeUnit::raw).collect::<Vec<_>>(),
        vec![0xdc00, u16::from(b'c')]
    );
    assert_eq!(chars_permit.live.load(Ordering::SeqCst), 0);
}

#[test]
fn ordinary_alias_shares_children_and_exact_adoption_keeps_original_backing() {
    let ordinary = OutputListStorage::shared_unbounded(vec![Value::Num(7.0)]);
    let pointer = ordinary.as_slice().as_ptr();
    let alias = ordinary.clone();
    assert_eq!(alias.as_slice().as_ptr(), pointer);
    assert!(!ordinary.is_exact());
    assert!(ordinary.try_into_owned().is_err());
    let mut growing = alias.clone();
    assert!(growing.try_push_unbounded(Value::Num(8.0)).is_err());
    growing.make_unique_unbounded().unwrap();
    growing.try_push_unbounded(Value::Num(8.0)).unwrap();
    assert_eq!(growing.as_slice(), &[Value::Num(7.0), Value::Num(8.0)]);
    assert_eq!(alias.as_slice(), &[Value::Num(7.0)]);
    drop(alias);

    let mut bytes = Vec::with_capacity(8);
    bytes.extend_from_slice(b"abc");
    let pointer = bytes.as_ptr();
    let quoted = ExactArrayStorage::<u8>::requested_bytes_for_capacity(bytes.capacity()).unwrap();
    let denied = Permit {
        limit: quoted - 1,
        live: Arc::new(AtomicUsize::new(0)),
    };
    let denied_error = ExactArrayStorage::try_adopt_exact(bytes, &denied).unwrap_err();
    assert_eq!(
        denied_error.source(),
        &ExactHostNumericAllocationError::ExceedsPermit
    );
    let bytes = denied_error.try_into_unclaimed_values().unwrap();
    assert_eq!(bytes.as_ptr(), pointer);
    assert_eq!(denied.live.load(Ordering::SeqCst), 0);
    let permit = Permit {
        limit: quoted,
        live: denied.live,
    };
    let adopted = ExactArrayStorage::try_adopt_exact(bytes, &permit).unwrap();
    assert!(adopted.is_exact());
    assert_eq!(adopted.as_slice().as_ptr(), pointer);
    assert_eq!(permit.live.load(Ordering::SeqCst), quoted);
    let alias = adopted.clone();
    drop(adopted);
    assert_eq!(permit.live.load(Ordering::SeqCst), quoted);
    drop(alias);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);

    let mut bytes = Vec::with_capacity(8);
    bytes.extend_from_slice(b"xyz");
    let pointer = bytes.as_ptr();
    // SAFETY: the allocator returns null and therefore transfers no block.
    let failed = unsafe {
        ExactArrayStorage::try_adopt_exact_with(bytes, &permit, |_| std::ptr::null_mut())
    }
    .unwrap_err();
    assert_eq!(
        failed.source(),
        &ExactHostNumericAllocationError::AllocationFailed
    );
    assert_eq!(failed.values.as_ptr(), pointer);
    assert_eq!(permit.live.load(Ordering::SeqCst), quoted);
    let failed = failed.try_into_unclaimed_values().unwrap_err();
    drop(failed);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
}

#[test]
fn admitted_char_and_cell_arrays_keep_both_backing_claims_through_aliases() {
    let char_bytes = ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(8).unwrap();
    let cell_bytes = OutputListStorage::requested_bytes(2).unwrap();
    let shape_bytes = SharedShape::requested_bytes_for_rank(2).unwrap();
    let char_shape_bytes = SharedShape::requested_bytes_for_rank(3).unwrap();
    let live = Arc::new(AtomicUsize::new(0));
    let permit = Permit {
        limit: char_bytes + cell_bytes + shape_bytes + char_shape_bytes,
        live: live.clone(),
    };
    let chars = ExactArrayStorage::<Utf16CodeUnit>::try_initialize(8, &permit, |index| {
        Ok::<_, ()>(Utf16CodeUnit::from_raw(
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][index] as u16,
        ))
    })
    .unwrap();
    let char_shape = SharedShape::new_admitted_from_slice(&[2, 2, 2], &permit).unwrap();
    let mut chars = CharArray::from_admitted(chars, char_shape).unwrap();
    assert!(chars.validate_layout().is_ok());
    chars.rows = 0;
    assert!(chars.validate_layout().is_err());
    chars.rows = 2;
    let char_alias = chars.clone();
    assert!(chars.try_data_mut_unique().is_none());
    assert_eq!(
        chars
            .data()
            .iter()
            .map(|unit| unit.raw())
            .collect::<Vec<_>>(),
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].map(|value| value as u16)
    );
    assert_eq!(
        String::from_utf16(
            &chars
                .iter_column_major()
                .map(|unit| unit.raw())
                .collect::<Vec<_>>()
        )
        .unwrap(),
        "acbdegfh"
    );
    assert_eq!(
        chars.to_column_major(),
        char_alias.iter_column_major().collect::<Vec<_>>()
    );
    assert_eq!(chars.data().as_ptr(), char_alias.data().as_ptr());

    let cells = OutputListStorage::try_initialize(2, &permit, |index| {
        Ok::<_, ()>(Value::Num(index as f64))
    })
    .unwrap();
    let cell_shape = SharedShape::new_admitted_from_slice(&[1, 2], &permit).unwrap();
    let mut cells = CellArray::from_admitted(cells, cell_shape).unwrap();
    let cell_alias = cells.clone();
    assert!(cells.try_data_mut_unique().is_none());
    assert!(cells.push_row_element(Value::Num(3.0)).is_err());
    assert_eq!(cells.data(), &[Value::Num(0.0), Value::Num(1.0)]);
    assert_eq!(live.load(Ordering::SeqCst), permit.limit);
    drop((chars, cells));
    assert_eq!(live.load(Ordering::SeqCst), permit.limit);
    drop((char_alias, cell_alias));
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let mut outer = OutputListStorage::try_initialize(1, &permit, |_| {
        Ok::<_, ()>(Value::Cell(
            CellArray::new(vec![Value::Num(37.0)], 1, 1).unwrap(),
        ))
    })
    .unwrap();
    assert!(matches!(
        outer.retain_value_at(1),
        Err(RetainedArrayValueError::IndexOutOfBounds)
    ));
    let parent = outer.retain_value_at(0).unwrap();
    let child = parent
        .project(|value| match value {
            Value::Cell(cell) => cell.data.first(),
            _ => None,
        })
        .unwrap();
    assert!(outer.try_as_mut_slice_unique().is_none());
    let child_pointer = child.value() as *const Value;
    let alias = child.clone();
    drop((outer, parent, child));
    assert_eq!(alias.value() as *const Value, child_pointer);
    assert_eq!(alias.value(), &Value::Num(37.0));
    assert_eq!(
        live.load(Ordering::SeqCst),
        OutputListStorage::requested_bytes(1).unwrap()
    );
    assert!(alias.project(|_| None).is_none());
    drop(alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert!(matches!(
        OutputListStorage::unbounded(vec![Value::Num(0.0)]).retain_value_at(0),
        Err(RetainedArrayValueError::UnsharedBacking)
    ));

    for shape in [vec![], vec![0, 3], vec![3], vec![2, 1, 2]] {
        let count = if shape.is_empty() {
            0
        } else {
            shape.iter().product()
        };
        let mut chars =
            CharArray::from_unicode_scalars_with_shape(vec![' '; count], shape).unwrap();
        assert!(chars.validate_layout().is_ok());
        chars.cols += 1;
        assert!(chars.validate_layout().is_err());
    }
}

#[test]
fn string_array_admission_retains_text_collection_and_shape_without_byte_copy() {
    assert!(StringArray::new(Vec::new(), vec![usize::MAX, 2]).is_err());
    let mut original = String::with_capacity(12);
    original.push_str("hello");
    let pointer = original.as_ptr();
    let text_bytes = Utf8Text::requested_bytes_for_capacity(original.capacity()).unwrap();
    let outer_bytes = ExactArrayStorage::<StringElement>::requested_bytes(1).unwrap();
    let shape_bytes = SharedShape::requested_bytes_for_rank(2).unwrap();
    assert_eq!(
        StringArray::requested_bytes_for_new_dimensions(1, 2, [original.capacity()]).unwrap(),
        text_bytes + outer_bytes + shape_bytes
    );
    let live = Arc::new(AtomicUsize::new(0));
    let permit = Permit {
        limit: text_bytes + outer_bytes + shape_bytes,
        live: live.clone(),
    };
    let text = StringElement::text(original)
        .try_admit_exact(&permit)
        .unwrap();
    assert_eq!(text.as_str().unwrap().as_ptr(), pointer);
    let ordinary_values = vec![text.clone()];
    let outer_pointer = ordinary_values.as_ptr();
    let ordinary = StringArray::from_elements_unbounded(ordinary_values, vec![1, 1]).unwrap();
    assert_eq!(ordinary.data().as_ptr(), outer_pointer);
    assert_eq!(ordinary.data()[0].as_str().unwrap().as_ptr(), pointer);
    assert_eq!(ordinary.data, vec!["hello"]);
    assert_eq!(ordinary.data, ["hello"]);
    assert_ne!(ordinary.data, ["other"]);
    assert_eq!(ordinary.shape, [1usize, 1]);
    assert_eq!(ordinary.shape, [1usize, 1].as_slice());
    assert_eq!(live.load(Ordering::SeqCst), text_bytes);
    assert!(!ordinary.data.is_exact());
    let mut grown = ordinary.clone();
    grown.grow_unbounded(vec![2, 1]).unwrap();
    assert_eq!(grown.shape(), [2, 1]);
    assert_eq!((grown.rows, grown.cols), (2, 1));
    assert_eq!(grown.data()[0].as_str().unwrap().as_ptr(), pointer);
    assert_eq!(grown.data()[1].as_str(), Some(""));
    assert_eq!(ordinary.shape(), [1, 1]);
    assert!(grown.grow_unbounded(vec![usize::MAX, 2]).is_err());
    assert_eq!(grown.shape(), [2, 1]);
    drop(grown);
    drop(ordinary);
    assert_eq!(live.load(Ordering::SeqCst), text_bytes);
    assert!(StringArray::from_texts_unbounded(Vec::new(), vec![]).is_err());
    let scalar = Value::String(text.clone());
    assert_eq!(scalar.string_text().unwrap().as_ptr(), pointer);
    let mut text = Some(text);
    let data = ExactArrayStorage::try_initialize(1, &permit, |_| Ok::<_, ()>(text.take().unwrap()))
        .unwrap();
    let shape = SharedShape::new_admitted_from_slice(&[1, 1], &permit).unwrap();
    let array = StringArray::from_admitted(data, shape).unwrap();
    assert_eq!(array.data()[0].as_str().unwrap().as_ptr(), pointer);
    assert_eq!(live.load(Ordering::SeqCst), permit.limit);
    let view = StringArray::from_shared_storage(array.data.clone(), array.shape.clone()).unwrap();
    assert_eq!(view.data().as_ptr(), array.data().as_ptr());
    assert_eq!(view.shape().as_ptr(), array.shape().as_ptr());
    assert_eq!(live.load(Ordering::SeqCst), permit.limit);
    drop(view);
    let alias = array.clone();
    let mut rejected_growth = array.clone();
    assert!(rejected_growth.grow_unbounded(vec![2, 1]).is_err());
    assert_eq!(rejected_growth.data().as_ptr(), array.data().as_ptr());
    assert_eq!(rejected_growth.shape(), [1, 1]);
    drop(rejected_growth);
    drop(array);
    assert_eq!(live.load(Ordering::SeqCst), permit.limit);
    drop(alias);
    assert_eq!(live.load(Ordering::SeqCst), text_bytes);
    drop(scalar);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let rank_zero = StringArray::new(vec!["only".to_owned()], vec![]).unwrap();
    assert_eq!(rank_zero.data()[0].as_str(), Some("only"));
    assert!(StringArray::new(Vec::new(), vec![]).is_err());
    let rank_zero_text = StringElement::text("only")
        .try_admit_exact(&permit)
        .unwrap();
    let mut rank_zero_text = Some(rank_zero_text);
    let rank_zero_data = ExactArrayStorage::try_initialize(1, &permit, |_| {
        Ok::<_, ()>(rank_zero_text.take().unwrap())
    })
    .unwrap();
    let rank_zero_shape = SharedShape::new_admitted_from_slice(&[], &permit).unwrap();
    let rank_zero_exact = StringArray::from_admitted(rank_zero_data, rank_zero_shape).unwrap();
    assert!(rank_zero_exact.shape().is_empty());
    assert_eq!(rank_zero_exact.data()[0].as_str(), Some("only"));
    drop(rank_zero_exact);
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn missing_string_elements_are_distinct_from_literal_text_and_keep_storage_owners() {
    for literal in ["<missing>", "<Missing>", "", "λ🙂"] {
        let element = StringElement::text(literal);
        assert!(!element.is_missing());
        assert_eq!(element.as_str(), Some(literal));
        assert_eq!(String::try_from(&Value::String(element)).unwrap(), literal);
    }
    let live = Arc::new(AtomicUsize::new(0));
    let permit = Permit {
        limit: 0,
        live: live.clone(),
    };
    let missing = StringElement::Missing.try_admit_exact(&permit).unwrap();
    assert!(missing.is_missing());
    assert_eq!(missing.capacity(), 0);
    assert!(missing.shared_storage_identity().is_none());
    let mut checkpoints = 0;
    let copied = missing
        .try_copy_exact_with(&permit, |_| {
            checkpoints += 1;
            Ok::<(), ()>(())
        })
        .unwrap();
    assert_eq!(checkpoints, 1);
    assert_eq!(copied, StringElement::Missing);
    assert!(matches!(
        missing.try_copy_exact_with(&permit, |_| Err("stop")),
        Err(ExactArrayInitializationError::Element("stop"))
    ));
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert!(String::try_from(&Value::String(missing.clone())).is_err());
    let array = StringArray::from_elements_unbounded(
        vec![missing.clone(), StringElement::text("<missing>")],
        vec![1, 2],
    )
    .unwrap();
    let alias = array.clone();
    assert_eq!(array.data().as_ptr(), alias.data().as_ptr());
    assert!(alias.data()[0].is_missing());
    assert_eq!(alias.data()[1].as_str(), Some("<missing>"));
    assert_eq!(
        format!("{alias}"),
        "1x2 string array\n  <missing>  \"<missing>\""
    );
    let distinct =
        std::collections::HashSet::from([missing.clone(), StringElement::text("<missing>")]);
    assert_eq!(distinct.len(), 2);
    let mut sorted = [
        missing,
        StringElement::text("<missing>"),
        StringElement::text(""),
    ];
    sorted.sort();
    assert!(sorted.last().unwrap().is_missing());
}

#[test]
fn initialization_refuses_before_elements_and_refunds_failed_construction() {
    assert_eq!(
        ExactArrayStorage::<()>::requested_bytes(1),
        Err(ExactHostNumericAllocationError::LayoutOverflow)
    );
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
    let mut empty = OutputListStorage::try_initialize(0, &permit, |_| Err::<Value, _>(()))
        .expect("empty output does not initialize elements");
    assert!(empty.is_empty());
    assert!(empty.try_push_unbounded(Value::Num(1.0)).is_err());
    assert_eq!(empty, OutputListStorage::unbounded(Vec::new()));
    assert!(empty
        .try_into_owned()
        .unwrap()
        .try_into_unbounded_vec()
        .is_err());
    let mut ordinary_empty = OutputListStorage::unbounded(Vec::new());
    ordinary_empty.try_push_unbounded(Value::Num(1.0)).unwrap();
    assert_eq!(ordinary_empty.as_slice(), &[Value::Num(1.0)]);
    let mut default_empty = OutputListStorage::empty();
    default_empty.try_push_unbounded(Value::Num(2.0)).unwrap();
    assert_eq!(default_empty.as_slice(), &[Value::Num(2.0)]);
    let original = CellArray::new(vec![Value::Num(3.0)], 1, 1).unwrap();
    let mut ordinary_alias = original.clone();
    assert_eq!(original.data().as_ptr(), ordinary_alias.data().as_ptr());
    ordinary_alias.push_row_element(Value::Num(4.0)).unwrap();
    assert_eq!(original.data(), &[Value::Num(3.0)]);
    assert_eq!(ordinary_alias.data(), &[Value::Num(3.0), Value::Num(4.0)]);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
}

#[test]
fn sequence_map_preserves_exact_backing_and_refuses_shared_aliases() {
    let bytes = OutputListStorage::requested_bytes(2).unwrap();
    let permit = Permit {
        limit: bytes,
        live: Arc::new(AtomicUsize::new(0)),
    };
    let list = OutputListStorage::try_initialize(2, &permit, |index| {
        Ok::<_, ()>(Value::Num(index as f64 + 1.0))
    })
    .unwrap();
    let alias = list.clone();
    let sequence = ValueSequence::from_output_storage(list).unwrap();
    let mut calls = 0;
    let error = sequence
        .try_map::<ValueSequenceError>(
            |value| {
                calls += 1;
                Ok(value)
            },
            std::convert::identity,
        )
        .unwrap_err();
    assert_eq!(error, ValueSequenceError::SharedOutputStorage);
    assert_eq!(calls, 0);
    assert_eq!(permit.live.load(Ordering::SeqCst), bytes);
    drop(alias);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);

    let list = OutputListStorage::try_initialize(2, &permit, |index| {
        Ok::<_, ()>(Value::Num(index as f64 + 1.0))
    })
    .unwrap();
    let pointer = list.as_slice().as_ptr();
    let sequence = ValueSequence::from_output_storage(list).unwrap();
    let mapped = sequence
        .try_map::<ValueSequenceError>(
            |value| match value {
                Value::Num(number) => Ok(Value::Num(number * 2.0)),
                _ => unreachable!("fixture contains only numeric values"),
            },
            std::convert::identity,
        )
        .unwrap();
    assert_eq!(mapped.as_slice().as_ptr(), pointer);
    assert_eq!(mapped.as_slice(), &[Value::Num(2.0), Value::Num(4.0)]);
    assert_eq!(permit.live.load(Ordering::SeqCst), bytes);
    drop(mapped);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
}
