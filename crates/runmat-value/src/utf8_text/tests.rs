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
fn exact_text_and_callable_generation_preserve_physical_owners() {
    let mut source = String::with_capacity(31);
    source.push_str("callback");
    let pointer = source.as_ptr();
    let requested = Utf8Text::requested_bytes_for_capacity(source.capacity()).unwrap();
    let live = Arc::new(AtomicUsize::new(0));
    let permit = Permit {
        limit: requested,
        live: live.clone(),
    };
    let text = Utf8Text::new(source);
    let original_alias = text.clone();
    assert_eq!(original_alias.as_str().as_ptr(), pointer);
    assert!(!original_alias.has_retained_claim());
    let text = text.try_admit_exact(&permit).unwrap();
    assert!(original_alias.has_retained_claim());
    drop(text.clone().try_admit_exact(&permit).unwrap());
    assert_eq!(live.load(Ordering::SeqCst), requested);
    drop(original_alias);
    assert_eq!(text.as_str().as_ptr(), pointer);
    assert!(text == "callback");
    assert_eq!(String::from("callback"), text);
    assert_eq!(
        text.cmp(&Utf8Text::new("callback".to_owned())),
        std::cmp::Ordering::Equal
    );
    assert!(Utf8Text::default() < text);
    let mut indexed = std::collections::HashSet::new();
    indexed.insert(text.clone());
    assert!(indexed.contains("callback"));
    drop(indexed);
    let mut ordered = std::collections::BTreeSet::new();
    ordered.insert(text.clone());
    assert!(ordered.contains("callback"));
    drop(ordered);
    assert_eq!(live.load(Ordering::SeqCst), requested);
    let alias = text.clone();
    drop(text);
    assert_eq!(alias.as_str().as_ptr(), pointer);
    drop(alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let mut adopted = String::with_capacity(37);
    adopted.push_str("member");
    let pointer = adopted.as_ptr();
    let bytes = Utf8Text::requested_bytes_for_capacity(adopted.capacity()).unwrap();
    let adopted = Utf8Text::try_adopt_string_exact(
        adopted,
        &Permit {
            limit: bytes,
            live: live.clone(),
        },
    )
    .unwrap();
    assert_eq!(adopted.as_str().as_ptr(), pointer);
    assert_eq!(live.load(Ordering::SeqCst), bytes);
    drop(adopted);
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let denied = String::from("member");
    let pointer = denied.as_ptr();
    let denied = Utf8Text::try_adopt_string_exact(
        denied,
        &Permit {
            limit: 0,
            live: live.clone(),
        },
    )
    .unwrap_err()
    .try_into_unclaimed_values()
    .unwrap();
    assert_eq!(denied.as_ptr(), pointer);
    assert_eq!(denied, b"member");
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let source = "Aé𝄞";
    let staged_bytes = Utf8Text::requested_bytes_for_capacity(source.len()).unwrap();
    let staged_permit = Permit {
        limit: staged_bytes,
        live: live.clone(),
    };
    let mut staged = Utf8Text::try_begin_copy_exact(source, &staged_permit).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), staged_bytes);
    assert_eq!(staged.advance(std::num::NonZeroUsize::new(2).unwrap()), 2);
    assert_eq!(staged.copied_bytes(), 2); // inside the UTF-8 encoding of é
    let mut staged = staged.try_finish().unwrap_err();
    assert_eq!(staged.advance(std::num::NonZeroUsize::new(2).unwrap()), 2);
    assert!(!staged.is_complete());
    assert_eq!(staged.advance(std::num::NonZeroUsize::new(64).unwrap()), 3);
    let staged = staged.try_finish().unwrap();
    assert_eq!(staged.as_str(), source);
    assert!(staged.has_retained_claim());
    let alias = staged.clone();
    drop(staged);
    assert_eq!(live.load(Ordering::SeqCst), staged_bytes);
    drop(alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let mut cancelled = Utf8Text::try_begin_copy_exact(source, &staged_permit).unwrap();
    cancelled.advance(std::num::NonZeroUsize::new(1).unwrap());
    drop(cancelled);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let checkpoints = std::cell::RefCell::new(Vec::new());
    let large = "é".repeat(4097);
    let bytes = Utf8Text::requested_bytes_for_capacity(large.len()).unwrap();
    let permit = Permit {
        limit: bytes,
        live: live.clone(),
    };
    let copied = Utf8Text::try_copy_str_exact_with(&large, &permit, |offset| {
        checkpoints.borrow_mut().push(offset);
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(copied.as_str(), large);
    assert_eq!(*checkpoints.borrow(), vec![0, 4096, 8192]);
    drop(copied);
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert!(matches!(
        Utf8Text::try_copy_str_exact_with(&large, &permit, |_| Err::<(), _>("stop")),
        Err(crate::ExactArrayInitializationError::Element("stop"))
    ));
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let mut denied = String::with_capacity(19);
    denied.push_str("retained");
    let denied_pointer = denied.as_ptr();
    let denied = Utf8Text::new(denied)
        .try_admit_exact(&Permit {
            limit: 0,
            live: live.clone(),
        })
        .unwrap_err()
        .try_into_unclaimed_text()
        .unwrap();
    assert_eq!(denied.as_str().as_ptr(), denied_pointer);

    let mut callable_name = String::with_capacity(23);
    callable_name.push_str("solver_step");
    let callable_pointer = callable_name.as_ptr();
    let callable_name_bytes =
        Utf8Text::requested_bytes_for_capacity(callable_name.capacity()).unwrap();
    let callable = crate::Value::function_handle(callable_name);
    let callable_name_alias = callable.clone();
    let quote = crate::SharedValueGeneration::callable_quote(&callable)
        .unwrap()
        .unwrap();
    let callable_permit = Permit {
        limit: quote.requested_bytes,
        live: live.clone(),
    };
    let mut callable = crate::CallableGenerationInput::new(callable);
    let generation =
        crate::SharedValueGeneration::prepare_callable(&mut callable, &callable_permit).unwrap();
    let crate::Value::FunctionHandle(handle) = generation.value() else {
        panic!("callable generation changed its Value kind")
    };
    assert_eq!(handle.name().as_ptr(), callable_pointer);
    let generation_alias = generation.clone();
    drop(generation);
    assert_eq!(live.load(Ordering::SeqCst), quote.requested_bytes);
    drop(generation_alias);
    assert_eq!(live.load(Ordering::SeqCst), callable_name_bytes);
    drop(callable_name_alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    // Both callable metadata strings belong to their original shared owners,
    // not to the temporary generation carrying this callable.
    let closure = crate::Closure::new(
        "retained_closure".to_owned(),
        None,
        runmat_types::FunctionSourceRole::Unknown,
        Some("source.m".to_owned()),
        Vec::new(),
        None,
    );
    let name_pointer = closure.function_name.as_str().as_ptr();
    let source_pointer = closure.source_file.as_ref().unwrap().as_str().as_ptr();
    let detached = closure.clone().into_data();
    assert_eq!(detached.function_name.as_str().as_ptr(), name_pointer);
    assert_eq!(
        detached.source_file.as_ref().unwrap().as_str().as_ptr(),
        source_pointer
    );
    let callable = crate::Value::Closure(closure);
    let quote = crate::SharedValueGeneration::callable_quote(&callable)
        .unwrap()
        .unwrap();
    let metadata_bytes = quote.requested_bytes
        - crate::SharedValueGeneration::requested_bytes()
        - crate::Closure::resolved_header_bytes();
    let permit = Permit {
        limit: quote.requested_bytes,
        live: live.clone(),
    };
    let mut input = crate::CallableGenerationInput::new(callable);
    let generation = crate::SharedValueGeneration::prepare_callable(&mut input, &permit).unwrap();
    assert!(detached.function_name.has_retained_claim());
    assert!(detached.source_file.as_ref().unwrap().has_retained_claim());
    drop(generation);
    assert_eq!(live.load(Ordering::SeqCst), metadata_bytes);
    drop(detached);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let closure = crate::Closure::new(
        "resolve_alias".to_owned(),
        None,
        runmat_types::FunctionSourceRole::Unknown,
        None,
        Vec::new(),
        None,
    );
    let original = closure.clone();
    let binding = crate::BoundFunctionBinding {
        function: 7,
        authority: crate::BoundFunctionAuthority::unavailable(7),
    };
    let bytes = crate::Closure::resolved_header_bytes();
    let denied = Permit {
        limit: bytes - 1,
        live: live.clone(),
    };
    assert!(matches!(
        closure.try_bind_resolved(
            binding.clone(),
            runmat_types::FunctionSourceRole::Unknown,
            None,
            &denied
        ),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert!(original.binding.is_none());
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let permit = Permit {
        limit: bytes,
        live: live.clone(),
    };
    let resolved = closure
        .try_bind_resolved(
            binding,
            runmat_types::FunctionSourceRole::Unknown,
            None,
            &permit,
        )
        .unwrap();
    assert_ne!(original.instance_id(), resolved.instance_id());
    assert!(original.binding.is_none());
    assert_eq!(resolved.bound_function(), Some(7));
    assert_eq!(
        original.function_name.as_str().as_ptr(),
        resolved.function_name.as_str().as_ptr()
    );
    let alias = resolved.clone();
    drop(resolved);
    assert_eq!(live.load(Ordering::SeqCst), bytes);
    drop(alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    // A future's output contract owns nested vectors. Cloning the Value must
    // retain those exact original pointers, including spare physical capacity.
    let mut shape = Vec::with_capacity(11);
    shape.push(1);
    let shape_pointer = shape.as_ptr();
    let future = crate::SharedFutureHandle::new(runmat_execution::FutureHandle {
        id: runmat_execution::FutureId::derive(&[b"shared-future"]),
        scope_id: runmat_execution::ExecutionScopeId::derive(&[b"shared-scope"]),
        outputs: runmat_execution::OutputContract {
            requested_outputs: 1,
            owned_work: Some(runmat_execution::task::OwnedBuiltinResponseContract {
                dense: vec![runmat_execution::task::OwnedBuiltinDenseOutputContract {
                    shape,
                    allowed_element_types: vec![runmat_execution::value::ElementType::F64],
                }],
                max_wire_bytes: runmat_execution::task::OWNED_BUILTIN_FAILURE_WIRE_BYTES + 64,
            }),
        },
    });
    let alias = future.clone();
    assert!(future.shares_allocation_with(&alias));
    assert_eq!(
        alias.outputs.owned_work.as_ref().unwrap().dense[0]
            .shape
            .as_ptr(),
        shape_pointer
    );
    let bytes = future.metadata_unclaimed_bytes().unwrap().unwrap();
    let denied = Permit {
        limit: bytes - 1,
        live: live.clone(),
    };
    assert!(matches!(
        future.try_promote_metadata(&denied),
        Err(crate::ExecutionMetadataQuoteError::Allocation(
            ExactHostNumericAllocationError::ExceedsPermit
        ))
    ));
    assert!(!alias.has_retained_metadata_claim());
    let permit = Permit {
        limit: bytes,
        live: live.clone(),
    };
    future.try_promote_metadata(&permit).unwrap();
    assert!(alias.has_retained_metadata_claim());
    alias.try_promote_metadata(&permit).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), bytes);
    drop(future);
    assert_eq!(live.load(Ordering::SeqCst), bytes);
    drop(alias);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let characters = ['A', 'é', '𝄞'];
    let (byte_len, quoted) =
        Utf8Text::requested_bytes_for_chars_with(&characters, |_| Ok::<(), ()>(())).unwrap();
    assert_eq!(byte_len, "Aé𝄞".len());
    let unicode_permit = Permit {
        limit: quoted,
        live: live.clone(),
    };
    let unicode =
        Utf8Text::try_copy_chars_exact_with(&characters, &unicode_permit, |_| Ok::<(), ()>(()))
            .unwrap();
    assert_eq!(unicode.as_str(), "Aé𝄞");
    assert_eq!(live.load(Ordering::SeqCst), quoted);
    drop(unicode);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let denied = Utf8Text::try_copy_chars_exact_with(
        &characters,
        &Permit {
            limit: quoted - 1,
            live: live.clone(),
        },
        |_| Ok::<(), ()>(()),
    );
    assert!(matches!(
        denied,
        Err(crate::ExactArrayInitializationError::Allocation(
            ExactHostNumericAllocationError::ExceedsPermit
        ))
    ));
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let cancelled = Utf8Text::try_copy_chars_exact_with(&characters, &unicode_permit, |_| Err(()));
    assert!(matches!(
        cancelled,
        Err(crate::ExactArrayInitializationError::Element(()))
    ));
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_utf16_copy_decodes_pairs_and_rejects_unpaired_units_before_allocation() {
    let live = Arc::new(AtomicUsize::new(0));
    let units = crate::encode_utf16_code_units("A🙂é");
    let (length, quoted) =
        Utf8Text::requested_bytes_for_utf16_with(&units, |_| Ok::<(), ()>(())).unwrap();
    assert_eq!(length, "A🙂é".len());
    let permit = Permit {
        limit: quoted,
        live: live.clone(),
    };
    let output =
        Utf8Text::try_copy_utf16_exact_with(&units, &permit, |_| Ok::<(), ()>(())).unwrap();
    assert_eq!(output.as_str(), "A🙂é");
    assert_eq!(live.load(Ordering::SeqCst), quoted);
    drop(output);
    assert_eq!(live.load(Ordering::SeqCst), 0);

    let malformed = [crate::Utf16CodeUnit::from_raw(0xD83D)];
    let error =
        Utf8Text::requested_bytes_for_utf16_with(&malformed, |_| Ok::<(), ()>(())).unwrap_err();
    assert!(matches!(
        error,
        crate::ExactArrayInitializationError::Element(crate::Utf16TextCopyError::Decode(
            crate::Utf16DecodeError { index: 0, .. }
        ))
    ));
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_utf16_quote_checkpoints_mixed_unit_widths() {
    let mut units = vec![crate::Utf16CodeUnit::from_raw(u16::from(b'A'))];
    units.extend(crate::encode_utf16_code_units(&"🙂".repeat(5_000)));
    let mut visited = Vec::new();
    Utf8Text::requested_bytes_for_utf16_with(&units, |offset| {
        visited.push(offset);
        Ok::<(), ()>(())
    })
    .unwrap();
    assert_eq!(visited.first(), Some(&0));
    assert_eq!(visited.last(), Some(&units.len()));
    assert!(
        visited.len() >= 4,
        "mixed-width quote skipped checkpoints: {visited:?}"
    );
    assert!(visited.windows(2).all(|pair| pair[1] - pair[0] <= 4_097));
}
