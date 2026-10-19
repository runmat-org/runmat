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
    let text = Utf8Text::new(source).try_admit_exact(&permit).unwrap();
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
    let callable = crate::Value::function_handle(callable_name);
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
