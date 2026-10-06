//! Callable values belong to their runtime session, not the process-wide scene.
//! The scene retains only an opaque, non-reusable slot and owner identity.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;

use crate::context::RuntimeContextLocalId;
use crate::BuiltinResult;

use super::super::plotting_error;

static NEXT_SLOT: AtomicU64 = AtomicU64::new(1);

runmat_thread_local! {
    // The explicit direct-call owner supports the legacy Rust API when no
    // RuntimeContext is active. Figure teardown removes these roots.
    static DIRECT_CALLABLES: RefCell<HashMap<u64, Value>> = RefCell::new(HashMap::new());
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlotCallableToken {
    slot: u64,
    owner: PlotCallableOwner,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlotCallableOwner {
    Session(RuntimeContextLocalId),
    DirectCall,
}

pub fn retain(value: Value) -> PlotCallableToken {
    let slot = NEXT_SLOT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |slot| {
            slot.checked_add(1)
        })
        .expect("plot callback identity space exhausted");
    if let Some(runtime) = crate::context::legacy::active() {
        runtime.retain_plot_callback(slot, value);
        PlotCallableToken {
            slot,
            owner: PlotCallableOwner::Session(runtime.local_identity()),
        }
    } else {
        DIRECT_CALLABLES.with(|values| {
            values.borrow_mut().insert(slot, value);
        });
        PlotCallableToken {
            slot,
            owner: PlotCallableOwner::DirectCall,
        }
    }
}

pub fn resolve(token: PlotCallableToken, builtin: &'static str) -> BuiltinResult<Value> {
    let active = crate::context::legacy::active();
    let value = match (token.owner, active) {
        (PlotCallableOwner::Session(owner), Some(runtime)) if runtime.local_identity() == owner => {
            runtime.plot_callback(token.slot)
        }
        (PlotCallableOwner::DirectCall, None) => {
            DIRECT_CALLABLES.with(|values| values.borrow().get(&token.slot).cloned())
        }
        _ => None,
    };
    value.ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: function callback belongs to another or ended session"),
        )
    })
}

pub fn release(token: PlotCallableToken) {
    match token.owner {
        PlotCallableOwner::Session(owner) => {
            if let Some(runtime) = crate::context::legacy::active() {
                if runtime.local_identity() == owner {
                    runtime.release_plot_callback(token.slot);
                }
            }
        }
        PlotCallableOwner::DirectCall => {
            DIRECT_CALLABLES.with(|values| {
                values.borrow_mut().remove(&token.slot);
            });
        }
    }
}

#[cfg(test)]
pub fn direct_root_count() -> usize {
    DIRECT_CALLABLES.with(|values| values.borrow().len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    #[test]
    fn callback_is_visible_only_to_its_owner_and_released_with_session() {
        let first = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let token = {
            let _active = first.enter();
            retain(Value::Num(17.0))
        };
        {
            let _active = second.enter();
            assert!(resolve(token, "get").is_err());
        }
        {
            let _active = first.enter();
            assert_eq!(resolve(token, "get").unwrap(), Value::Num(17.0));
        }
        drop(first);
        assert!(resolve(token, "get").is_err());
    }

    #[test]
    fn direct_call_owner_releases_its_callback() {
        let token = retain(Value::Num(3.0));
        assert_eq!(resolve(token, "get").unwrap(), Value::Num(3.0));
        release(token);
        assert!(resolve(token, "get").is_err());
    }
}
