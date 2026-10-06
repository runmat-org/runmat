//! Callable values belong to their runtime session, not the process-wide scene.
//! The scene retains only an opaque, non-reusable slot and owner identity.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;

use crate::context::RuntimeContextLocalId;
use crate::BuiltinResult;

use super::super::plotting_error;

static NEXT_SLOT: AtomicU64 = AtomicU64::new(1);

runmat_thread_local! {
    // The explicit direct-call owner supports the legacy Rust API when no
    // RuntimeContext is active. Figure teardown removes these roots.
    static DIRECT_CALLABLES: RefCell<HashMap<u64, PlotCallbackRoot>> = RefCell::new(HashMap::new());
}

#[derive(Debug)]
pub(crate) struct PlotCallbackRoot {
    pub(crate) value: Value,
    pub(crate) live: Arc<AtomicBool>,
}

#[derive(Clone, Debug)]
pub struct PlotCallableToken {
    slot: u64,
    owner: PlotCallableOwner,
    live: Arc<AtomicBool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlotCallableOwner {
    Session(RuntimeContextLocalId),
    DirectCall,
}

fn reserve_slot(counter: &AtomicU64, builtin: &'static str) -> BuiltinResult<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |slot| {
            slot.checked_add(1)
        })
        .map_err(|_| {
            plotting_error(
                builtin,
                format!("{builtin}: plot callback identity space exhausted"),
            )
        })
}

fn prune_revoked(values: &mut HashMap<u64, PlotCallbackRoot>) {
    values.retain(|_, root| root.live.load(Ordering::Acquire));
}

pub fn retain(value: Value, builtin: &'static str) -> BuiltinResult<PlotCallableToken> {
    let slot = reserve_slot(&NEXT_SLOT, builtin)?;
    let live = Arc::new(AtomicBool::new(true));
    let root = PlotCallbackRoot {
        value,
        live: Arc::clone(&live),
    };
    if let Some(runtime) = crate::context::legacy::active() {
        runtime.retain_plot_callback(slot, root);
        Ok(PlotCallableToken {
            slot,
            owner: PlotCallableOwner::Session(runtime.local_identity()),
            live,
        })
    } else {
        DIRECT_CALLABLES.with(|values| {
            let mut values = values.borrow_mut();
            prune_revoked(&mut values);
            values.insert(slot, root);
        });
        Ok(PlotCallableToken {
            slot,
            owner: PlotCallableOwner::DirectCall,
            live,
        })
    }
}

pub fn resolve(token: &PlotCallableToken, builtin: &'static str) -> BuiltinResult<Value> {
    let active = crate::context::legacy::active();
    let value = match (token.owner, active) {
        (PlotCallableOwner::Session(owner), Some(runtime)) if runtime.local_identity() == owner => {
            runtime.plot_callback(token.slot)
        }
        (PlotCallableOwner::DirectCall, None) => DIRECT_CALLABLES.with(|values| {
            let mut values = values.borrow_mut();
            prune_revoked(&mut values);
            values.get(&token.slot).map(|root| root.value.clone())
        }),
        _ => None,
    };
    value.ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: function callback belongs to another or ended session"),
        )
    })
}

pub fn release(token: &PlotCallableToken) {
    token.live.store(false, Ordering::Release);
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
    DIRECT_CALLABLES.with(|values| {
        let mut values = values.borrow_mut();
        prune_revoked(&mut values);
        values.len()
    })
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
            retain(Value::Num(17.0), "fsurf").unwrap()
        };
        {
            let _active = second.enter();
            assert!(resolve(&token, "get").is_err());
        }
        {
            let _active = first.enter();
            assert_eq!(resolve(&token, "get").unwrap(), Value::Num(17.0));
        }
        drop(first);
        assert!(resolve(&token, "get").is_err());
    }

    #[test]
    fn direct_call_owner_releases_its_callback() {
        let token = retain(Value::Num(3.0), "fsurf").unwrap();
        assert_eq!(resolve(&token, "get").unwrap(), Value::Num(3.0));
        release(&token);
        assert!(resolve(&token, "get").is_err());
    }

    #[test]
    fn sibling_scene_deletion_revokes_then_owner_prunes_its_root() {
        let first = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let token = {
            let _active = first.enter();
            retain(Value::Num(19.0), "fsurf").unwrap()
        };
        assert_eq!(first.plot_callback_root_count(), 1);
        {
            let _active = second.enter();
            release(&token);
        }
        assert!(!token.live.load(Ordering::Acquire));
        {
            let _active = first.enter();
            assert!(resolve(&token, "get").is_err());
        }
        assert_eq!(first.plot_callback_root_count(), 0);
    }

    #[test]
    fn exhausted_slot_space_is_a_typed_error_without_consuming_identity() {
        let counter = AtomicU64::new(u64::MAX);
        let error = reserve_slot(&counter, "fsurf").expect_err("no wrapping callback slot");
        assert!(error.message().contains("identity space exhausted"));
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }
}
