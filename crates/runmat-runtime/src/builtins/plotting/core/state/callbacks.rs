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
    _roots: crate::execution::RootedValueLease,
    // No role is published merely because its property value was stored.
    pub(crate) behavior_authority: Option<std::rc::Rc<crate::context::BackgroundCallbackAuthority>>,
}

#[path = "callbacks/behavior.rs"]
mod behavior;
pub(super) use behavior::{retain_behavior, retain_behavior_in_context};

#[derive(Clone, Debug)]
pub struct PlotCallableToken(Arc<PlotCallableLease>);

struct PlotCallableLease {
    slot: u64,
    owner: PlotCallableOwner,
    origin_thread: std::thread::ThreadId,
    live: Arc<AtomicBool>,
}

impl std::fmt::Debug for PlotCallableLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlotCallableLease")
            .field("slot", &self.slot)
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

impl Drop for PlotCallableLease {
    fn drop(&mut self) {
        // Copied plots share this lease. Only the last plot/snapshot owner may
        // revoke the callable; a sibling's retirement cannot invalidate it.
        self.live.store(false, Ordering::Release);
        if self.owner == PlotCallableOwner::DirectCall
            && std::thread::current().id() == self.origin_thread
        {
            // A finalizer may already be borrowing this TLS map. In that
            // case the revoked entry is pruned at the next owner-lane access
            // or with the thread-local map at thread teardown.
            let _ = DIRECT_CALLABLES.try_with(|values| {
                if let Ok(mut values) = values.try_borrow_mut() {
                    let retired = values.remove(&self.slot);
                    drop(values);
                    drop(retired);
                }
            });
        }
        // Session roots are never touched from an ambient context in Drop.
        // The originating context prunes revoked slots on access and owns
        // their deterministic final retirement at context teardown.
    }
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

pub(crate) fn take_revoked(values: &mut HashMap<u64, PlotCallbackRoot>) -> Vec<PlotCallbackRoot> {
    values
        .extract_if(|_, root| !root.live.load(Ordering::Acquire))
        .map(|(_, root)| root)
        .collect()
}

pub fn retain(value: Value, builtin: &'static str) -> BuiltinResult<PlotCallableToken> {
    retain_root(value, None, builtin)
}

/// Canonical chart ingress supplies the actual session; no ambient owner is
/// substituted when a direct embedding call deliberately supplies none.
pub(crate) fn retain_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<PlotCallableToken> {
    retain_root_in_context(runtime, value, None, builtin)
}

fn retain_root(
    value: Value,
    authority: Option<crate::context::BackgroundCallbackAuthority>,
    builtin: &'static str,
) -> BuiltinResult<PlotCallableToken> {
    retain_root_in_context(
        crate::context::legacy::active().as_ref(),
        value,
        authority,
        builtin,
    )
}

fn retain_root_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    authority: Option<crate::context::BackgroundCallbackAuthority>,
    builtin: &'static str,
) -> BuiltinResult<PlotCallableToken> {
    let slot = reserve_slot(&NEXT_SLOT, builtin)?;
    let live = Arc::new(AtomicBool::new(true));
    let roots = crate::execution::RootedValueLease::for_traced(&value)
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    let root = PlotCallbackRoot {
        value,
        live: Arc::clone(&live),
        _roots: roots,
        behavior_authority: authority.map(std::rc::Rc::new),
    };
    if let Some(runtime) = runtime {
        runtime.retain_plot_callback(slot, root);
        Ok(PlotCallableToken(Arc::new(PlotCallableLease {
            slot,
            owner: PlotCallableOwner::Session(runtime.local_identity()),
            origin_thread: std::thread::current().id(),
            live,
        })))
    } else {
        DIRECT_CALLABLES.with(|values| {
            let (retired, replaced) = {
                let mut values = values.borrow_mut();
                let retired = take_revoked(&mut values);
                let replaced = values.insert(slot, root);
                (retired, replaced)
            };
            drop(replaced);
            drop(retired);
        });
        Ok(PlotCallableToken(Arc::new(PlotCallableLease {
            slot,
            owner: PlotCallableOwner::DirectCall,
            origin_thread: std::thread::current().id(),
            live,
        })))
    }
}

pub fn resolve(token: &PlotCallableToken, builtin: &'static str) -> BuiltinResult<Value> {
    let active = crate::context::legacy::active();
    let value = match (token.0.owner, active) {
        (PlotCallableOwner::Session(owner), Some(runtime)) if runtime.local_identity() == owner => {
            runtime.plot_callback(token.0.slot)
        }
        (PlotCallableOwner::DirectCall, None)
            if std::thread::current().id() == token.0.origin_thread =>
        {
            DIRECT_CALLABLES.with(|values| {
                let (retired, value) = {
                    let mut values = values.borrow_mut();
                    let retired = take_revoked(&mut values);
                    let value = values.get(&token.0.slot).map(|root| root.value.clone());
                    (retired, value)
                };
                drop(retired);
                value
            })
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

#[cfg(test)]
pub fn direct_root_count() -> usize {
    DIRECT_CALLABLES.with(|values| {
        let (retired, count) = {
            let mut values = values.borrow_mut();
            let retired = take_revoked(&mut values);
            (retired, values.len())
        };
        drop(retired);
        count
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
    use runmat_value::{ForeignRef, ForeignResourceKey, ForeignResourceRelease};
    use std::rc::Rc;

    runmat_thread_local! {
        static ON_FOREIGN_RELEASE: RefCell<Option<Box<dyn Fn()>>> = RefCell::new(None);
    }

    #[derive(Debug)]
    struct ReenterRootMap;

    impl ForeignResourceRelease for ReenterRootMap {
        fn release(&self, _: &ForeignResourceKey) {
            let callback = ON_FOREIGN_RELEASE.with(|slot| slot.borrow_mut().take());
            callback.expect("selected reentrant release witness")();
        }
    }

    fn reentrant_value() -> Value {
        Value::Foreign(ForeignRef::managed(
            ForeignResourceKey {
                host_identity: "plot-callback-test".into(),
                handle: 1,
                generation: 1,
            },
            ForeignTypeIdentity {
                family: "test".into(),
                name: "reentrant-plot-callback-root".into(),
                version: 1,
            },
            ForeignOwnership::Shared,
            ForeignAffinity::OriginThread,
            ForeignLifetime::Session,
            Arc::new(ReenterRootMap),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn direct_root_retirement_reenters_after_tls_borrow_ends() {
        let before = direct_root_count();
        let token = retain(reentrant_value(), "fsurf").expect("retain managed value");
        ON_FOREIGN_RELEASE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || assert_eq!(direct_root_count(), before)));
        });
        drop(token);
        ON_FOREIGN_RELEASE.with(|slot| assert!(slot.borrow().is_none()));
    }

    #[test]
    fn session_root_pruning_reenters_after_context_borrow_ends() {
        let runtime = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let token = {
            let _active = runtime.enter();
            retain(reentrant_value(), "fsurf").expect("retain managed value")
        };
        let selected = runtime.clone();
        ON_FOREIGN_RELEASE.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                assert_eq!(selected.plot_callback_root_count(), 0)
            }));
        });
        drop(token);
        assert_eq!(runtime.plot_callback_root_count(), 0);
        ON_FOREIGN_RELEASE.with(|slot| assert!(slot.borrow().is_none()));
    }

    #[test]
    fn callback_is_visible_only_to_its_owner_and_released_with_session() {
        let first = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let token = {
            let _active = first.enter();
            retain_behavior(Value::Num(17.0), "pan").unwrap()
        };
        {
            let _active = second.enter();
            assert!(resolve(&token, "get").is_err());
        }
        {
            let _active = first.enter();
            assert_eq!(resolve(&token, "get").unwrap(), Value::Num(17.0));
        }
        let session = first.session_lease();
        let surviving_copy = token.clone();
        drop(first);
        assert!(session.upgrade().is_none());
        assert!(resolve(&token, "get").is_err());
        assert!(resolve(&surviving_copy, "get").is_err());
    }

    #[test]
    fn copied_chart_tokens_retain_one_direct_callable_until_last_owner_drops() {
        let token = retain(Value::Num(3.0), "fsurf").unwrap();
        assert_eq!(resolve(&token, "get").unwrap(), Value::Num(3.0));
        let copy = token.clone();
        drop(token);
        assert_eq!(resolve(&copy, "get").unwrap(), Value::Num(3.0));
        assert_eq!(direct_root_count(), 1);
        drop(copy);
        assert_eq!(direct_root_count(), 0);
    }

    #[test]
    fn foreign_last_owner_drop_revokes_and_origin_session_prunes_its_root() {
        let first = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let token = {
            let _active = first.enter();
            retain(Value::Num(19.0), "fsurf").unwrap()
        };
        let shared_live = Arc::clone(&token.0.live);
        let copy = token.clone();
        assert_eq!(first.plot_callback_root_count(), 1);
        {
            let _active = second.enter();
            drop(token);
        }
        {
            let _active = first.enter();
            assert_eq!(resolve(&copy, "get").unwrap(), Value::Num(19.0));
        }
        {
            let _active = second.enter();
            drop(copy);
        }
        assert!(!shared_live.load(Ordering::Acquire));
        assert_eq!(first.plot_callback_root_count(), 0);
    }

    #[test]
    fn foreign_thread_direct_last_drop_only_revokes_origin_tls_root() {
        let token = retain(Value::Num(23.0), "fsurf").unwrap();
        let shared_live = Arc::clone(&token.0.live);
        assert_eq!(direct_root_count(), 1);
        std::thread::spawn(move || drop(token))
            .join()
            .expect("foreign last drop");
        assert!(!shared_live.load(Ordering::Acquire));
        assert_eq!(direct_root_count(), 0);
    }

    #[test]
    fn exhausted_slot_space_is_a_typed_error_without_consuming_identity() {
        let counter = AtomicU64::new(u64::MAX);
        let error = reserve_slot(&counter, "fsurf").expect_err("no wrapping callback slot");
        assert!(error.message().contains("identity space exhausted"));
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }
}
