use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

type SharedCounters = Rc<RefCell<BTreeMap<u64, u64>>>;

runmat_thread_local::runmat_thread_local! {
    static ACTIVE: RefCell<HashMap<runmat_execution::ExecutionScopeId, Vec<SharedCounters>>> =
        RefCell::new(HashMap::new());
}

/// Scope-owned backend-neutral coverage counters for one exact executable
/// invocation. VM, native, and browser-capable executors report the same stable
/// counter keys through this runtime boundary.
#[derive(Debug)]
pub struct CoverageSession {
    scope_id: runmat_execution::ExecutionScopeId,
    counters: SharedCounters,
}

impl CoverageSession {
    pub fn start(runtime: &crate::context::RuntimeContext) -> Self {
        let scope_id = runtime.execution().scope_id();
        let counters = Rc::new(RefCell::new(BTreeMap::new()));
        ACTIVE.with(|active| {
            active
                .borrow_mut()
                .entry(scope_id)
                .or_default()
                .push(counters.clone());
        });
        Self { scope_id, counters }
    }

    pub fn counts(&self) -> BTreeMap<u64, u64> {
        self.counters.borrow().clone()
    }
}

impl Drop for CoverageSession {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            let remove_scope = if let Some(stack) = active.get_mut(&self.scope_id) {
                let popped = stack.pop();
                debug_assert!(popped
                    .as_ref()
                    .is_some_and(|counters| Rc::ptr_eq(counters, &self.counters)));
                stack.is_empty()
            } else {
                debug_assert!(false, "coverage session scope missing during cleanup");
                false
            };
            if remove_scope {
                active.remove(&self.scope_id);
            }
        });
    }
}

#[inline]
pub fn hit_sites(sites: &[u64]) {
    let Some(runtime) = crate::context::legacy::active() else {
        return;
    };
    hit_sites_in(&runtime, sites);
}

#[inline]
pub fn hit_sites_in(runtime: &crate::context::RuntimeContext, sites: &[u64]) {
    if sites.is_empty() {
        return;
    }
    let scope_id = runtime.execution().scope_id();
    ACTIVE.with(|active| {
        let counters = active
            .borrow()
            .get(&scope_id)
            .and_then(|stack| stack.last().cloned());
        let Some(counters) = counters else { return };
        let mut counters = counters.borrow_mut();
        for site in sites {
            let counter = counters.entry(*site).or_default();
            *counter = counter.saturating_add(1);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
    }

    #[test]
    fn nested_sessions_restore_the_outer_collector() {
        let runtime = runtime();
        futures::executor::block_on(runtime.scope(async {
            let outer = CoverageSession::start(&runtime);
            hit_sites(&[0]);
            {
                let inner = CoverageSession::start(&runtime);
                hit_sites(&[1]);
                assert_eq!(inner.counts(), BTreeMap::from([(1, 1)]));
            }
            hit_sites(&[0]);
            assert_eq!(outer.counts(), BTreeMap::from([(0, 2)]));
        }));
    }

    #[test]
    fn independently_scoped_collectors_do_not_share_hits() {
        let first = runtime();
        let second = runtime();
        let first_coverage = CoverageSession::start(&first);
        let second_coverage = CoverageSession::start(&second);
        hit_sites_in(&first, &[1]);
        hit_sites_in(&second, &[2]);
        assert_eq!(first_coverage.counts(), BTreeMap::from([(1, 1)]));
        assert_eq!(second_coverage.counts(), BTreeMap::from([(2, 1)]));
    }
}
