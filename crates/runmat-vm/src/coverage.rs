use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

type SharedCounters = Rc<RefCell<BTreeMap<u64, u64>>>;

runmat_thread_local::runmat_thread_local! {
    static ACTIVE: RefCell<Option<SharedCounters>> = const { RefCell::new(None) };
}

/// Thread-confined coverage counters shared by nested interpreter and JIT calls
/// for one exact executable invocation.
#[derive(Debug)]
pub struct CoverageSession {
    counters: SharedCounters,
    previous: Option<SharedCounters>,
}

impl CoverageSession {
    pub fn start() -> Self {
        let counters = Rc::new(RefCell::new(BTreeMap::new()));
        let previous = ACTIVE.with(|active| active.replace(Some(counters.clone())));
        Self { counters, previous }
    }

    pub fn counts(&self) -> BTreeMap<u64, u64> {
        self.counters.borrow().clone()
    }
}

impl Drop for CoverageSession {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.replace(self.previous.take());
        });
    }
}

#[inline]
pub fn hit_sites(sites: &[u64]) {
    if sites.is_empty() {
        return;
    }
    ACTIVE.with(|active| {
        let Some(counters) = active.borrow().as_ref().cloned() else {
            return;
        };
        let mut counters = counters.borrow_mut();
        for site in sites {
            let counter = counters.entry(*site).or_default();
            *counter = counter.saturating_add(1);
        }
    });
}

/// Stable host callback imported by Turbine-generated native code.
pub extern "C" fn hit_site_from_jit(site: u64) {
    hit_sites(&[site]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_sessions_restore_the_outer_collector() {
        let outer = CoverageSession::start();
        hit_sites(&[0]);
        {
            let inner = CoverageSession::start();
            hit_sites(&[1]);
            assert_eq!(inner.counts(), BTreeMap::from([(1, 1)]));
        }
        hit_sites(&[0]);
        assert_eq!(outer.counts(), BTreeMap::from([(0, 2)]));
    }
}
