//! `.m` design values answered from a table, for hosts that run `.m` code outside the compile.
//!
//! The Desktop and the browser compile designs in the CAD host and run `.m` code in the RunMat
//! runtime worker, so a compile can't call the runtime while it runs. They compile in two
//! phases instead:
//! 1. The compile runs with a [`TableRunner`] holding the results evaluated so far. Each call the
//!    table can't answer is recorded as a [`PendingCall`] and fails with [`PENDING`].
//! 2. The host runs the pending calls in the runtime, adds the results under their
//!    [`call_key`]s, and compiles again. A value whose arguments come from another call's result
//!    needs one more round, so the host repeats until nothing is pending.

use super::m_functions::MRunner;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

pub use runmat_design_syntax::m_calls::{call_key, PendingCall, TableResult, PENDING};

/// Answers `.m` calls from evaluated results and records the calls it can't answer.
pub struct TableRunner {
    results: BTreeMap<String, TableResult>,
    pending: Rc<RefCell<Vec<PendingCall>>>,
    runtime_version: String,
}

impl TableRunner {
    /// A runner over `results` (by [`call_key`]); `runtime_version` names the runtime that
    /// evaluated them. The returned list fills with the calls still to evaluate.
    pub fn new(
        results: BTreeMap<String, TableResult>,
        runtime_version: impl Into<String>,
    ) -> (Self, Rc<RefCell<Vec<PendingCall>>>) {
        let pending = Rc::new(RefCell::new(Vec::new()));
        let runner = TableRunner {
            results,
            pending: pending.clone(),
            runtime_version: runtime_version.into(),
        };
        (runner, pending)
    }
}

impl MRunner for TableRunner {
    fn call(&self, sources: &[(String, String)], function: &str, arguments: &[f64]) -> Result<Vec<f64>, String> {
        let key = call_key(sources, function, arguments);
        match self.results.get(&key) {
            Some(TableResult::Values(v)) => Ok(v.clone()),
            Some(TableResult::Error { error }) => Err(error.clone()),
            None => {
                let mut pending = self.pending.borrow_mut();
                if !pending.iter().any(|p| p.key == key) {
                    pending.push(PendingCall {
                        key,
                        function: function.to_owned(),
                        sources: sources.to_vec(),
                        arguments: arguments.to_vec(),
                    });
                }
                Err(PENDING.into())
            }
        }
    }

    fn runtime_version(&self) -> String {
        self.runtime_version.clone()
    }
}
