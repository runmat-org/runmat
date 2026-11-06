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
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// The error a call fails with while its result is still to be evaluated.
pub const PENDING: &str = "waiting for the RunMat runtime to evaluate this value";

/// One call the table couldn't answer: what the host runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingCall {
    /// The [`call_key`] the result is stored under.
    pub key: String,
    /// The function, defined in `sources[0]`.
    pub function: String,
    /// `(path, text)` of the function's file, then the `.m` files it can call without a `use`.
    pub sources: Vec<(String, String)>,
    /// SI doubles, in the order the `function` line declares them.
    pub arguments: Vec<f64>,
}

/// A result the host evaluated: the result's elements, or the error the function raised.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TableResult {
    Values(Vec<f64>),
    Error { error: String },
}

/// The key of one call: the sources' paths and texts, the function and the arguments bit for
/// bit. Editing any reached `.m` file, or changing an argument, gives a new key.
pub fn call_key(sources: &[(String, String)], function: &str, arguments: &[f64]) -> String {
    let mut h = Sha256::new();
    let mut part = |b: &[u8]| {
        h.update((b.len() as u64).to_be_bytes());
        h.update(b);
    };
    part(b"runmat-cad/m-call/v1");
    part(function.as_bytes());
    for (path, text) in sources {
        part(path.as_bytes());
        part(text.as_bytes());
    }
    for a in arguments {
        part(&a.to_bits().to_be_bytes());
    }
    hex::encode(h.finalize())
}

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
