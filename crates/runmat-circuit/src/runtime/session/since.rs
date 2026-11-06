//! Recompiles for a client that holds an earlier projection: what changed since it.
//!
//! Each sync that may change the projection numbers it with the next generation, and a sync
//! that patched the board (or recompiled it whole over a board it held) keeps the delta from the
//! generation before ([`super::super::delta`]). A client that names the generation it holds
//! gets that delta, nothing when it is current, and the whole projection otherwise. The session
//! keeps no copy of the projection for this: a patch keeps only the rows it rebuilt.

use super::*;

/// What a client holding projection `since` needs.
#[derive(Debug, PartialEq)]
pub enum Since<'a> {
    /// The projection is the one it holds.
    Same,
    /// The change from the projection it holds.
    Delta(&'a serde_json::Value),
    /// The session can't say what changed: send the projection whole.
    Whole,
}

impl Session {
    /// The current projection's number.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// What a client holding projection number `since` needs to hold the current one.
    pub fn since(&self, since: u64) -> Since<'_> {
        if since == self.generation {
            Since::Same
        } else if since + 1 == self.generation {
            self.last_delta.as_ref().map_or(Since::Whole, Since::Delta)
        } else {
            Since::Whole
        }
    }

    /// [`compile`](Self::compile) for a client holding projection `since`, written as JSON:
    /// `{ seq, diagnostics, delta }` when it can be patched, else `{ seq, diagnostics, index? }`.
    /// `seq` is the new projection's number, for the next call.
    pub fn compile_since(
        &mut self,
        path: &str,
        text: &str,
        files: &Files,
        options: &CompileOptions,
        since: Option<u64>,
    ) -> String {
        let mut pending = String::new();
        let diagnostics = match self.sync(path, text, files, options) {
            Err(d) => {
                self.board = None;
                if let Some(j) = super::super::values::pending_json(&d, path, text, files, &options.values) {
                    pending = format!(",\"pending_values\":{j}");
                }
                d
            }
            Ok(()) => self.board.as_ref().map(|b| b.rule_errors.clone()).unwrap_or_default(),
        };
        let seq = self.generation;
        let diagnostics = serde_json::to_string(&diagnostics).unwrap_or_else(|_| "[]".into());
        // Written straight to text: the whole projection is the large case.
        let body = match (self.index(), since.map(|s| self.since(s))) {
            (None, _) => String::new(),
            (Some(_), Some(Since::Same)) => ",\"delta\":{\"fields\":{}}".to_owned(),
            (Some(_), Some(Since::Delta(d))) => format!(",\"delta\":{d}"),
            (Some(index), _) => format!(",\"index\":{}", serde_json::to_string(index).unwrap_or_else(|_| "null".into())),
        };
        format!("{{\"seq\":{seq},\"diagnostics\":{diagnostics}{body}{pending}}}")
    }
}
