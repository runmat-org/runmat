//! The host's `.m` runner for design values (report 12 §2.6, the value phase).
//!
//! A part regenerates with [`runmat_cad::value::m_functions::MFunctions`] when a host is set
//! for the thread: the workspace files (the `.m` files and the design files that declare the
//! folder scope) and the runner that executes `.m` code. Without one, a value that calls a
//! function fails with "no dynamic value evaluator is configured".

use runmat_cad::modules::ModuleScope;
use runmat_cad::value::m_functions::{MFunctions, MRunner};
use runmat_cad::value::{DynamicValueEvaluator, NoDynamicValues};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// What a host lends the regeneration: its workspace files and its `.m` runner.
pub struct ValueHost {
    /// Workspace path → text, with paths spelled as the regenerated documents' paths are.
    pub files: BTreeMap<String, String>,
    pub runner: Box<dyn MRunner>,
}

thread_local! {
    static HOST: RefCell<Option<Rc<ValueHost>>> = const { RefCell::new(None) };
}

/// Runs `f` with `host` serving `.m` calls on this thread, then restores the previous host.
pub fn with_value_host<R>(host: ValueHost, f: impl FnOnce() -> R) -> R {
    let previous = HOST.with(|h| h.replace(Some(Rc::new(host))));
    struct Restore(Option<Rc<ValueHost>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            HOST.with(|h| *h.borrow_mut() = previous);
        }
    }
    let _restore = Restore(previous);
    f()
}

/// Runs `calls` through the thread's host runner: each call's result (or the error it raised)
/// by its key, for a compile that lists its calls instead of running them (a circuit's
/// `pending_values`). `None` without a host.
pub fn run_calls(
    calls: &[runmat_design_syntax::m_calls::PendingCall],
) -> Option<BTreeMap<String, runmat_design_syntax::m_calls::TableResult>> {
    use runmat_design_syntax::m_calls::TableResult;
    let host = HOST.with(|h| h.borrow().clone())?;
    Some(
        calls
            .iter()
            .map(|call| {
                let result = match host.runner.call(&call.sources, &call.function, &call.arguments) {
                    Ok(values) => TableResult::Values(values),
                    Err(error) => TableResult::Error { error },
                };
                (call.key.clone(), result)
            })
            .collect(),
    )
}

/// The evaluator for the document at `path`: `.m` calls through the thread's host, if any.
pub(crate) fn with_evaluator<R>(path: &str, f: impl FnOnce(&dyn DynamicValueEvaluator) -> R) -> R {
    match HOST.with(|h| h.borrow().clone()) {
        Some(host) => {
            let scope = ModuleScope::from_files(host.files.iter().map(|(p, t)| (p.as_str(), t.as_str())));
            let values = MFunctions::new(&scope, &host.files, path, host.runner.as_ref());
            f(&values)
        }
        None => f(&NoDynamicValues),
    }
}

/// Fills each value's `file`: the `.m` file its function resolves to from `path`, through the
/// thread's host. Without a host the files stay unknown.
pub fn resolve_let_files(path: &str, lets: &mut [runmat_cad::index::LetRow]) {
    let Some(host) = HOST.with(|h| h.borrow().clone()) else { return };
    if lets.iter().all(|l| l.function.is_none()) {
        return;
    }
    let scope = ModuleScope::from_files(host.files.iter().map(|(p, t)| (p.as_str(), t.as_str())));
    let values = MFunctions::new(&scope, &host.files, path, host.runner.as_ref());
    for row in lets.iter_mut() {
        if let Some(function) = &row.function {
            row.file = values.resolve(function).ok().map(|f| f.file);
        }
    }
}

/// The Desktop's and the browser's compile entry (`options`: `{ configuration?, values?, known? }`):
/// the first phase of the two-phase `.m` compile ([`runmat_cad::value::m_table`]).
/// `values` holds the results evaluated so far by call key. The output is the compile JSON
/// plus `pending_values`, the calls still to run in the runtime, when there are any.
pub fn compile_json_with_values(path: &str, text: &str, files: &super::Files, options: &str) -> String {
    use runmat_cad::value::m_table::{TableResult, TableRunner};
    let configuration = super::compile_options(options);
    let parsed: serde_json::Value = serde_json::from_str(options).unwrap_or_default();
    let results: BTreeMap<String, TableResult> = parsed
        .get("values")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let version = parsed
        .get("runtime_version")
        .and_then(|v| v.as_str())
        .unwrap_or("runtime")
        .to_owned();
    let (runner, pending) = TableRunner::new(results, version);
    let mut all = files.clone();
    all.insert(path.to_owned(), text.to_owned());
    let host = ValueHost { files: all, runner: Box::new(runner) };
    // Part displays the caller holds, by digest: an assembly sends those prototypes `reused`.
    let known = super::prototype_reuse::known_from_options(options);
    // A drawing compiles in the stage the options ask for (drawing/progress.rs).
    let stage = crate::drawing::progress::Stage::from_options(options);
    let out = with_value_host(host, || crate::drawing::progress::with_stage(stage, || super::compile_json_known(path, text, files, configuration.as_deref(), &known)));
    let pending = pending.borrow();
    if pending.is_empty() {
        return out;
    }
    // The compile JSON is an object; add the field without re-reading a large display.
    let list = serde_json::to_string(&*pending).unwrap_or_else(|_| "[]".into());
    match out.trim_end().strip_suffix('}') {
        Some(head) if head.trim_end().ends_with('{') => format!("{head}\"pending_values\":{list}}}"),
        Some(head) => format!("{head},\"pending_values\":{list}}}"),
        None => out,
    }
}
