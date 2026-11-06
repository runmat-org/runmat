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
use std::collections::{BTreeMap, BTreeSet};
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

/// The Desktop's and the browser's compile entry (`options`: `{ configuration?, values? }`):
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
    let out = with_value_host(host, || super::compile_json_in(path, text, files, configuration.as_deref()));
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

/// The folders a part's `.m` calls can reach, as folder dependencies (`/p/`): its own folder
/// and `private/`, the folders its `use`s bring in, and `+pkg` folders above it for `pkg.fn`.
/// Empty when the part calls no function.
pub fn call_folders(path: &str, text: &str, document: &runmat_cad::model::PartDocument) -> Vec<String> {
    let mut functions = BTreeSet::new();
    if let Ok(json) = serde_json::to_value(document) {
        collect_calls(&json, &mut functions);
    }
    if functions.is_empty() {
        return Vec::new();
    }
    let folder = runmat_cad::modules::folder_of(path);
    let listing = |f: &str| if f.ends_with('/') { f.to_owned() } else { format!("{f}/") };
    let mut out: BTreeSet<String> = BTreeSet::new();
    out.insert(listing(folder));
    out.insert(listing(&runmat_cad::modules::normalize(&format!("{folder}/private"))));
    out.extend(runmat_cad::modules::dependencies(path, text));
    for function in &functions {
        let segments: Vec<&str> = function.split('.').collect();
        if segments.len() < 2 {
            continue;
        }
        let package = segments[..segments.len() - 1].iter().map(|s| format!("+{s}")).collect::<Vec<_>>().join("/");
        for at in runmat_cad::modules::ancestors(folder) {
            out.insert(listing(&runmat_cad::modules::normalize(&format!("{at}/{package}"))));
        }
    }
    out.into_iter().collect()
}

fn collect_calls(value: &serde_json::Value, out: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(obj) => {
            if obj.get("form").and_then(|f| f.as_str()) == Some("dynamic") {
                if let Some(f) = obj.get("call").and_then(|c| c.get("function")).and_then(|f| f.as_str()) {
                    out.insert(f.to_owned());
                }
            }
            obj.values().for_each(|v| collect_calls(v, out));
        }
        serde_json::Value::Array(items) => items.iter().for_each(|v| collect_calls(v, out)),
        _ => {}
    }
}
