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
