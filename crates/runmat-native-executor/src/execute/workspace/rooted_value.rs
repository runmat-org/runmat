//! One workspace payload with safe, reference-counted GC protection.
//!
//! Trace visits immediate handles in the ordinary Value tree; it does not
//! borrow or traverse GC objects. ExplicitRoot then lets the collector trace
//! those objects normally. No root scanner accesses the workspace RefCell.
use runmat_gc::{ExplicitRoot, GcHandle, Trace, Tracer};
use runmat_value::Value;

pub(super) struct RootedWorkspaceValue {
    pub(super) value: Value,
    _roots: Vec<ExplicitRoot>,
}

impl RootedWorkspaceValue {
    pub(super) fn new(value: Value) -> Result<Self, Box<runmat_runtime::RuntimeError>> {
        #[derive(Default)]
        struct Roots {
            guards: Vec<ExplicitRoot>,
            error: Option<runmat_gc::GcError>,
        }
        impl Tracer for Roots {
            fn mark(&mut self, handle: GcHandle) {
                if self.error.is_none() {
                    match runmat_gc::gc_root(handle) {
                        Ok(root) => self.guards.push(root),
                        Err(error) => self.error = Some(error),
                    }
                }
            }
        }
        let mut roots = Roots::default();
        value.trace(&mut roots);
        if let Some(error) = roots.error {
            return Err(Box::new(runmat_runtime::runtime_error::semantic_error(
                "NativeWorkspaceRootRegistration",
                format!("unable to protect native workspace value: {error}"),
            )));
        }
        Ok(Self {
            value,
            _roots: roots.guards,
        })
    }
}
