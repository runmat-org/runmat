//! Session-owned GC registration for persistent native MEX handle tokens.

use std::rc::Rc;

use runmat_gc::{GcHandle, GcRoot, ValueRoot, ValueRootRegistration};
use runmat_mex::MxValueContext;
use runmat_value::Value;

/// The registration and its stable token table have exactly the same owner.
/// Native code receives opaque tokens, never a second GC root inventory.
pub(super) struct RegisteredMxValueContext {
    context: Rc<MxValueContext>,
    _root: ValueRootRegistration,
}

impl RegisteredMxValueContext {
    pub(super) fn new(context: Rc<MxValueContext>) -> runmat_gc::Result<Self> {
        let root = ValueRootRegistration::new(Box::new(MexTokenRoot(Rc::clone(&context))))?;
        Ok(Self {
            context,
            _root: root,
        })
    }

    pub(super) fn context(&self) -> Rc<MxValueContext> {
        Rc::clone(&self.context)
    }

    pub(super) fn visit_retained_handles(
        &self,
        visitor: &mut dyn FnMut(&runmat_value::HandleRef) -> Option<()>,
    ) -> Option<()> {
        self.context.visit_retained_handles(visitor)
    }
}

struct MexTokenRoot(Rc<MxValueContext>);

impl GcRoot for MexTokenRoot {
    fn scan(&self) -> Vec<GcHandle> {
        let mut handles = Vec::new();
        self.0
            .visit_retained_handles(&mut |handle| {
                handles.push(handle.target);
                Some(())
            })
            .expect("a registered MEX token table must be readable on its owner thread");
        handles
    }

    fn description(&self) -> String {
        "persistent MEX handle tokens".into()
    }
}

impl ValueRoot for MexTokenRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        self.0.visit_retained_handles(&mut |handle| {
            // A HandleRef clone copies only the identity, never its GC payload.
            visitor(&Value::HandleObject(handle.clone()))
        })
    }
}
