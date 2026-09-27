use super::{HandleRef, MxValueContext};

impl MxValueContext {
    /// Borrow the origin-thread handle identities retained by this conversion
    /// context. The visitor must not mutate the context. Target values are not
    /// copied or resolved here; their existing GC owner controls access.
    ///
    /// This covers only the token table, not arbitrary native module pointers,
    /// callback services, or a lane's queued MxArray arguments.
    pub fn visit_retained_handles(
        &self,
        visitor: &mut dyn FnMut(&HandleRef) -> Option<()>,
    ) -> Option<()> {
        self.require_origin_thread().ok()?;
        for entry in self.handles.try_borrow().ok()?.values() {
            visitor(&entry.value)?;
        }
        Some(())
    }
}
