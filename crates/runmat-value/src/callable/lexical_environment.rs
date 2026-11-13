use runmat_gc_api::GcHandle;
use std::rc::Rc;

/// Identity of a private GC lexical activation. Binding state lives only in
/// the typed GC payload; this host metadata carries no parallel slot map.
#[derive(Debug, Clone)]
pub struct LexicalEnvironment {
    node: Rc<LexicalEnvironmentNode>,
}

#[derive(Debug)]
struct LexicalEnvironmentNode {
    handle: GcHandle,
    parent: Option<LexicalEnvironment>,
}

impl LexicalEnvironment {
    pub fn new(handle: GcHandle, parent: Option<LexicalEnvironment>) -> Self {
        Self {
            node: Rc::new(LexicalEnvironmentNode { handle, parent }),
        }
    }

    pub fn handle(&self) -> GcHandle {
        self.node.handle
    }

    pub(crate) fn trace_parent(&self) -> Option<&Self> {
        self.node.parent.as_ref()
    }

    /// Host-local graph traversal only. The returned activation is never a
    /// portable identity; the admitted function layout supplies that proof.
    pub fn parent(&self) -> Option<Self> {
        self.node.parent.clone()
    }

    pub fn for_each_handle(&self, mut visit: impl FnMut(GcHandle)) {
        let mut current = Some(self);
        while let Some(environment) = current {
            visit(environment.node.handle);
            current = environment.node.parent.as_ref();
        }
    }
}

impl PartialEq for LexicalEnvironment {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.node, &other.node)
    }
}
