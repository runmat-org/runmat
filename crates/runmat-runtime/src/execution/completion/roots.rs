//! Incremental ownership of GC roots retained by decoded values.

use std::collections::HashSet;
use std::rc::Rc;

use runmat_gc::{ExplicitRoot, GcError, GcHandle, Trace, Tracer};
use runmat_value::Value;

/// Managed handles reachable from owned Values while an asynchronous owner
/// retains them. This roots identities, not cloned payloads or device buffers.
#[derive(Clone, Debug)]
pub(crate) struct RootedValueHandles {
    pub(super) _roots: Option<Rc<Vec<ExplicitRoot>>>,
}

/// One decoded value paired with the roots retained while constructing it.
///
/// `root` traces a standalone value. `aggregate` is a trusted crate-private
/// construction boundary: its caller must transfer every managed child edge
/// from the same aggregate into `children` and must not add an untracked edge.
#[derive(Debug)]
pub(crate) struct RootedValueMember {
    value: Value,
    roots: RetainedValueRoots,
}

impl RootedValueMember {
    pub(crate) fn root(value: Value) -> Result<Self, GcError> {
        let roots = RetainedValueRoots::direct(RootedValueHandles::for_traced(&value)?);
        Ok(Self { value, roots })
    }

    pub(crate) fn aggregate(value: Value, children: RootedValueChildren) -> Self {
        Self {
            value,
            roots: children.finish(),
        }
    }

    pub(crate) fn into_unrooted_value(self) -> Value {
        self.value
    }

    /// Promote a decoded member without retiring its construction roots first.
    pub(crate) fn into_rooted_value(
        self,
    ) -> Result<super::RootedValue, super::super::ExecutionServiceError> {
        let Self { value, roots } = self;
        let rooted = super::RootedValue::new(value)?;
        drop(roots);
        Ok(rooted)
    }

    pub(crate) fn into_child(self, children: &mut RootedValueChildren) -> Value {
        let Self { value, roots } = self;
        children.push(roots);
        value
    }

    pub(super) fn into_parts(self) -> (Value, RetainedValueRoots) {
        (self.value, self.roots)
    }
}

const ROOT_CHUNK_CAPACITY: usize = 64;

#[derive(Clone, Debug)]
pub(super) enum RetainedValueRoots {
    Empty,
    Direct { _roots: RootedValueHandles },
    Aggregate { _root: Rc<RootChunk> },
}

impl RetainedValueRoots {
    pub(super) fn direct(roots: RootedValueHandles) -> Self {
        if roots.is_empty() {
            Self::Empty
        } else {
            Self::Direct { _roots: roots }
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

/// Incremental root owner for aggregate children. Fixed-size linked chunks
/// make each transfer bounded without flattening descendant roots at every
/// ancestor. The final-drop path iteratively retires unique sibling chunks;
/// nested aggregate depth remains bounded by `ValueLimits::max_depth`.
#[derive(Debug, Default)]
pub(crate) struct RootedValueChildren {
    sealed: Option<Rc<RootChunk>>,
    current: Vec<RetainedValueRoots>,
}

impl RootedValueChildren {
    pub(crate) fn new() -> Self {
        Self {
            sealed: None,
            current: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, roots: RetainedValueRoots) {
        if roots.is_empty() {
            return;
        }
        if self.current.len() == ROOT_CHUNK_CAPACITY {
            let node = RootChunk {
                _roots: std::mem::replace(
                    &mut self.current,
                    Vec::with_capacity(ROOT_CHUNK_CAPACITY),
                ),
                previous: self.sealed.take(),
            };
            self.sealed = Some(Rc::new(node));
        }
        self.current.push(roots);
    }

    pub(super) fn finish(mut self) -> RetainedValueRoots {
        if self.current.is_empty() {
            return self
                .sealed
                .take()
                .map_or(RetainedValueRoots::Empty, |root| {
                    RetainedValueRoots::Aggregate { _root: root }
                });
        }
        RetainedValueRoots::Aggregate {
            _root: Rc::new(RootChunk {
                _roots: std::mem::take(&mut self.current),
                previous: self.sealed.take(),
            }),
        }
    }
}

#[derive(Debug)]
pub(super) struct RootChunk {
    _roots: Vec<RetainedValueRoots>,
    previous: Option<Rc<RootChunk>>,
}

impl Drop for RootChunk {
    fn drop(&mut self) {
        let mut previous = self.previous.take();
        while let Some(node) = previous {
            let Ok(mut node) = Rc::try_unwrap(node) else {
                break;
            };
            previous = node.previous.take();
            // `node` now owns no sibling link. Its bounded root chunk drops
            // here; nested aggregates are independently depth-limited.
        }
    }
}

impl RootedValueHandles {
    pub(crate) fn for_values<'a>(values: impl Iterator<Item = &'a Value>) -> Result<Self, GcError> {
        let mut collector = HandleCollector::default();
        for value in values {
            value.trace(&mut collector);
        }
        Self::from_handles(collector)
    }

    pub(crate) fn for_traced(value: &impl Trace) -> Result<Self, GcError> {
        let mut collector = HandleCollector::default();
        value.trace(&mut collector);
        Self::from_handles(collector)
    }

    fn from_handles(collector: HandleCollector) -> Result<Self, GcError> {
        if collector.handles.is_empty() {
            return Ok(Self { _roots: None });
        }
        let mut roots = Vec::with_capacity(collector.handles.len());
        for handle in collector.handles {
            roots.push(runmat_gc::gc_root(handle)?);
        }
        Ok(Self {
            _roots: Some(Rc::new(roots)),
        })
    }

    fn is_empty(&self) -> bool {
        self._roots.is_none()
    }
}

#[derive(Default)]
struct HandleCollector {
    seen: HashSet<GcHandle>,
    handles: Vec<GcHandle>,
}

impl Tracer for HandleCollector {
    fn mark(&mut self, handle: GcHandle) {
        if self.seen.insert(handle) {
            self.handles.push(handle);
        }
    }
}
