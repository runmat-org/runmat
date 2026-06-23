use crate::GcHandle;

/// Receives GC handles discovered while tracing an owned value graph.
pub trait Tracer {
    fn mark(&mut self, handle: GcHandle);
}

/// Describes how a type exposes nested GC handles.
pub trait Trace {
    fn trace(&self, tracer: &mut dyn Tracer);
}
