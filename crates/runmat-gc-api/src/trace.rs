use crate::GcHandle;

/// Receives GC handles discovered while tracing an owned value graph.
pub trait Tracer<T> {
    fn mark(&mut self, handle: GcHandle<T>);
}

/// Describes how a type exposes nested GC handles.
pub trait Trace<T> {
    fn trace(&self, tracer: &mut dyn Tracer<T>);
}
