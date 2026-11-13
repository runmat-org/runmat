//! Iterative Value tracing with one resumable frame per active aggregate.

use crate::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
    ObjectInstance, StructArray, StructValue, Value,
};
use runmat_gc_api::{GcHandle, Trace, Tracer};
use std::alloc::{alloc, Layout};

/// Reusable traversal workspace. The admitted form charges each exact stack
/// backing before it is allocated; the unbounded form preserves ordinary
/// `Trace` compatibility. Neither form copies a Value or its children.
pub struct ValueTraceWorkspace<'value, 'permit> {
    frames: Vec<Frame<'value>>,
    claim: Option<ExactHostAllocationClaim>,
    permit: Option<&'permit dyn ExactHostAllocationPermit>,
}

impl<'value, 'permit> ValueTraceWorkspace<'value, 'permit> {
    pub fn unbounded() -> Self {
        Self {
            frames: Vec::new(),
            claim: None,
            permit: None,
        }
    }

    pub fn admitted(permit: &'permit dyn ExactHostAllocationPermit) -> Self {
        Self {
            frames: Vec::new(),
            claim: None,
            permit: Some(permit),
        }
    }

    /// Trace a protected Value graph. An admitted allocation failure must be
    /// propagated before publishing any registration from the partial trace.
    pub fn try_trace(
        &mut self,
        source: &'value Value,
        tracer: &mut dyn Tracer,
    ) -> Result<(), ExactHostNumericAllocationError> {
        self.frames.clear();
        let mut current = Some(Step::Value(source));
        loop {
            match current.take() {
                Some(Step::Mark(handle)) => tracer.mark(handle),
                Some(Step::Value(value)) => {
                    if let Value::Closure(closure) = value {
                        // The former pending stack popped ancestor handles
                        // before captures. These frames follow that same
                        // order without collecting the ancestry in a Vec.
                        if !closure.captures.is_empty() {
                            self.push_or_clear(Frame::Slice(closure.captures.iter()))?;
                        }
                        let mut ancestor = closure.lexical_environment.as_ref();
                        while let Some(environment) = ancestor {
                            self.push_or_clear(Frame::Mark(Some(environment.handle())))?;
                            ancestor = environment.trace_parent();
                        }
                    } else if let Some(mut frame) = Frame::for_value(value, tracer) {
                        if let Some(child) = frame.next() {
                            self.push_or_clear(frame)?;
                            current = Some(child);
                        }
                    }
                }
                None => {}
            }
            if current.is_some() {
                continue;
            }
            let Some(frame) = self.frames.last_mut() else {
                debug_assert!(self.frames.is_empty());
                return Ok(());
            };
            if let Some(step) = frame.next() {
                current = Some(step);
            } else {
                self.frames.pop();
            }
        }
    }

    fn push_or_clear(
        &mut self,
        frame: Frame<'value>,
    ) -> Result<(), ExactHostNumericAllocationError> {
        if let Err(error) = self.push(frame) {
            self.frames.clear();
            return Err(error);
        }
        Ok(())
    }

    fn push(&mut self, frame: Frame<'value>) -> Result<(), ExactHostNumericAllocationError> {
        if self.frames.len() == self.frames.capacity() {
            if let Some(permit) = self.permit {
                self.grow_admitted(permit)?;
            } else {
                self.frames
                    .try_reserve(1)
                    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
            }
        }
        self.frames.push(frame);
        Ok(())
    }

    fn grow_admitted(
        &mut self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactHostNumericAllocationError> {
        let capacity = self
            .frames
            .capacity()
            .checked_mul(2)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?
            .max(4);
        let layout = Layout::array::<Frame<'value>>(capacity)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
        let claim = permit.reserve_exact_claim(layout.size())?;
        // SAFETY: the non-null pointer owns exactly the requested frame-array
        // layout; its initialized length starts at zero.
        let pointer = unsafe { alloc(layout) }.cast::<Frame<'value>>();
        if pointer.is_null() {
            return Err(ExactHostNumericAllocationError::AllocationFailed);
        }
        let mut replacement = unsafe { Vec::from_raw_parts(pointer, 0, capacity) };
        for frame in self.frames.drain(..) {
            replacement.push(frame);
        }
        let retired = std::mem::replace(&mut self.frames, replacement);
        drop(retired); // Free old backing before refunding its claim.
        self.claim = Some(claim);
        Ok(())
    }
}

enum Step<'a> {
    Value(&'a Value),
    Mark(GcHandle),
}

enum Frame<'a> {
    Mark(Option<GcHandle>),
    Slice(std::slice::Iter<'a, Value>),
    Struct {
        source: &'a StructValue,
        index: usize,
    },
    StructArray {
        source: &'a StructArray,
        field: usize,
        element: usize,
    },
    Object {
        source: &'a ObjectInstance,
        native_index: usize,
        native_handle_index: usize,
        property_index: usize,
        metadata_index: usize,
    },
}

impl<'a> Frame<'a> {
    fn for_value(value: &'a Value, tracer: &mut dyn Tracer) -> Option<Self> {
        match value {
            Value::Cell(cells) => Some(Self::Slice(cells.data.iter())),
            Value::Struct(source) => Some(Self::Struct { source, index: 0 }),
            Value::StructArray(source) => Some(Self::StructArray {
                source,
                field: 0,
                element: 0,
            }),
            Value::HandleObject(handle) => {
                tracer.mark(handle.target);
                None
            }
            Value::Listener(listener) => {
                tracer.mark(listener.target);
                tracer.mark(listener.callback);
                None
            }
            Value::Closure(_) => None, // The cursor installs ordered ancestry frames.
            Value::Object(source) => Some(Self::Object {
                source,
                native_index: 0,
                native_handle_index: 0,
                property_index: 0,
                metadata_index: 0,
            }),
            Value::ObjectArray(array) => Some(Self::Slice(array.data().iter())),
            Value::OutputList(values) => Some(Self::Slice(values.iter())),
            Value::Int(_)
            | Value::Num(_)
            | Value::Complex(_, _)
            | Value::Bool(_)
            | Value::LogicalArray(_)
            | Value::String(_)
            | Value::StringArray(_)
            | Value::CharArray(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::ComplexTensor(_)
            | Value::Symbolic(_)
            | Value::SymbolicArray(_)
            | Value::GpuTensor(_)
            | Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::ClassRef(_)
            | Value::MException(_)
            | Value::Future(_)
            | Value::Task(_)
            | Value::Pool(_)
            | Value::Job(_)
            | Value::Distributed(_)
            | Value::Composite(_)
            | Value::Foreign(_) => None,
        }
    }

    fn next(&mut self) -> Option<Step<'a>> {
        match self {
            Self::Mark(handle) => handle.take().map(Step::Mark),
            Self::Slice(values) => values.next().map(Step::Value),
            Self::Struct { source, index } => {
                let child = source.fields.get_index(*index)?.1;
                *index += 1;
                Some(Step::Value(child))
            }
            Self::StructArray {
                source,
                field,
                element,
            } => loop {
                let values = source.trace_field_values(*field)?;
                if let Some(child) = values.get(*element) {
                    *element += 1;
                    return Some(Step::Value(child));
                }
                *field += 1;
                *element = 0;
            },
            Self::Object {
                source,
                native_index,
                native_handle_index,
                property_index,
                metadata_index,
            } => {
                if let Some(handle) = source.trace_native_handle_at(*native_handle_index) {
                    *native_handle_index += 1;
                    return Some(Step::Mark(handle));
                }
                if let Some(child) = source.trace_native_value_at(*native_index) {
                    *native_index += 1;
                    return Some(Step::Value(child));
                }
                if let Some(child) = source
                    .properties()
                    .get_index(*property_index)
                    .map(|entry| entry.1)
                {
                    *property_index += 1;
                    return Some(Step::Value(child));
                }
                loop {
                    let property = source
                        .dynamic_property_definitions()
                        .nth(*metadata_index)?
                        .1;
                    *metadata_index += 1;
                    if let Some(handle) = property.metadata_handle {
                        return Some(Step::Mark(handle));
                    }
                }
            }
        }
    }
}

impl Trace for Value {
    fn trace(&self, tracer: &mut dyn Tracer) {
        ValueTraceWorkspace::unbounded()
            .try_trace(self, tracer)
            .expect("unbounded Value trace workspace allocation failed");
    }
}
