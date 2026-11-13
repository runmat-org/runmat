//! Staged final diagnostic backing. No public mutable String/Vec is silently
//! certified: only allocations prepared by this construction are retained.
use super::{CallFrame, RetainedDiagnosticAllocation, RuntimeError};
use std::{alloc::Layout, fmt, num::NonZeroUsize};

#[derive(Debug)]
pub enum DiagnosticConstructionError<E> {
    Admission(E),
    LayoutOverflow,
    AllocationFailed,
    InvalidState,
}
impl<E: fmt::Display> fmt::Display for DiagnosticConstructionError<E> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => fmt::Display::fmt(error, out),
            Self::LayoutOverflow => out.write_str("diagnostic allocation layout overflow"),
            Self::AllocationFailed => out.write_str("diagnostic allocation failed"),
            Self::InvalidState => out.write_str("diagnostic construction state is invalid"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for DiagnosticConstructionError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            _ => None,
        }
    }
}

/// Physical capacities for the final RuntimeError Strings and two context
/// arrays. Retained source/display owners are independently owned, not copied.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeDiagnosticLayout {
    pub text_bytes: usize,
    pub call_frames: usize,
    pub call_stack: usize,
}
impl RuntimeDiagnosticLayout {
    pub fn requested_bytes<E>(self) -> Result<usize, DiagnosticConstructionError<E>> {
        Layout::array::<CallFrame>(self.call_frames)
            .ok()
            .and_then(|frames| {
                Layout::array::<String>(self.call_stack)
                    .ok()
                    .and_then(|stack| frames.size().checked_add(stack.size()))
            })
            .and_then(|arrays| arrays.checked_add(self.text_bytes))
            .ok_or(DiagnosticConstructionError::LayoutOverflow)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeDiagnosticField {
    Message,
    Identifier,
    Builtin,
    TaskId,
    Frame(usize),
    CallStack(usize),
    Phase,
}

#[derive(Clone, Copy, Debug)]
pub struct DiagnosticCopyProgress {
    pub copied_bytes: usize,
    pub complete: bool,
}

pub struct RuntimeDiagnosticConstruction {
    // Partial Strings, frames and stack retire before their retained claim,
    // which is the last field of this original RuntimeError.
    error: RuntimeError,
    remaining_text: usize,
    expected_frames: usize,
    expected_stack: usize,
    active: Option<(RuntimeDiagnosticField, usize)>,
}
impl RuntimeDiagnosticConstruction {
    /// Admission runs before either context array or any String allocation.
    /// The caller must quote all final String capacities in text_bytes.
    pub fn try_new_exact<E>(
        layout: RuntimeDiagnosticLayout,
        admit: impl FnOnce(usize) -> Result<RetainedDiagnosticAllocation, E>,
    ) -> Result<Self, DiagnosticConstructionError<E>> {
        let claim =
            admit(layout.requested_bytes()?).map_err(DiagnosticConstructionError::Admission)?;
        let mut error = RuntimeError::new(String::new());
        error.retained_diagnostic_allocation = Some(claim);
        error.context.call_frames = exact_vec(layout.call_frames)?;
        error.context.call_stack = exact_vec(layout.call_stack)?;
        Ok(Self {
            error,
            remaining_text: layout.text_bytes,
            expected_frames: layout.call_frames,
            expected_stack: layout.call_stack,
            active: None,
        })
    }

    /// The frame's function text is initialized in its final slot by begin_text.
    /// Its source provenance/display already retains its original owner.
    pub fn push_frame<E>(
        &mut self,
        frame: CallFrame,
    ) -> Result<usize, DiagnosticConstructionError<E>> {
        if self.error.context.call_frames.len() == self.expected_frames
            || !frame.function.is_empty()
            || frame.function.capacity() != 0
        {
            return Err(DiagnosticConstructionError::InvalidState);
        }
        let index = self.error.context.call_frames.len();
        self.error.context.call_frames.push(frame);
        Ok(index)
    }

    pub fn push_call_stack_slot<E>(&mut self) -> Result<usize, DiagnosticConstructionError<E>> {
        if self.error.context.call_stack.len() == self.expected_stack {
            return Err(DiagnosticConstructionError::InvalidState);
        }
        let index = self.error.context.call_stack.len();
        self.error.context.call_stack.push(String::new());
        Ok(index)
    }

    pub fn begin_text<E>(
        &mut self,
        field: RuntimeDiagnosticField,
        bytes: usize,
    ) -> Result<(), DiagnosticConstructionError<E>> {
        if self.active.is_some() || bytes > self.remaining_text {
            return Err(DiagnosticConstructionError::InvalidState);
        }
        let target = self
            .target(field)
            .ok_or(DiagnosticConstructionError::InvalidState)?;
        if target.capacity() != 0 || !target.is_empty() {
            return Err(DiagnosticConstructionError::InvalidState);
        }
        // SAFETY: the empty exact Vec is valid UTF-8. Subsequent writes append
        // only complete source str segments and can never exceed this capacity.
        *target = unsafe { String::from_utf8_unchecked(exact_vec(bytes)?) };
        self.remaining_text -= bytes;
        self.active = Some((field, bytes));
        Ok(())
    }

    pub fn advance_text<E>(
        &mut self,
        source: &str,
        work: NonZeroUsize,
    ) -> Result<DiagnosticCopyProgress, DiagnosticConstructionError<E>> {
        let (field, bytes) = self
            .active
            .ok_or(DiagnosticConstructionError::InvalidState)?;
        if bytes != source.len() {
            return Err(DiagnosticConstructionError::InvalidState);
        }
        let target = self
            .target(field)
            .ok_or(DiagnosticConstructionError::InvalidState)?;
        let start = target.len();
        let end = text_end(source, start, work.get());
        target.push_str(&source[start..end]);
        let complete = end == bytes;
        if complete {
            self.active = None;
        }
        Ok(DiagnosticCopyProgress {
            copied_bytes: end - start,
            complete,
        })
    }

    fn target(&mut self, field: RuntimeDiagnosticField) -> Option<&mut String> {
        match field {
            RuntimeDiagnosticField::Message => Some(&mut self.error.message),
            RuntimeDiagnosticField::Identifier => {
                Some(self.error.identifier.get_or_insert_with(String::new))
            }
            RuntimeDiagnosticField::Builtin => {
                Some(self.error.context.builtin.get_or_insert_with(String::new))
            }
            RuntimeDiagnosticField::TaskId => {
                Some(self.error.context.task_id.get_or_insert_with(String::new))
            }
            RuntimeDiagnosticField::Frame(index) => self
                .error
                .context
                .call_frames
                .get_mut(index)
                .map(|frame| &mut frame.function),
            RuntimeDiagnosticField::CallStack(index) => {
                self.error.context.call_stack.get_mut(index)
            }
            RuntimeDiagnosticField::Phase => {
                Some(self.error.context.phase.get_or_insert_with(String::new))
            }
        }
    }

    /// An incomplete private destination is retired on failure; finishing
    /// never allocates an error box or publishes a partial diagnostic.
    pub fn try_finish(
        self,
    ) -> Result<RuntimeError, DiagnosticConstructionError<std::convert::Infallible>> {
        if self.active.is_none()
            && self.remaining_text == 0
            && self.error.context.call_frames.len() == self.expected_frames
            && self.error.context.call_stack.len() == self.expected_stack
        {
            Ok(self.error)
        } else {
            Err(DiagnosticConstructionError::InvalidState)
        }
    }
}

pub(super) fn exact_vec<T, E>(count: usize) -> Result<Vec<T>, DiagnosticConstructionError<E>> {
    let layout =
        Layout::array::<T>(count).map_err(|_| DiagnosticConstructionError::LayoutOverflow)?;
    if layout.size() == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: a successful allocation has exactly Vec<T>'s global-allocator
    // layout. Only the initialized prefix is exposed or dropped.
    let pointer = unsafe { std::alloc::alloc(layout) }.cast::<T>();
    if pointer.is_null() {
        return Err(DiagnosticConstructionError::AllocationFailed);
    }
    Ok(unsafe { Vec::from_raw_parts(pointer, 0, count) })
}

pub(super) fn text_end(source: &str, start: usize, work: usize) -> usize {
    let mut end = start.saturating_add(work).min(source.len());
    while !source.is_char_boundary(end) {
        end -= 1;
    }
    if end == start && start < source.len() {
        end += source[start..]
            .chars()
            .next()
            .expect("remaining source character")
            .len_utf8();
    }
    end
}
