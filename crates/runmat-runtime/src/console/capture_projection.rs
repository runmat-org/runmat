use super::ConsoleEntry;
use crate::context::poll_projection::PollProjection;
use runmat_value::{exact_capacity_vec, ExactHostNumericAllocationError};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

struct CaptureFrame {
    active: Cell<bool>,
    entries: RefCell<Vec<ConsoleEntry>>,
}

/// Poll-selected capture stack. Snapshots share only active outer collectors;
/// each child owns its nested stack and cannot capture a sibling's output.
#[derive(Default)]
pub(crate) struct ConsoleCaptureProjection {
    next_token: u64,
    frames: Vec<(u64, Rc<CaptureFrame>)>,
}

impl ConsoleCaptureProjection {
    pub(crate) fn poll_snapshot_scratch_bytes(&self) -> Option<u64> {
        let count = self
            .frames
            .iter()
            .filter(|(_, frame)| frame.active.get())
            .count();
        count
            .checked_mul(std::mem::size_of::<(u64, Rc<CaptureFrame>)>())
            .and_then(|bytes| u64::try_from(bytes).ok())
    }

    pub(crate) fn snapshot(&self) -> Self {
        let count = self
            .frames
            .iter()
            .filter(|(_, frame)| frame.active.get())
            .count();
        let mut frames = Vec::with_capacity(count);
        frames.extend(
            self.frames
                .iter()
                .filter(|(_, frame)| frame.active.get())
                .map(|(_, frame)| (0, Rc::clone(frame))),
        );
        Self {
            next_token: 0,
            frames,
        }
    }

    pub(crate) fn poll_snapshot(&self) -> Result<Self, ExactHostNumericAllocationError> {
        let count = self
            .frames
            .iter()
            .filter(|(_, frame)| frame.active.get())
            .count();
        let mut frames = exact_capacity_vec(count)?;
        frames.extend(
            self.frames
                .iter()
                .filter(|(_, frame)| frame.active.get())
                .map(|(_, frame)| (0, Rc::clone(frame))),
        );
        Ok(Self {
            next_token: 0,
            frames,
        })
    }

    pub(crate) fn record(&self, entry: ConsoleEntry) -> bool {
        let Some(frame) = self
            .frames
            .iter()
            .rev()
            .find(|(_, frame)| frame.active.get())
            .map(|(_, frame)| Rc::clone(frame))
        else {
            return false;
        };
        frame.entries.borrow_mut().push(entry);
        true
    }

    pub(crate) fn begin(owner: &PollProjection<Self>) -> CaptureLease {
        let frame = Rc::new(CaptureFrame {
            active: Cell::new(true),
            entries: RefCell::new(Vec::new()),
        });
        let token = {
            let mut projection = owner.borrow_mut();
            let token = projection.next_token;
            projection.next_token = token
                .checked_add(1)
                .expect("console capture identity exhausted");
            projection.frames.push((token, Rc::clone(&frame)));
            token
        };
        CaptureLease {
            owner: owner.clone(),
            token,
            frame,
        }
    }
}

pub(crate) struct CaptureLease {
    owner: PollProjection<ConsoleCaptureProjection>,
    token: u64,
    frame: Rc<CaptureFrame>,
}

impl CaptureLease {
    pub(crate) fn finish(self) -> Vec<ConsoleEntry> {
        let entries = std::mem::take(&mut *self.frame.entries.borrow_mut());
        drop(self);
        entries
    }
}

impl Drop for CaptureLease {
    fn drop(&mut self) {
        self.frame.active.set(false);
        let mut projection = self.owner.borrow_mut();
        if let Some(index) = projection
            .frames
            .iter()
            .position(|(token, frame)| *token == self.token && Rc::ptr_eq(frame, &self.frame))
        {
            projection.frames.remove(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::ConsoleStream;

    fn entry(text: &str) -> ConsoleEntry {
        ConsoleEntry {
            stream: ConsoleStream::Stdout,
            text: text.into(),
            timestamp_ms: 0,
        }
    }

    #[test]
    fn inherited_outer_capture_survives_child_overlay_but_closure_is_exact() {
        let parent = PollProjection::legacy(ConsoleCaptureProjection::default());
        let outer = ConsoleCaptureProjection::begin(&parent);
        let child = PollProjection::legacy(parent.borrow().snapshot());
        let inner = ConsoleCaptureProjection::begin(&child);
        assert!(child.borrow().record(entry("inner")));
        assert_eq!(inner.finish()[0].text, "inner");
        assert!(child.borrow().record(entry("outer")));
        assert_eq!(outer.finish()[0].text, "outer");
        assert!(!child.borrow().record(entry("after parent retirement")));
    }
}
