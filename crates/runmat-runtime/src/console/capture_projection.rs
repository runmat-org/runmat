use super::ConsoleEntry;
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
    pub(crate) fn snapshot(&self) -> Self {
        Self {
            next_token: 0,
            frames: self
                .frames
                .iter()
                .filter(|(_, frame)| frame.active.get())
                .map(|(_, frame)| (0, Rc::clone(frame)))
                .collect(),
        }
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

    pub(crate) fn begin(owner: &Rc<RefCell<Self>>) -> CaptureLease {
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
            owner: Rc::clone(owner),
            token,
            frame,
        }
    }
}

pub(crate) struct CaptureLease {
    owner: Rc<RefCell<ConsoleCaptureProjection>>,
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
        let parent = Rc::new(RefCell::new(ConsoleCaptureProjection::default()));
        let outer = ConsoleCaptureProjection::begin(&parent);
        let child = Rc::new(RefCell::new(parent.borrow().snapshot()));
        let inner = ConsoleCaptureProjection::begin(&child);
        assert!(child.borrow().record(entry("inner")));
        assert_eq!(inner.finish()[0].text, "inner");
        assert!(child.borrow().record(entry("outer")));
        assert_eq!(outer.finish()[0].text, "outer");
        assert!(!child.borrow().record(entry("after parent retirement")));
    }
}
