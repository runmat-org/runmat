//! Invocation-scoped foreign callback ancestry. Each pushed frame shares its
//! parent; no foreign payload, workspace, or adapter registry is copied.

use std::rc::Rc;

pub(super) struct CallbackLineage<T> {
    authority: T,
    parent: Option<Rc<Self>>,
}

impl<T> CallbackLineage<T> {
    pub(super) fn push(parent: Option<Rc<Self>>, authority: T) -> Rc<Self> {
        Rc::new(Self { authority, parent })
    }

    /// The most recent frame for this exact lower host wins, even when stale.
    /// Passing a stale match to the lower owner must reject; it must not fall
    /// back to ordinary submission or resurrect an older frame.
    pub(super) fn newest_for(&self, mut belongs_to: impl FnMut(&T) -> bool) -> Option<&T> {
        let mut frame = Some(self);
        while let Some(current) = frame {
            if belongs_to(&current.authority) {
                return Some(&current.authority);
            }
            frame = current.parent.as_deref();
        }
        None
    }
}
