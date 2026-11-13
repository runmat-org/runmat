//! Poll-local selection of the session's numeric presentation mode.

use super::FormatMode;
use runmat_thread_local::runmat_thread_local;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// The mutable format for one session, shared by its nested invocation frames.
#[derive(Clone, Debug)]
pub struct DisplayFormatOwner(Rc<Cell<FormatMode>>);

impl Default for DisplayFormatOwner {
    fn default() -> Self {
        Self::new(FormatMode::default())
    }
}

impl DisplayFormatOwner {
    pub fn new(mode: FormatMode) -> Self {
        Self(Rc::new(Cell::new(mode)))
    }

    pub fn get(&self) -> FormatMode {
        self.0.get()
    }

    pub fn set(&self, mode: FormatMode) {
        self.0.set(mode);
    }
}

runmat_thread_local! {
    static SELECTED_OWNER: RefCell<Option<DisplayFormatOwner>> = const { RefCell::new(None) };
    static BARE_FORMAT: Cell<FormatMode> = const { Cell::new(FormatMode::Short) };
}

/// Restore the previous selection after one synchronous poll or nested call.
#[must_use]
pub struct DisplayFormatSelectionGuard {
    previous: Option<DisplayFormatOwner>,
}

impl Drop for DisplayFormatSelectionGuard {
    fn drop(&mut self) {
        SELECTED_OWNER.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

/// Select the owner itself, not a copied mode. Setters remain visible to every
/// nested invocation sharing that owner and survive suspension of this poll.
pub fn select_display_format_owner(owner: &DisplayFormatOwner) -> DisplayFormatSelectionGuard {
    select_optional_display_format_owner(Some(owner))
}

/// Explicitly select the bare fallback without inheriting a foreign session.
pub fn select_optional_display_format_owner(
    owner: Option<&DisplayFormatOwner>,
) -> DisplayFormatSelectionGuard {
    let previous = SELECTED_OWNER.with(|slot| slot.replace(owner.cloned()));
    DisplayFormatSelectionGuard { previous }
}

pub fn set_display_format(mode: FormatMode) {
    let selected = SELECTED_OWNER.with(|slot| slot.borrow().clone());
    if let Some(owner) = selected {
        owner.set(mode);
    } else {
        BARE_FORMAT.with(|slot| slot.set(mode));
    }
}

pub fn get_display_format() -> FormatMode {
    let selected = SELECTED_OWNER.with(|slot| slot.borrow().clone());
    selected.map_or_else(|| BARE_FORMAT.with(Cell::get), |owner| owner.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_selection_updates_owner_and_restores_caller_and_bare_fallback() {
        let prior_bare = get_display_format();
        set_display_format(FormatMode::Short);
        let parent = DisplayFormatOwner::new(FormatMode::Long);
        let child = DisplayFormatOwner::new(FormatMode::Rational);

        let parent_scope = select_display_format_owner(&parent);
        assert_eq!(get_display_format(), FormatMode::Long);
        set_display_format(FormatMode::Hex);
        assert_eq!(parent.get(), FormatMode::Hex);

        let child_scope = select_display_format_owner(&child);
        assert_eq!(get_display_format(), FormatMode::Rational);
        set_display_format(FormatMode::LongG);
        assert_eq!(child.get(), FormatMode::LongG);
        assert_eq!(parent.get(), FormatMode::Hex);
        drop(child_scope);

        assert_eq!(get_display_format(), FormatMode::Hex);
        drop(parent_scope);
        assert_eq!(get_display_format(), FormatMode::Short);
        set_display_format(prior_bare);
    }

    #[test]
    fn alternating_poll_selections_keep_independent_modes() {
        let left = DisplayFormatOwner::new(FormatMode::Short);
        let right = DisplayFormatOwner::new(FormatMode::Long);
        {
            let _poll = select_display_format_owner(&left);
            set_display_format(FormatMode::ShortE);
        }
        {
            let _poll = select_display_format_owner(&right);
            assert_eq!(get_display_format(), FormatMode::Long);
            set_display_format(FormatMode::Rational);
        }
        {
            let _poll = select_display_format_owner(&left);
            assert_eq!(get_display_format(), FormatMode::ShortE);
        }
        assert_eq!(right.get(), FormatMode::Rational);
    }
}
