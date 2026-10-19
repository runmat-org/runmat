//! Bounded inline storage with an overflow path for unselected callers.

pub(super) struct InlineSlots<T, const N: usize> {
    slots: [Option<T>; N],
    overflow: Vec<T>,
}

impl<T, const N: usize> Default for InlineSlots<T, N> {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            overflow: Vec::new(),
        }
    }
}

impl<T, const N: usize> InlineSlots<T, N> {
    pub(super) fn push(&mut self, value: T) {
        if let Some(slot) = self.slots.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(value);
        } else {
            self.overflow.push(value);
        }
    }

    pub(super) fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.slots
            .iter_mut()
            .filter_map(Option::take)
            .chain(self.overflow.drain(..))
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots
            .iter()
            .filter_map(Option::as_ref)
            .chain(self.overflow.iter())
    }

    pub(super) fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
        self.overflow.clear();
    }
}
