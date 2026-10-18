//! Inline storage for the common single-phase, single-scope operation.
//! General nested operations retain their ordered, dynamically sized storage.
use std::ops::{Deref, DerefMut};

pub(super) enum PhaseStorage<T> {
    Single([T; 1]),
    Multiple(Box<[T]>),
}

impl<T> PhaseStorage<T> {
    pub(super) fn from_array<const N: usize>(values: [T; N]) -> Self {
        if N == 1 {
            return match values.into_iter().next() {
                Some(value) => Self::Single([value]),
                None => unreachable!("a single-element array has one phase"),
            };
        }
        Self::Multiple(Box::new(values))
    }
}

impl<T: Clone> PhaseStorage<T> {
    pub(super) fn filled(value: T, len: usize) -> Self {
        if len == 1 {
            Self::Single([value])
        } else {
            Self::Multiple(vec![value; len].into_boxed_slice())
        }
    }
}

impl<T> Deref for PhaseStorage<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Self::Single(value) => value,
            Self::Multiple(values) => values,
        }
    }
}

impl<T> DerefMut for PhaseStorage<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Single(value) => value,
            Self::Multiple(values) => values,
        }
    }
}

pub(super) struct ScopeStorage<T> {
    first: Option<T>,
    additional: Vec<T>,
}

impl<T> Default for ScopeStorage<T> {
    fn default() -> Self {
        Self {
            first: None,
            additional: Vec::new(),
        }
    }
}

impl<T> ScopeStorage<T> {
    pub(super) fn len(&self) -> usize {
        usize::from(self.first.is_some()) + self.additional.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.first.is_none()
    }

    pub(super) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        match index {
            0 => self.first.as_mut(),
            _ => self.additional.get_mut(index - 1),
        }
    }

    pub(super) fn push(&mut self, value: T) {
        if self.first.is_none() {
            self.first = Some(value);
        } else {
            self.additional.push(value);
        }
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &T> {
        self.first.iter().chain(self.additional.iter())
    }
}
