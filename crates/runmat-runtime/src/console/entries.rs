//! Captured entries retain their collection allocation through consumption.

use super::ConsoleEntry;
use crate::context::overlay_storage::{AdmittedOverlayIntoIter, AdmittedOverlayVec};
use std::ops::Deref;

pub struct CapturedConsoleEntries {
    storage: AdmittedOverlayVec<ConsoleEntry>,
}

impl CapturedConsoleEntries {
    pub(crate) fn from_storage(storage: AdmittedOverlayVec<ConsoleEntry>) -> Self {
        Self { storage }
    }

    pub(crate) fn into_storage(self) -> AdmittedOverlayVec<ConsoleEntry> {
        self.storage
    }
}

impl From<Vec<ConsoleEntry>> for CapturedConsoleEntries {
    fn from(values: Vec<ConsoleEntry>) -> Self {
        Self::from_storage(AdmittedOverlayVec::Unbounded {
            values,
            generation: 0,
        })
    }
}

impl Default for CapturedConsoleEntries {
    fn default() -> Self {
        Self::from_storage(AdmittedOverlayVec::unbounded())
    }
}

impl Deref for CapturedConsoleEntries {
    type Target = [ConsoleEntry];

    fn deref(&self) -> &Self::Target {
        self.storage.as_slice()
    }
}

impl std::fmt::Debug for CapturedConsoleEntries {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}

impl IntoIterator for CapturedConsoleEntries {
    type Item = ConsoleEntry;
    type IntoIter = CapturedConsoleEntriesIter;

    fn into_iter(self) -> Self::IntoIter {
        CapturedConsoleEntriesIter {
            inner: self.storage.into_iter(),
        }
    }
}

impl<'a> IntoIterator for &'a CapturedConsoleEntries {
    type Item = &'a ConsoleEntry;
    type IntoIter = std::slice::Iter<'a, ConsoleEntry>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct CapturedConsoleEntriesIter {
    inner: AdmittedOverlayIntoIter<ConsoleEntry>,
}

impl Iterator for CapturedConsoleEntriesIter {
    type Item = ConsoleEntry;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for CapturedConsoleEntriesIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl ExactSizeIterator for CapturedConsoleEntriesIter {}
