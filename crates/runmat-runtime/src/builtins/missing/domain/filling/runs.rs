//! Missing-run geometry shared by nearest numeric and text filling.
//! The cursor observes each original boundary before any replacement occurs.

pub(super) struct MissingRuns {
    cursor: usize,
    length: usize,
}

pub(super) struct MissingRun {
    pub start: usize,
    pub end: usize,
    previous: Option<usize>,
    next: Option<usize>,
}

impl MissingRuns {
    pub fn new(length: usize) -> Self {
        Self { cursor: 0, length }
    }

    pub fn next(&mut self, mut is_missing: impl FnMut(usize) -> bool) -> Option<MissingRun> {
        while self.cursor < self.length && !is_missing(self.cursor) {
            self.cursor += 1;
        }
        if self.cursor == self.length {
            return None;
        }
        let start = self.cursor;
        while self.cursor < self.length && is_missing(self.cursor) {
            self.cursor += 1;
        }
        Some(MissingRun {
            start,
            end: self.cursor,
            previous: start.checked_sub(1),
            next: (self.cursor < self.length).then_some(self.cursor),
        })
    }
}

impl MissingRun {
    pub fn nearest_source(&self, position: usize) -> Option<usize> {
        debug_assert!(position >= self.start && position < self.end);
        match (self.previous, self.next) {
            (Some(previous), Some(next)) if next - position < position - previous => Some(next),
            (Some(previous), _) => Some(previous),
            (_, next) => next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_sources_use_original_boundaries_and_previous_on_ties() {
        let missing = [true, false, true, true, true, false, true];
        let mut runs = MissingRuns::new(missing.len());
        let mut sources = Vec::new();
        while let Some(run) = runs.next(|index| missing[index]) {
            sources.extend((run.start..run.end).map(|index| run.nearest_source(index)));
        }
        assert_eq!(sources, vec![Some(1), Some(1), Some(1), Some(5), Some(5)]);
    }

    #[test]
    fn long_missing_run_is_scanned_linearly_and_all_missing_has_no_source() {
        let length = 10_000;
        let mut calls = 0;
        let mut runs = MissingRuns::new(length);
        let run = runs
            .next(|_| {
                calls += 1;
                true
            })
            .unwrap();
        assert_eq!((run.start, run.end), (0, length));
        assert_eq!(run.nearest_source(0), None);
        assert_eq!(run.nearest_source(length - 1), None);
        assert!(runs
            .next(|_| panic!("terminal cursor must not inspect data"))
            .is_none());
        assert_eq!(calls, length + 1);
        assert!(MissingRuns::new(0).next(|_| panic!("empty line")).is_none());
    }
}
