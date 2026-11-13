/// Monotonic identity for one committed base-workspace state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkspaceRevision(u64);

impl WorkspaceRevision {
    pub const INITIAL: Self = Self(0);

    pub const fn get(self) -> u64 {
        self.0
    }

    pub(super) fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }

    #[cfg(test)]
    pub(super) const fn from_raw(value: u64) -> Self {
        Self(value)
    }
}
