#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainState {
    Accepting,
    Draining,
    Complete,
}

impl DrainState {
    pub fn begin(&mut self) {
        if *self == Self::Accepting {
            *self = Self::Draining;
        }
    }

    pub fn complete_if_idle(&mut self, active_allocations: usize) -> bool {
        if *self == Self::Draining && active_allocations == 0 {
            *self = Self::Complete;
            true
        } else {
            false
        }
    }
}
