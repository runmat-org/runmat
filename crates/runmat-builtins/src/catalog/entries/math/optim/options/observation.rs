//! Shared solver-observer phases, decoded at public text boundaries.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimObserverState {
    Init,
    Iter,
    Done,
}

impl OptimObserverState {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Iter => "iter",
            Self::Done => "done",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "init" => Some(Self::Init),
            "iter" => Some(Self::Iter),
            "done" => Some(Self::Done),
            _ => None,
        }
    }
}
