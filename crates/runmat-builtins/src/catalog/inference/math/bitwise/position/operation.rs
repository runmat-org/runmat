#[derive(Clone, Copy)]
pub(super) enum PositionOperation {
    Get,
    Set,
}

impl PositionOperation {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Get => "bitget",
            Self::Set => "bitset",
        }
    }

    pub(super) const fn maximum_arity(self) -> usize {
        match self {
            Self::Get => 3,
            Self::Set => 4,
        }
    }

    pub(super) const fn value_argument_count(self) -> usize {
        match self {
            Self::Get => 2,
            Self::Set => 3,
        }
    }
}
