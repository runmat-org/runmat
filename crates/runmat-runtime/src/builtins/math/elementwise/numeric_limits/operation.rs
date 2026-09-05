use runmat_builtins::{FloatingLimitKind, IntegerLimitKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LimitOperation {
    Integer {
        name: &'static str,
        kind: IntegerLimitKind,
    },
    Floating {
        name: &'static str,
        kind: FloatingLimitKind,
    },
}

impl LimitOperation {
    pub(super) const INTMAX: Self = Self::Integer {
        name: "intmax",
        kind: IntegerLimitKind::Maximum,
    };
    pub(super) const INTMIN: Self = Self::Integer {
        name: "intmin",
        kind: IntegerLimitKind::Minimum,
    };
    pub(super) const REALMAX: Self = Self::Floating {
        name: "realmax",
        kind: FloatingLimitKind::LargestFinite,
    };
    pub(super) const REALMIN: Self = Self::Floating {
        name: "realmin",
        kind: FloatingLimitKind::SmallestNormal,
    };
    pub(super) const FLINTMAX: Self = Self::Floating {
        name: "flintmax",
        kind: FloatingLimitKind::LargestConsecutiveInteger,
    };
}
