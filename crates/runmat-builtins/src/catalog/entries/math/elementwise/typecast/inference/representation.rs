use runmat_types::{NumericDomain, NumericFact, ValueKindFact};

#[derive(Clone, Copy)]
pub(super) enum OutputRepresentation {
    Numeric(NumericFact),
    Logical,
}

impl OutputRepresentation {
    pub(super) fn kind(self) -> ValueKindFact {
        match self {
            Self::Numeric(fact) => ValueKindFact::Numeric(fact),
            Self::Logical => ValueKindFact::Logical,
        }
    }

    pub(super) fn byte_width(self) -> usize {
        match self {
            Self::Numeric(fact) => fact.class.byte_width() * lanes(fact.domain),
            Self::Logical => 1,
        }
    }
}

pub(super) const fn lanes(domain: NumericDomain) -> usize {
    match domain {
        NumericDomain::Real => 1,
        NumericDomain::Complex => 2,
    }
}
