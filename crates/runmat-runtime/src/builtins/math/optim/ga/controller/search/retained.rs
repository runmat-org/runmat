//! Generic searches retain the same quoted original control owner.
pub(in super::super) type Retained<G = super::super::super::linear::Domain> =
    super::super::super::control::Retained<super::Search<G>>;
