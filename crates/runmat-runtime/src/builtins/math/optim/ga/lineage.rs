//! Public per-row creation labels describe work that actually produced the
//! installed generation, not the raw objective's measurement provenance.
#[derive(Clone, Copy)]
pub(super) struct Parents {
    pub first: usize,
    pub second: Option<usize>,
}
#[derive(Clone, Copy)]
pub(super) enum Lineage {
    Supplied,
    Creation,
    Elite,
    Crossover,
    Mutation,
    Retained,
    Migration,
    Observer,
}
impl Lineage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Supplied => "supplied",
            Self::Creation => "creation",
            Self::Elite => "elite",
            Self::Crossover => "crossover",
            Self::Mutation => "mutation",
            Self::Retained => "retained",
            Self::Migration => "migration",
            Self::Observer => "observer",
        }
    }
}
