/// Stable local placement policy. Both margins must be cleared before a more
/// expensive-risk candidate can displace the current best candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlacementPolicy {
    pub(crate) absolute_margin_ns: u64,
    pub(crate) relative_margin_basis_points: u32,
}

impl Default for PlacementPolicy {
    fn default() -> Self {
        Self {
            absolute_margin_ns: 5_000,
            relative_margin_basis_points: 250,
        }
    }
}

impl PlacementPolicy {
    pub(crate) fn required_improvement_ns(self, incumbent_ns: u64) -> Option<u64> {
        let relative = incumbent_ns
            .checked_mul(u64::from(self.relative_margin_basis_points))?
            .checked_add(9_999)?
            / 10_000;
        Some(self.absolute_margin_ns.max(relative))
    }
}
