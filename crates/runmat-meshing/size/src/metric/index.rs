//! Compact indices into canonical contributions, grouped by borrowed scope.
use super::*;
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ContributionIndex(Vec<usize>);
fn entity(c: &MetricContribution) -> &PersistentEntityId {
    match &c.scope {
        MetricContributionScope::Region { region_id } => region_id,
        MetricContributionScope::Entity { entity_id } => entity_id,
    }
}
impl ContributionIndex {
    pub(super) fn new(request: &MetricFieldRequest) -> Self {
        let mut indices = (0..request.contributions.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|a, b| {
            entity(&request.contributions[*a])
                .cmp(entity(&request.contributions[*b]))
                .then(a.cmp(b))
        });
        Self(indices)
    }
    pub(super) fn applicable(
        &self,
        request: &MetricFieldRequest,
        incident: &BTreeSet<PersistentEntityId>,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        for id in incident {
            let start = self
                .0
                .partition_point(|i| entity(&request.contributions[*i]) < id);
            let end = self
                .0
                .partition_point(|i| entity(&request.contributions[*i]) <= id);
            result.extend_from_slice(&self.0[start..end]);
        }
        result.sort_unstable();
        result
    }
}
