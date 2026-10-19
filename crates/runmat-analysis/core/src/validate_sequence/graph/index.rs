//! Original declaration lookup and compressed adjacency with counted flat storage.
use super::*;
pub(in crate::validate_sequence) struct StepIndex<'a>(pub(super) Vec<(usize, &'a ProcedureStep)>);
impl<'a> StepIndex<'a> {
    pub(super) fn index(&self, id: &str) -> Option<usize> {
        self.0
            .binary_search_by(|(_, step)| step.step_id.as_str().cmp(id))
            .ok()
    }
    pub(in crate::validate_sequence) fn get(&self, id: &str) -> Option<&'a ProcedureStep> {
        self.index(id).map(|i| self.0[i].1)
    }
    pub(in crate::validate_sequence) fn values(
        &self,
    ) -> impl Iterator<Item = &'a ProcedureStep> + '_ {
        self.0.iter().map(|(_, step)| *step)
    }
    pub(super) fn keys(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.values().map(|step| step.step_id.as_str())
    }
}
pub(super) struct Adjacency {
    pub(super) offsets: Vec<usize>,
    pub(super) neighbors: Vec<usize>,
}
impl Adjacency {
    pub(super) fn from_edges(count: usize, edges: &[(usize, usize)], reverse: bool) -> Self {
        let mut offsets = vec![0; count + 1];
        for &(from, to) in edges {
            offsets[if reverse { to } else { from } + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut cursor = offsets[..count].to_vec();
        let mut neighbors = vec![0; edges.len()];
        for &(from, to) in edges {
            let (a, b) = if reverse { (to, from) } else { (from, to) };
            neighbors[cursor[a]] = b;
            cursor[a] += 1;
        }
        Self { offsets, neighbors }
    }
    pub(super) fn row(&self, i: usize) -> &[usize] {
        &self.neighbors[self.offsets[i]..self.offsets[i + 1]]
    }
}
