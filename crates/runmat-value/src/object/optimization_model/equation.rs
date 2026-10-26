//! Actual equation values use the existing native state and iterative Value tracer.
use super::*;
#[derive(Debug, PartialEq)]
pub struct EquationProblemData {
    pub description: Value,
    pub equations: Value,
}
impl EquationProblemData {
    pub fn value_at(&self, index: usize) -> Option<&Value> {
        match index {
            0 => Some(&self.description),
            1 => Some(&self.equations),
            _ => None,
        }
    }
}
impl Trace for EquationProblemData {
    fn trace(&self, tracer: &mut dyn Tracer) {
        let mut index = 0;
        while let Some(value) = self.value_at(index) {
            value.trace(tracer);
            index += 1;
        }
    }
}
