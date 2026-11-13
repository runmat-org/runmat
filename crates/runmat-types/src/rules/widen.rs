use crate::{DimensionFact, FactJoin, ShapeFact, ValueFact, ValueSequenceFact};

pub trait FactWiden: Sized {
    fn widen(&self, next: &Self) -> Self;
}

impl FactWiden for ValueFact {
    fn widen(&self, next: &Self) -> Self {
        let mut widened = self.join(next);
        if let ShapeFact::Shaped { dims } = &mut widened.shape {
            for dim in dims {
                if matches!(dim, DimensionFact::Symbolic(_)) {
                    *dim = DimensionFact::Unknown;
                }
            }
        }
        widened
    }
}

impl FactWiden for ValueSequenceFact {
    fn widen(&self, next: &Self) -> Self {
        Self {
            outputs: if self.outputs.len() == next.outputs.len() {
                self.outputs
                    .iter()
                    .zip(&next.outputs)
                    .map(|(current, incoming)| current.widen(incoming))
                    .collect()
            } else {
                Vec::new()
            },
            variadic: self.variadic || next.variadic || self.outputs.len() != next.outputs.len(),
        }
    }
}
