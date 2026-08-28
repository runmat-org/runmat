use crate::{DimensionFact, FactJoin, ShapeFact, ValueFact};

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
