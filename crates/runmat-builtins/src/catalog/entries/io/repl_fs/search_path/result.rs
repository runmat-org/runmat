use runmat_types::{ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(in crate::catalog::entries::io::repl_fs) fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}
