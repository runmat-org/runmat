mod validation;
#[cfg(test)]
mod tests;

use runmat_types::{ShapeFact, ValueFact};

pub(in crate::catalog::entries::io) fn scalar(value: &ValueFact) -> bool {
    validation::scalar(value)
}

pub(in crate::catalog::entries::io) fn character_container(shape: &ShapeFact) -> bool {
    validation::character_container(shape)
}
