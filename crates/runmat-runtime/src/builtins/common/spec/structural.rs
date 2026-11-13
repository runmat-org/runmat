//! Binding-owned structural graph operations without scalar shader templates.

/// Authorizes only graph construction. Operand class, shape, layout and
/// provider legality remain with the existing operation executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionStructuralOperation {
    MatrixMultiply,
}

impl FusionStructuralOperation {
    pub const fn input_arity(self) -> usize {
        match self {
            Self::MatrixMultiply => 2,
        }
    }
}
