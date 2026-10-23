//! Closed durable numerical factor data. This is never an execution capability.
use super::super::{DenseValue, SparseValue};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableFactorPermutation {
    pub forward: Vec<u64>,
    pub inverse: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableOrthogonalFactor {
    pub r: SparseValue,
    pub reflectors: SparseValue,
    pub permutation: DurableFactorPermutation,
    pub rank: u64,
}

/// Canonical packed dense orthogonal state, not a reconstructed sparse factor.
/// The coefficient tensor's first axis is faer's retained block size.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableDenseOrthogonalFactor {
    pub packed: DenseValue,
    pub coefficients: DenseValue,
    pub permutation: DurableFactorPermutation,
    pub rank: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurableEmptyFactorKind {
    Lu,
    Cholesky,
    Qr,
    Cod,
}

/// A structural decomposition retains its original matrix and recipe. It has
/// no numerical factorization payload and must be checked by the Runtime's
/// canonical decomposition owner before publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurableStructuralDecomposition {
    Triangular,
    Diagonal,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DurableDecompositionPayload {
    Data(Box<DurableNumericFactor>),
    Structural(DurableStructuralDecomposition),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DurableNumericFactor {
    Empty {
        kind: DurableEmptyFactorKind,
        rows: u64,
        columns: u64,
        class: super::super::ElementType,
    },
    Lu {
        // Box only the wide sparse members; serde preserves their original
        // field shape while the closed enum stays bounded on the stack.
        lower: Box<SparseValue>,
        upper: Box<SparseValue>,
        rows: DurableFactorPermutation,
        columns: DurableFactorPermutation,
        row_scale: Option<DenseValue>,
    },
    Cholesky {
        lower: SparseValue,
        permutation: DurableFactorPermutation,
    },
    DenseLu {
        packed: DenseValue,
        rows: DurableFactorPermutation,
    },
    DenseCholesky {
        lower: DenseValue,
    },
    DenseQr(Box<DurableDenseOrthogonalFactor>),
    DenseCod {
        first: Box<DurableDenseOrthogonalFactor>,
        completion: Option<Box<DurableDenseOrthogonalFactor>>,
    },
    Qr(DurableOrthogonalFactor),
    Cod {
        first: Box<DurableOrthogonalFactor>,
        completion: Option<Box<DurableOrthogonalFactor>>,
    },
}
