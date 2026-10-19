use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseFormat {
    Csr,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseIndexWidth {
    U32,
    U64,
}

impl SparseIndexWidth {
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::U32 => 4,
            Self::U64 => 8,
        }
    }

    pub(super) fn accepts(self, value: u64) -> bool {
        match self {
            Self::U32 => u32::try_from(value).is_ok(),
            Self::U64 => true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseTranspose {
    None,
    Transpose,
    ConjugateTranspose,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparsePreconditionerKind {
    Jacobi,
    Ilu0,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseKrylovMethod {
    ConjugateGradient,
    Gmres,
    Bicgstab,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseKrylovTermination {
    Converged,
    MaximumIterations,
    NumericalBreakdown,
    Cancelled,
    ProviderLost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseProviderLossReason {
    Unknown,
    Destroyed,
}
