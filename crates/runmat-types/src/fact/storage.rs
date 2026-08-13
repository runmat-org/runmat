use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageFact {
    Unknown,
    Scalar,
    Dense,
    Sparse,
    Opaque,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutFact {
    Unknown,
    ColumnMajor,
    RowMajor,
    Strided,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContiguityFact {
    Unknown,
    Contiguous,
    NonContiguous,
}

/// Whether a value owns its materialized storage or is a view over another
/// value. The fact deliberately carries no address, offset, or physical-layout
/// promise; those are runtime representation details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViewFact {
    Unknown,
    Materialized,
    ReadOnlyView,
    MutableView,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidencyFact {
    Unknown,
    Host,
    Device { provider: Option<String> },
    Remote { pool: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AliasFact {
    Unknown,
    Unique,
    Shared,
    Identity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MutationFact {
    Unknown,
    Immutable,
    ValueSemantics,
    HandleSemantics,
}
