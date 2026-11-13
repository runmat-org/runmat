//! Dataset namespace effects whose path and result facts must be certified together.

use runmat_execution::Digest;
use runmat_types::ValueFact;

/// Full typed value fact plus an exact bounded-control digest. `None` means the
/// input's semantic identity is not proven at this boundary; a finite route
/// cannot be certified from shape/class alone.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DatasetHandleInput {
    pub fact: ValueFact,
    pub identity_digest: Option<Digest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum DatasetPathOperation {
    Create,
    Open,
    Exists,
    Delete,
    Copy,
    Move,
    Import,
    Export,
    List,
    Inspect,
}

/// The exact managed receiver role, supplied only after sealed handle
/// resolution. Names and IDs are data, never builtin-spelling dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum DatasetHandleRole<'a> {
    Dataset,
    Array(&'a str),
    Transaction(&'a str),
}

/// Registry-issued identity; public object fields cannot construct this fact.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DatasetHandleKey<'a> {
    pub host_identity: &'a str,
    pub handle: u64,
    pub generation: u64,
}

/// One typed identity per public receiver form, including the commit alias.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum DatasetHandleOperation {
    DatasetArray,
    DatasetArrays,
    DatasetAttrs,
    DatasetBegin,
    DatasetGetAttr,
    DatasetHasArray,
    DatasetId,
    DatasetPath,
    DatasetRefresh,
    DatasetSetAttr,
    DatasetSetAttrs,
    DatasetSnapshot,
    DatasetVersion,
    ArrayChunkShape,
    ArrayCodec,
    ArrayDType,
    ArrayFill,
    ArrayName,
    ArrayRank,
    ArrayRead,
    ArrayResize,
    ArrayShape,
    ArrayWrite,
    TransactionAbort,
    TransactionCommit,
    TransactionCreateArray,
    TransactionDeleteArray,
    TransactionFill,
    TransactionId,
    TransactionResize,
    TransactionSetAttr,
    TransactionSetAttrs,
    TransactionStatus,
    TransactionWrite,
    CommitAlias,
}

impl DatasetHandleOperation {
    pub const fn accepts_role(self, role: DatasetHandleRole<'_>) -> bool {
        matches!(
            (self, role),
            (
                Self::DatasetArray
                    | Self::DatasetArrays
                    | Self::DatasetAttrs
                    | Self::DatasetBegin
                    | Self::DatasetGetAttr
                    | Self::DatasetHasArray
                    | Self::DatasetId
                    | Self::DatasetPath
                    | Self::DatasetRefresh
                    | Self::DatasetSetAttr
                    | Self::DatasetSetAttrs
                    | Self::DatasetSnapshot
                    | Self::DatasetVersion,
                DatasetHandleRole::Dataset,
            ) | (
                Self::ArrayChunkShape
                    | Self::ArrayCodec
                    | Self::ArrayDType
                    | Self::ArrayFill
                    | Self::ArrayName
                    | Self::ArrayRank
                    | Self::ArrayRead
                    | Self::ArrayResize
                    | Self::ArrayShape
                    | Self::ArrayWrite,
                DatasetHandleRole::Array(_),
            ) | (
                Self::TransactionAbort
                    | Self::TransactionCommit
                    | Self::TransactionCreateArray
                    | Self::TransactionDeleteArray
                    | Self::TransactionFill
                    | Self::TransactionId
                    | Self::TransactionResize
                    | Self::TransactionSetAttr
                    | Self::TransactionSetAttrs
                    | Self::TransactionStatus
                    | Self::TransactionWrite
                    | Self::CommitAlias,
                DatasetHandleRole::Transaction(_),
            )
        )
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::DatasetArray => "Dataset.array",
            Self::DatasetArrays => "Dataset.arrays",
            Self::DatasetAttrs => "Dataset.attrs",
            Self::DatasetBegin => "Dataset.begin",
            Self::DatasetGetAttr => "Dataset.get_attr",
            Self::DatasetHasArray => "Dataset.has_array",
            Self::DatasetId => "Dataset.id",
            Self::DatasetPath => "Dataset.path",
            Self::DatasetRefresh => "Dataset.refresh",
            Self::DatasetSetAttr => "Dataset.set_attr",
            Self::DatasetSetAttrs => "Dataset.set_attrs",
            Self::DatasetSnapshot => "Dataset.snapshot",
            Self::DatasetVersion => "Dataset.version",
            Self::ArrayChunkShape => "DataArray.chunk_shape",
            Self::ArrayCodec => "DataArray.codec",
            Self::ArrayDType => "DataArray.dtype",
            Self::ArrayFill => "DataArray.fill",
            Self::ArrayName => "DataArray.name",
            Self::ArrayRank => "DataArray.rank",
            Self::ArrayRead => "DataArray.read",
            Self::ArrayResize => "DataArray.resize",
            Self::ArrayShape => "DataArray.shape",
            Self::ArrayWrite => "DataArray.write",
            Self::TransactionAbort => "DataTransaction.abort",
            Self::TransactionCommit => "DataTransaction.commit",
            Self::TransactionCreateArray => "DataTransaction.create_array",
            Self::TransactionDeleteArray => "DataTransaction.delete_array",
            Self::TransactionFill => "DataTransaction.fill",
            Self::TransactionId => "DataTransaction.id",
            Self::TransactionResize => "DataTransaction.resize",
            Self::TransactionSetAttr => "DataTransaction.set_attr",
            Self::TransactionSetAttrs => "DataTransaction.set_attrs",
            Self::TransactionStatus => "DataTransaction.status",
            Self::TransactionWrite => "DataTransaction.write",
            Self::CommitAlias => "commit",
        }
    }
}
