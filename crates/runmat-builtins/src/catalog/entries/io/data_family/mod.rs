//! Public class identities shared by dataset path and handle contracts.

pub const DATASET_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("Dataset");
pub const DATA_ARRAY_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("DataArray");
pub const DATA_TRANSACTION_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("DataTransaction");
