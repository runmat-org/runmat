//! Cloud-ready dataset persistence builtins (`data.*`).

pub(crate) mod array;
pub(crate) mod dataset;
mod execution;
mod handle_admission;
mod handle_identity;
mod kernel;
pub(crate) mod path;
mod path_authority;
pub(crate) mod transaction;

use kernel::*;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(all(test, not(target_arch = "wasm32")))]
use runmat_builtins::{DATA_CREATE_DESCRIPTOR, DATA_CREATE_INTEGER_CAPABILITIES};
use runmat_filesystem::data_contract::{DataChunkDescriptor, DataChunkUploadRequest};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericScalar, ObjectInstance, Tensor, Value};

use crate::builtins::common::json::int_value_to_json;
use crate::builtins::common::tensor as tensor_utils;
use crate::data::authority::{self, AuthorizedHandle, HandleRole};
#[cfg(all(test, not(target_arch = "wasm32")))]
use crate::data::dataset_root;
use crate::data::{
    array_object, data_error, dataset_object, ensure_manifest_sequence, get_object_prop,
    manifest_path, manifest_version_token, new_tx_id, now_rfc3339, parse_schema, parse_string,
    remove_tx, start_tx_with_id, transaction_object, validate_chunk_shape, with_tx, with_tx_mut,
    DataArrayMeta, DataArrayPayload, DataChunkIndex, DataChunkIndexEntry, DataManifest,
    PendingCreateArray, PendingFill, PendingResize, PendingWrite, TxnStatus,
};
use crate::{make_cell, BuiltinResult};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
