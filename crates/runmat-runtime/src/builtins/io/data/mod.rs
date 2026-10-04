//! Cloud-ready dataset persistence builtins (`data.*`).

mod array;
mod dataset;
mod execution;
mod handle_admission;
mod kernel;
mod path;
mod path_authority;
mod transaction;

use kernel::*;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(test)]
use runmat_builtins::{DATA_CREATE_DESCRIPTOR, DATA_CREATE_INTEGER_CAPABILITIES};
use runmat_filesystem::data_contract::{DataChunkDescriptor, DataChunkUploadRequest};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericScalar, ObjectInstance, StructValue, Tensor, Value};

use crate::builtins::common::json::int_value_to_json;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::data::authority::{self, AuthorizedHandle, HandleRole};
#[cfg(test)]
use crate::data::dataset_root;
use crate::data::{
    array_object, data_error, dataset_object, ensure_manifest_sequence, get_object_prop,
    manifest_path, manifest_version_token, new_tx_id, now_rfc3339, parse_schema, parse_string,
    remove_tx, sha256_hex, start_tx_with_id, transaction_object, validate_chunk_shape, with_tx,
    with_tx_mut, DataArrayMeta, DataArrayPayload, DataChunkIndex, DataChunkIndexEntry,
    DataManifest, PendingCreateArray, PendingFill, PendingResize, PendingWrite, TxnStatus,
};
use crate::{make_cell, BuiltinResult};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::data")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "data.*",
    op_kind: GpuOpKind::Custom("io-data"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Dataset operations are host I/O and metadata orchestration.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::data")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "data.*",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Data builtins are side-effecting and not fusible.",
};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
