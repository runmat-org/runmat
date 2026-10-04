//! MATLAB-compatible `delete` builtin for RunMat.

use runmat_filesystem as vfs;
use std::io;
use std::path::{Path, PathBuf};

use glob::{Pattern, PatternError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::builtins::common::fs::{contains_wildcards, expand_user_path, path_to_string};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::delete")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "delete",
    op_kind: GpuOpKind::Custom("io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Host-only filesystem operation. Resident arguments are rejected before provider access.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::delete")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "delete",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Filesystem side-effects are executed immediately; metadata registered for completeness.",
};

#[path = "delete/contract.rs"]
mod contract;
use contract::*;
pub use contract::{DELETE_DESCRIPTOR, DELETE_INTEGER_CAPABILITIES};

#[runtime_builtin(
    name = "delete",
    category = "io/repl_fs",
    summary = "Remove files.",
    keywords = "delete,remove file,wildcard delete,cleanup,temporary files,MATLAB delete",
    accel = "cpu",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::delete_type),
    descriptor(crate::builtins::io::repl_fs::delete::DELETE_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::repl_fs::delete::DELETE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::repl_fs::delete"
)]
async fn delete_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: missing filename input",
        ));
    }
    if args.iter().any(crate::value_contains_gpu) {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: resident arguments are not supported",
        ));
    }
    if let [Value::Pool(pool)] = args.as_slice() {
        let context = crate::context::legacy::active().ok_or_else(|| {
            delete_error_with(
                &DELETE_ERROR_RUNTIME_CONTEXT,
                DELETE_ERROR_RUNTIME_CONTEXT.message,
            )
        })?;
        crate::parallel::pool::close(&context, pool)?;
        return Ok(Value::Num(0.0));
    }
    let (operands, resolve_symbolic_links) = parse_delete_options(&args)?;
    let gathered = gather_arguments(operands).await?;

    if gathered.iter().all(is_handle_input) {
        if resolve_symbolic_links.is_some() {
            return Err(delete_error_with(
                &DELETE_ERROR_INVALID_INPUT,
                "delete: ResolveSymbolicLinks is only valid for filename inputs",
            ));
        }
        return delete_handles(&gathered).await;
    }

    if gathered.iter().any(contains_handle_input) {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_HANDLE,
            "delete: cannot mix handle and filename inputs",
        ));
    }

    let mut raw_targets = Vec::new();
    for value in &gathered {
        collect_targets(value, &mut raw_targets)?;
    }

    if raw_targets.is_empty() {
        return Ok(Value::Num(0.0));
    }

    for raw in raw_targets {
        delete_target(&raw, resolve_symbolic_links.unwrap_or(false)).await?;
    }

    Ok(Value::Num(0.0))
}

#[path = "delete/input.rs"]
mod input;
use input::*;

#[path = "delete/filesystem.rs"]
mod filesystem;
use filesystem::*;

#[path = "delete/handles.rs"]
mod handles;
use handles::*;

#[cfg(test)]
#[path = "delete/tests.rs"]
pub(crate) mod tests;
