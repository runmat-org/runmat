//! Audio file metadata and sample decoding builtins.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerClass, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_filesystem as fs;
use runmat_macros::runtime_builtin;
use runmat_value::{IntegerStorage, NumericDType, StructValue, Tensor, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod contract;
mod error;
mod flac_aiff;
mod ieee;
mod info;
mod mp3;
mod ogg;
mod read;
mod wave;
#[cfg(test)]
mod tests;

use contract::*;
pub use contract::{
    AUDIOINFO_DESCRIPTOR, AUDIOREAD_DESCRIPTOR, AUDIOREAD_FUSION_SPEC, AUDIOREAD_GPU_SPEC,
    AUDIOREAD_INTEGER_CAPABILITIES, FUSION_SPEC, GPU_SPEC,
};
use error::*;
use flac_aiff::*;
use ieee::*;
use info::*;
use mp3::*;
use ogg::*;
use read::*;
use wave::*;
