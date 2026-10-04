//! Audio file metadata and sample decoding builtins.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use runmat_builtins::AudioSampleCoding as WaveSampleCoding;
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    AudioReadDataType, AUDIOINFO_DESCRIPTOR, AUDIOINFO_ERROR_ARGUMENT, AUDIOINFO_ERROR_FORMAT,
    AUDIOINFO_ERROR_IO, AUDIOREAD_DESCRIPTOR, AUDIOREAD_ERROR_ARGUMENT, AUDIOREAD_ERROR_FORMAT,
    AUDIOREAD_ERROR_IO, AUDIOREAD_INTEGER_CAPABILITIES,
};
use runmat_filesystem as fs;
use runmat_macros::runtime_builtin;
use runmat_value::{IntegerStorage, NumericDType, StructValue, Tensor, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const AUDIOINFO_BUILTIN_NAME: &str = "audioinfo";
const AUDIOREAD_BUILTIN_NAME: &str = "audioread";
const MAX_AUDIOINFO_PREFIX_BYTES: u64 = 1024 * 1024;
const MAX_AUDIOINFO_TAIL_BYTES: u64 = 64 * 1024;

mod error;
mod admission;
mod cooperation;
mod flac_aiff;
mod ieee;
pub(crate) mod info;
mod input;
mod mp3;
mod ogg;
pub(crate) mod read;
mod stream;
mod wave;
#[cfg(test)]
mod tests;

use admission::*;
use cooperation::*;
use error::*;
use flac_aiff::*;
use ieee::*;
use info::*;
use input::*;
use mp3::*;
use ogg::*;
use read::*;
use stream::*;
use wave::*;
