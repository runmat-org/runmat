//! Typed host-effect identity shared with resource evidence and placement.

use runmat_builtins::AudioReadDataType;
use runmat_execution::Digest;
use std::path::Path;

mod dataset;
pub use dataset::{
    DatasetHandleInput, DatasetHandleKey, DatasetHandleOperation, DatasetHandleRole,
    DatasetPathOperation,
};
mod mat;
pub use mat::{
    MatAssignmentSuffix, MatDecodeOutput, MatDecodeSelection, MatEncodeMode, MatEncodeOutput,
    MatHandleControl, MatStructSelection,
};

/// The requested result is distinct from the committed side effects. No
/// output still needs a bound for temporary codec and result storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum HostEffectOutput {
    EffectOnly,
    ReturnedValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostEffectSource<'a> {
    LocalArchive(&'a Path),
    HttpArchive(&'a str),
}

/// One-based sample-frame selection. The audio domain validates this before
/// constructing the workload; this type carries it without reparsing text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct AudioFrameRange {
    pub first: u64,
    pub last: Option<u64>,
}

/// The two-result form retains both samples and the sample-rate value. The
/// zero-result form still requires evidence for temporary decoded storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum AudioDecodeOutput {
    EffectOnly,
    Samples,
    SamplesAndRate,
}

impl AudioDecodeOutput {
    pub const fn host_effect_output(self) -> HostEffectOutput {
        match self {
            Self::EffectOnly => HostEffectOutput::EffectOnly,
            Self::Samples | Self::SamplesAndRate => HostEffectOutput::ReturnedValue,
        }
    }
}

/// Typed operation identity at the service evidence boundary; no builtin
/// spelling switch or secondary scheduler is introduced here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostEffectWorkload<'a> {
    ArchiveExtract {
        source: HostEffectSource<'a>,
        destination: Option<&'a Path>,
        output: HostEffectOutput,
    },
    AudioMetadata {
        source: &'a Path,
        output: HostEffectOutput,
    },
    AudioDecode {
        source: &'a Path,
        range: Option<AudioFrameRange>,
        datatype: AudioReadDataType,
        output: AudioDecodeOutput,
    },
    ReadMatrixText {
        source: &'a Path,
        controls: Digest,
        output: HostEffectOutput,
    },
    DatasetPath {
        operation: DatasetPathOperation,
        source: &'a Path,
        destination: Option<&'a Path>,
        output: HostEffectOutput,
    },
    DatasetHandle {
        operation: DatasetHandleOperation,
        root: &'a Path,
        key: DatasetHandleKey<'a>,
        role: DatasetHandleRole<'a>,
        inputs: &'a [DatasetHandleInput],
        output: HostEffectOutput,
    },
    MatDecode {
        source: &'a Path,
        selection: MatDecodeSelection<'a>,
        output: MatDecodeOutput,
    },
    MatEncode {
        destination: &'a Path,
        mode: MatEncodeMode<'a>,
        output: MatEncodeOutput,
    },
    MatHandle {
        source: &'a Path,
        controls: &'a [MatHandleControl],
    },
}

#[derive(serde::Serialize)]
enum HostEffectSourceFacts<'a> {
    LocalArchive(&'a [u8]),
    HttpArchive(&'a str),
}

#[derive(serde::Serialize)]
enum AudioWorkloadFacts<'a> {
    Metadata {
        source: &'a [u8],
        output: HostEffectOutput,
    },
    Decode {
        source: &'a [u8],
        range: Option<AudioFrameRange>,
        datatype: AudioReadDataType,
        output: AudioDecodeOutput,
    },
}

#[derive(serde::Serialize)]
struct DatasetPathFacts<'a> {
    operation: DatasetPathOperation,
    source: &'a [u8],
    destination: Option<&'a [u8]>,
    output: HostEffectOutput,
}

#[derive(serde::Serialize)]
struct DatasetHandleFacts<'a> {
    operation: DatasetHandleOperation,
    root: &'a [u8],
    key: DatasetHandleKey<'a>,
    role: DatasetHandleRole<'a>,
    inputs: &'a [DatasetHandleInput],
    output: HostEffectOutput,
}

#[derive(serde::Serialize)]
struct MatDecodeFacts<'a> {
    source: &'a [u8],
    selection: MatDecodeSelection<'a>,
    output: MatDecodeOutput,
}

#[derive(serde::Serialize)]
struct MatEncodeFacts<'a> {
    destination: &'a [u8],
    mode: MatEncodeMode<'a>,
    output: MatEncodeOutput,
}

#[derive(serde::Serialize)]
struct MatHandleFacts<'a> {
    source: &'a [u8],
    controls: &'a [MatHandleControl],
}

#[derive(serde::Serialize)]
struct ReadMatrixTextWorkloadFacts<'a> {
    source: &'a [u8],
    controls: Digest,
    output: HostEffectOutput,
}

impl HostEffectWorkload<'_> {
    /// An absent control digest is not a fabricated reusable identity.
    pub fn identity_complete(self) -> bool {
        match self {
            Self::DatasetHandle { inputs, .. } => {
                inputs.iter().all(|input| input.identity_digest.is_some())
            }
            Self::MatEncode {
                mode:
                    MatEncodeMode::AssignMember {
                        suffix: MatAssignmentSuffix::Unavailable,
                        ..
                    },
                ..
            }
            | Self::MatDecode {
                selection:
                    MatDecodeSelection::MemberPath {
                        suffix: MatAssignmentSuffix::Unavailable,
                        ..
                    },
                ..
            } => false,
            _ => true,
        }
    }

    pub const fn output(self) -> HostEffectOutput {
        match self {
            Self::ArchiveExtract { output, .. }
            | Self::AudioMetadata { output, .. }
            | Self::ReadMatrixText { output, .. }
            | Self::DatasetPath { output, .. } => output,
            Self::DatasetHandle { output, .. } => output,
            Self::AudioDecode { output, .. } => output.host_effect_output(),
            Self::MatDecode { output, .. } => output.host_effect_output(),
            Self::MatEncode { output, .. } => output.host_effect_output(),
            Self::MatHandle { .. } => HostEffectOutput::ReturnedValue,
        }
    }

    /// Cache and service identities use the exact platform path encoding.
    /// The archive arm intentionally preserves its original serialized tuple
    /// so existing archive placement identities and invalidation stay stable.
    pub fn identity_bytes(self) -> Vec<u8> {
        match self {
            Self::ArchiveExtract {
                source,
                destination,
                output,
            } => {
                let source = match source {
                    HostEffectSource::LocalArchive(path) => {
                        HostEffectSourceFacts::LocalArchive(path.as_os_str().as_encoded_bytes())
                    }
                    HostEffectSource::HttpArchive(url) => HostEffectSourceFacts::HttpArchive(url),
                };
                serde_json::to_vec(&(
                    source,
                    destination.map(|path| path.as_os_str().as_encoded_bytes()),
                    output,
                ))
                .expect("typed host effect path bytes serialize")
            }
            Self::AudioMetadata { source, output } => {
                serde_json::to_vec(&AudioWorkloadFacts::Metadata {
                    source: source.as_os_str().as_encoded_bytes(),
                    output,
                })
                .expect("typed audio metadata facts serialize")
            }
            Self::AudioDecode {
                source,
                range,
                datatype,
                output,
            } => serde_json::to_vec(&AudioWorkloadFacts::Decode {
                source: source.as_os_str().as_encoded_bytes(),
                range,
                datatype,
                output,
            })
            .expect("typed audio decode facts serialize"),
            Self::ReadMatrixText {
                source,
                controls,
                output,
            } => serde_json::to_vec(&ReadMatrixTextWorkloadFacts {
                source: source.as_os_str().as_encoded_bytes(),
                controls,
                output,
            })
            .expect("typed numeric-text workload serializes"),
            Self::DatasetPath {
                operation,
                source,
                destination,
                output,
            } => serde_json::to_vec(&DatasetPathFacts {
                operation,
                source: source.as_os_str().as_encoded_bytes(),
                destination: destination.map(|path| path.as_os_str().as_encoded_bytes()),
                output,
            })
            .expect("typed dataset path facts serialize"),
            Self::DatasetHandle {
                operation,
                root,
                key,
                role,
                inputs,
                output,
            } => serde_json::to_vec(&DatasetHandleFacts {
                operation,
                root: root.as_os_str().as_encoded_bytes(),
                key,
                role,
                inputs,
                output,
            })
            .expect("typed dataset handle facts serialize"),
            Self::MatDecode {
                source,
                selection,
                output,
            } => serde_json::to_vec(&MatDecodeFacts {
                source: source.as_os_str().as_encoded_bytes(),
                selection,
                output,
            })
            .expect("typed MAT decode facts serialize"),
            Self::MatEncode {
                destination,
                mode,
                output,
            } => serde_json::to_vec(&MatEncodeFacts {
                destination: destination.as_os_str().as_encoded_bytes(),
                mode,
                output,
            })
            .expect("typed MAT encode facts serialize"),
            Self::MatHandle { source, controls } => serde_json::to_vec(&MatHandleFacts {
                source: source.as_os_str().as_encoded_bytes(),
                controls,
            })
            .expect("typed MAT handle facts serialize"),
        }
    }
}
