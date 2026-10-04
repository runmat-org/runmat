//! Typed host-effect identity shared with resource evidence and placement.

use runmat_builtins::AudioReadDataType;
use std::path::Path;

mod dataset;
pub use dataset::DatasetPathOperation;

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
    DatasetPath {
        operation: DatasetPathOperation,
        source: &'a Path,
        destination: Option<&'a Path>,
        output: HostEffectOutput,
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

impl HostEffectWorkload<'_> {
    pub const fn output(self) -> HostEffectOutput {
        match self {
            Self::ArchiveExtract { output, .. }
            | Self::AudioMetadata { output, .. }
            | Self::DatasetPath { output, .. } => output,
            Self::AudioDecode { output, .. } => output.host_effect_output(),
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
        }
    }
}
