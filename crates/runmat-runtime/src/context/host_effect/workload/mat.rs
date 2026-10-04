//! Closed MAT decode identity. The MAT domain has already parsed these
//! selectors; placement never interprets option spelling or regex syntax.

use super::HostEffectOutput;
use runmat_accelerate_api::GpuTensorWireDescriptor;
use runmat_execution::Digest;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum MatDecodeSelection<'a> {
    All,
    Selected {
        names: &'a [String],
        patterns: &'a [String],
    },
    MemberPath {
        member: &'a str,
        suffix: MatAssignmentSuffix,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum MatDecodeOutput {
    Workspace,
    Struct,
    SingleMember,
    RewriteInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct MatStructSelection<'a> {
    pub source: &'a str,
    pub fields: Option<&'a [String]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum MatEncodeMode<'a> {
    Save {
        append: bool,
        names: &'a [String],
        structs: &'a [MatStructSelection<'a>],
        patterns: &'a [String],
    },
    AssignMember {
        member: &'a str,
        suffix: MatAssignmentSuffix,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum MatAssignmentSuffix {
    Exact(Digest),
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub enum MatHandleControl {
    KnownWritable(bool),
    ResidentWritable(GpuTensorWireDescriptor),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum MatEncodeOutput {
    EffectOnly,
    Status,
    UpdatedObject,
}

impl MatEncodeOutput {
    pub const fn host_effect_output(self) -> HostEffectOutput {
        match self {
            Self::EffectOnly => HostEffectOutput::EffectOnly,
            Self::Status | Self::UpdatedObject => HostEffectOutput::ReturnedValue,
        }
    }
}

impl MatDecodeOutput {
    pub const fn host_effect_output(self) -> HostEffectOutput {
        match self {
            Self::Workspace | Self::RewriteInput => HostEffectOutput::EffectOnly,
            Self::Struct | Self::SingleMember => HostEffectOutput::ReturnedValue,
        }
    }
}
