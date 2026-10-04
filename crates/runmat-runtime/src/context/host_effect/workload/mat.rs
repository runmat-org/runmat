//! Closed MAT decode identity. The MAT domain has already parsed these
//! selectors; placement never interprets option spelling or regex syntax.

use super::HostEffectOutput;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum MatDecodeSelection<'a> {
    All,
    Selected {
        names: &'a [String],
        patterns: &'a [String],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum MatDecodeOutput {
    Workspace,
    Struct,
    SingleMember,
    RewriteInput,
}

impl MatDecodeOutput {
    pub const fn host_effect_output(self) -> HostEffectOutput {
        match self {
            Self::Workspace | Self::RewriteInput => HostEffectOutput::EffectOnly,
            Self::Struct | Self::SingleMember => HostEffectOutput::ReturnedValue,
        }
    }
}
