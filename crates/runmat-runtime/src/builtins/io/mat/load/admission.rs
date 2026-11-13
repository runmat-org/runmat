//! One complete load receipt, retained through workspace or value publication.

use super::{LoadRequest, MatFsPath};
use crate::context::host_effect::{
    HostEffectReceipt, HostEffectWorkload, MatDecodeOutput, MatDecodeSelection,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

pub(super) fn select(
    runtime: Option<&RuntimeContext>,
    source: &MatFsPath,
    request: &LoadRequest,
    output: MatDecodeOutput,
) -> BuiltinResult<Option<HostEffectReceipt>> {
    let selection = if request.variables.is_empty() && request.regex_tokens.is_empty() {
        MatDecodeSelection::All
    } else {
        MatDecodeSelection::Selected {
            names: &request.variables,
            patterns: &request.regex_tokens,
        }
    };
    let workload = HostEffectWorkload::MatDecode {
        source: source.path(),
        selection,
        output,
    };
    super::super::admission::select(runtime, source, "load", workload)
}

pub(super) fn finish<T>(receipt: Option<HostEffectReceipt>, result: &BuiltinResult<T>) {
    super::super::admission::finish(receipt, result)
}

pub(super) fn finish_error(receipt: Option<HostEffectReceipt>, error: &RuntimeError) {
    super::super::admission::finish_error(receipt, error)
}
