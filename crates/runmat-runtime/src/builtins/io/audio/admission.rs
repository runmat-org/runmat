//! Audio-specific facts passed to the shared complete-call host-effect selector.

use super::*;
use crate::context::host_effect::{
    select_host_effect, select_host_effect_after_stage, try_admit_host_effect_stage,
    AudioDecodeOutput, AudioFrameRange, HostEffectOutput, HostEffectReceipt, HostEffectRequest,
    HostEffectStagePermit, HostEffectWorkload,
};
use crate::context::RuntimeContext;
use runmat_execution::{Digest, WholeWorkObservationOutcome};

mod geometry;
pub(super) use geometry::AudioDecodeGeometry;

pub(super) fn select_decode(
    runtime: Option<&RuntimeContext>,
    input: &FsInputLease,
    options: AudioreadOptions,
    outputs: Option<usize>,
) -> BuiltinResult<Option<HostEffectReceipt>> {
    select_host_effect(runtime, decode_request(input, options, outputs, None)?)
}

pub(super) fn needs_decode_stage(runtime: Option<&RuntimeContext>) -> bool {
    runtime.is_some_and(|runtime| {
        runtime.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        })
    })
}

pub(super) fn admit_decode_stage(
    runtime: &RuntimeContext,
    input: &FsInputLease,
    options: AudioreadOptions,
    outputs: Option<usize>,
) -> BuiltinResult<Option<HostEffectStagePermit>> {
    try_admit_host_effect_stage(runtime, decode_request(input, options, outputs, None)?)
}

pub(super) fn select_decode_after_stage(
    runtime: &RuntimeContext,
    input: &FsInputLease,
    options: AudioreadOptions,
    outputs: Option<usize>,
    geometry: &AudioDecodeGeometry,
    stage: HostEffectStagePermit,
) -> BuiltinResult<HostEffectReceipt> {
    let facts = geometry.resource_facts();
    select_host_effect_after_stage(
        runtime,
        decode_request(input, options, outputs, Some(*geometry))?,
        crate::context::host_effect::HostEffectFinalFacts::AudioDecode(&facts),
        stage,
    )
}

fn decode_request<'a>(
    input: &'a FsInputLease,
    options: AudioreadOptions,
    outputs: Option<usize>,
    geometry: Option<AudioDecodeGeometry>,
) -> BuiltinResult<HostEffectRequest<'a>> {
    let output = match outputs {
        Some(0) => AudioDecodeOutput::EffectOnly,
        Some(2) => AudioDecodeOutput::SamplesAndRate,
        None | Some(1) => AudioDecodeOutput::Samples,
        Some(_) => unreachable!("output arity was admitted before filesystem capture"),
    };
    let range = options
        .range
        .map(|range| -> BuiltinResult<AudioFrameRange> {
            Ok(AudioFrameRange {
                first: u64::try_from(range.first).map_err(|_| {
                    audioread_error_with(
                        &AUDIOREAD_ERROR_ARGUMENT,
                        "audioread: sample range is too large",
                    )
                })?,
                last: range.last.map(u64::try_from).transpose().map_err(|_| {
                    audioread_error_with(
                        &AUDIOREAD_ERROR_ARGUMENT,
                        "audioread: sample range is too large",
                    )
                })?,
            })
        })
        .transpose()?;
    let workload = HostEffectWorkload::AudioDecode {
        source: input.path.as_path(),
        range,
        datatype: options.datatype,
        output,
    };
    Ok(HostEffectRequest {
        operation: AUDIOREAD_BUILTIN_NAME,
        workload,
        domain_facts: geometry.map_or_else(
            || Digest::sha256(workload.identity_bytes()),
            AudioDecodeGeometry::digest,
        ),
        filesystem: &input.provider,
    })
}

pub(super) fn select_metadata(
    runtime: Option<&RuntimeContext>,
    input: &FsInputLease,
    outputs: Option<usize>,
) -> BuiltinResult<Option<HostEffectReceipt>> {
    let output = match outputs {
        Some(0) => HostEffectOutput::EffectOnly,
        None | Some(1) => HostEffectOutput::ReturnedValue,
        Some(_) => unreachable!("output arity was admitted before filesystem capture"),
    };
    let workload = HostEffectWorkload::AudioMetadata {
        source: input.path.as_path(),
        output,
    };
    select_host_effect(
        runtime,
        HostEffectRequest {
            operation: AUDIOINFO_BUILTIN_NAME,
            workload,
            domain_facts: Digest::sha256(workload.identity_bytes()),
            filesystem: &input.provider,
        },
    )
}

pub(super) fn finish_receipt<T>(
    receipt: Option<HostEffectReceipt>,
    result: &BuiltinResult<T>,
    cancelled: bool,
) {
    if let Some(receipt) = receipt {
        receipt.finish(if cancelled {
            WholeWorkObservationOutcome::Cancelled
        } else if result.is_ok() {
            WholeWorkObservationOutcome::Success
        } else {
            WholeWorkObservationOutcome::OperationalFailure
        });
    }
}
