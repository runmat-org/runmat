//! Exact output allocation for a finally admitted floating WAVE route.

use super::*;

/// Zero-output calls still scan and read the selected samples, but do not
/// construct a tensor or demand a retained-output allocation permit.
pub(in crate::builtins::io::audio) async fn validate_prepared_wave_stream_discarded(
    mut prepared: PreparedWaveStream,
    path: &Path,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<()> {
    let facts = prepared.geometry;
    if facts.input_chunk_bytes > INPUT_CHUNK_BYTES {
        return Err(finite_decode_unavailable("input chunk allocation"));
    }
    let geometry = StreamGeometry {
        data_offset: facts.data_offset,
        channels: facts.channels,
        start: facts.start,
        end: facts.end,
        block_align: facts.block_align,
        bytes_per_channel: facts.bytes_per_channel,
    };
    let mut input_chunk = [0u8; INPUT_CHUNK_BYTES];
    validate_stream_samples(
        &mut prepared.file,
        path,
        geometry,
        cooperation,
        &mut input_chunk[..facts.input_chunk_bytes],
        facts.coding,
    )
    .await?;
    cooperation.completed(0).await
}

pub(in crate::builtins::io::audio) async fn decode_prepared_wave_stream_exact(
    mut prepared: PreparedWaveStream,
    path: &Path,
    cooperation: &mut AudioCooperation,
    permit: &dyn runmat_value::ExactHostAllocationPermit,
) -> BuiltinResult<DecodedAudio> {
    let facts = prepared.geometry;
    if !matches!(facts.output_dtype, NumericDType::F32 | NumericDType::F64)
        || facts.input_chunk_bytes > INPUT_CHUNK_BYTES
    {
        return Err(finite_decode_unavailable(
            "unsupported output or input chunk allocation",
        ));
    }
    cooperation.completed(0).await?;
    let mut allocation = runmat_value::ExactHostNumericAllocation::new(
        facts.output_dtype,
        facts.sample_count,
        permit,
    )
    .map_err(|source| {
        build_runtime_error(format!(
            "audioread: exact output allocation failed: {source}"
        ))
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .with_source(source)
        .build()
    })?;
    let geometry = StreamGeometry {
        data_offset: facts.data_offset,
        channels: facts.channels,
        start: facts.start,
        end: facts.end,
        block_align: facts.block_align,
        bytes_per_channel: facts.bytes_per_channel,
    };
    let mut input_chunk = [0u8; INPUT_CHUNK_BYTES];
    match facts.output_dtype {
        NumericDType::F32 => {
            let output = allocation.as_f32_slice_mut().expect("F32 allocation");
            fill_stream_samples(
                &mut prepared.file,
                path,
                geometry,
                cooperation,
                output,
                &mut input_chunk[..facts.input_chunk_bytes],
                |sample| f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]),
            )
            .await?;
        }
        NumericDType::F64 => {
            let output = allocation.as_f64_slice_mut().expect("F64 allocation");
            fill_stream_samples(
                &mut prepared.file,
                path,
                geometry,
                cooperation,
                output,
                &mut input_chunk[..facts.input_chunk_bytes],
                |sample| {
                    decode_wave_sample(
                        sample,
                        facts.coding,
                        facts.datatype == AudioReadDataType::Native,
                    )
                },
            )
            .await?;
        }
        _ => unreachable!("floating output checked above"),
    }
    let samples = Tensor::from_host_buffer(
        allocation.into_host_buffer(),
        vec![facts.end - facts.start, facts.channels],
    )
    .map_err(|err| format_error(format!("decoded audio tensor shape is invalid: {err}")))?;
    cooperation.completed(0).await?;
    Ok(DecodedAudio {
        samples,
        sample_rate: f64::from(prepared.parsed.format.sample_rate),
    })
}

fn finite_decode_unavailable(detail: &str) -> RuntimeError {
    build_runtime_error(format!(
        "audioread: finite decode evidence unavailable: {detail}"
    ))
    .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
    .build()
}
