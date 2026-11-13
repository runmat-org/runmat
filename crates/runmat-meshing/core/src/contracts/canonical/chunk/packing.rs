//! Walk the deterministic packing layout without allocating fragment metadata or payloads.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Fragments<'a> {
    pub records: &'a [Vec<u8>],
    pub fragment: Option<codec::MeshingRecordFragment<'a>>,
}
impl<'a> Fragments<'a> {
    pub fn iter(self) -> impl Iterator<Item = codec::MeshingRecordFragment<'a>> {
        self.records
            .iter()
            .map(|record| codec::MeshingRecordFragment {
                bytes: record,
                completes_record: true,
            })
            .chain(self.fragment)
    }
    pub fn len(self) -> usize {
        self.records.len() + usize::from(self.fragment.is_some())
    }
}

pub(super) fn visit(
    streams: &[MeshingChunkStream],
    policy: MeshingChunkPolicy,
    cancellation: &dyn super::super::MeshingCancellationSignal,
    mut consume: impl FnMut(
        u32,
        u64,
        &MeshingChunkStream,
        Fragments<'_>,
    ) -> Result<(), MeshingContractError>,
) -> Result<(), MeshingContractError> {
    policy.validate()?;
    validate_streams(streams)?;
    let payload_budget = policy.maximum_chunk_bytes as usize - CHUNK_OVERHEAD_RESERVE;
    let mut ordinal = 0_u32;
    let mut first = 0_u64;
    let mut emit = |stream: &MeshingChunkStream, fragments: Fragments<'_>| {
        if fragments.len() == 0 {
            return Ok(());
        }
        cancellation_checkpoint(cancellation)?;
        if ordinal as usize >= MAX_CHUNKS {
            return Err(MeshingContractError::invalid(
                "meshing chunks",
                "chunk count exceeds the manifest limit",
            ));
        }
        consume(ordinal, first, stream, fragments)?;
        ordinal += 1;
        first = first
            .checked_add(
                fragments
                    .iter()
                    .filter(|fragment| fragment.completes_record)
                    .count() as u64,
            )
            .ok_or_else(|| {
                MeshingContractError::invalid("meshing chunks", "entity ordinal overflow")
            })?;
        Ok(())
    };
    for stream in streams {
        let mut start = 0;
        let mut pending_bytes = 0_usize;
        for (index, record) in stream.records.iter().enumerate() {
            cancellation_checkpoint(cancellation)?;
            let cost = encoded_fragment_cost(record.len());
            if cost > payload_budget {
                emit(
                    stream,
                    Fragments {
                        records: &stream.records[start..index],
                        fragment: None,
                    },
                )?;
                let maximum_fragment_bytes = payload_budget - 16;
                let fragment_count = record.len().div_ceil(maximum_fragment_bytes);
                for (index, bytes) in record.chunks(maximum_fragment_bytes).enumerate() {
                    emit(
                        stream,
                        Fragments {
                            records: &[],
                            fragment: Some(codec::MeshingRecordFragment {
                                bytes,
                                completes_record: index + 1 == fragment_count,
                            }),
                        },
                    )?;
                }
                start = index + 1;
                pending_bytes = 0;
            } else {
                if index > start
                    && (index - start >= policy.maximum_records_per_chunk as usize
                        || pending_bytes.saturating_add(cost) > payload_budget)
                {
                    emit(
                        stream,
                        Fragments {
                            records: &stream.records[start..index],
                            fragment: None,
                        },
                    )?;
                    start = index;
                    pending_bytes = 0;
                }
                pending_bytes = pending_bytes.checked_add(cost).ok_or_else(|| {
                    MeshingContractError::invalid("meshing chunks", "chunk size overflow")
                })?;
            }
        }
        emit(
            stream,
            Fragments {
                records: &stream.records[start..],
                fragment: None,
            },
        )?;
    }
    Ok(())
}
