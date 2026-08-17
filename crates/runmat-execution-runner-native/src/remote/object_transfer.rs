use std::collections::HashMap;

use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef};
use runmat_execution::Digest;
use runmat_execution_transport_native::transfer::{ObjectChunk, ResumeState};

use super::protocol::{RemoteWorkerCommand, RemoteWorkerOutcome};
use super::RemoteObjectReceipt;
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) const MAX_REMOTE_OBJECT_BYTES: u64 = 64 * 1024 * 1024;
pub(super) const MAX_REMOTE_OBJECT_CHUNK_BYTES: usize = 256 * 1024;
pub(super) const MAX_REMOTE_OBJECTS: usize = 65_538;
pub(super) const MAX_REMOTE_OBJECT_TOTAL_BYTES: u64 = 256 * 1024 * 1024;

struct PartialObject {
    reference: ValueRef,
    resume: ResumeState,
    bytes: Vec<u8>,
}

#[derive(Default)]
pub(super) struct RemoteObjectCache {
    complete: HashMap<runmat_execution::identity::ValueId, (ValueRef, Vec<u8>)>,
    partial: HashMap<runmat_execution::identity::ValueId, PartialObject>,
    buffered_bytes: u64,
}

impl RemoteObjectCache {
    pub(super) fn probe(
        &mut self,
        reference: &ValueRef,
        authorization_scope: &str,
    ) -> NativeExecutionResult<RemoteObjectReceipt> {
        validate_reference(reference, authorization_scope)?;
        if let Some((stored, bytes)) = self.complete.get(&reference.id) {
            if stored != reference {
                return Err(protocol(
                    "remote object id was reused for a different reference",
                ));
            }
            return Ok(receipt(reference, bytes.len() as u64, true));
        }
        if let Some(partial) = self.partial.get(&reference.id) {
            if &partial.reference != reference {
                return Err(protocol("remote object id was reused during transfer"));
            }
            return Ok(receipt(reference, partial.resume.next_offset(), false));
        }
        if self.complete.len().saturating_add(self.partial.len()) >= MAX_REMOTE_OBJECTS {
            return Err(protocol("remote object inventory exceeds its count bound"));
        }
        self.partial.insert(
            reference.id,
            PartialObject {
                reference: reference.clone(),
                resume: ResumeState::new(reference.encoded_length).map_err(protocol)?,
                bytes: Vec::new(),
            },
        );
        Ok(receipt(reference, 0, false))
    }

    pub(super) fn put(
        &mut self,
        reference: &ValueRef,
        chunk: ObjectChunk,
        authorization_scope: &str,
    ) -> NativeExecutionResult<RemoteObjectReceipt> {
        if chunk.bytes.is_empty() || chunk.bytes.len() > MAX_REMOTE_OBJECT_CHUNK_BYTES {
            return Err(protocol(
                "remote object chunk is empty or exceeds its bound",
            ));
        }
        let position = self.probe(reference, authorization_scope)?;
        if position.complete {
            return Ok(position);
        }
        let partial = self
            .partial
            .get_mut(&reference.id)
            .expect("object probe created partial state");
        let next_total = self
            .buffered_bytes
            .checked_add(chunk.bytes.len() as u64)
            .ok_or_else(|| protocol("remote object inventory byte total overflowed"))?;
        if next_total > MAX_REMOTE_OBJECT_TOTAL_BYTES {
            return Err(protocol("remote object inventory exceeds its byte bound"));
        }
        partial
            .resume
            .accept(chunk.offset, chunk.bytes.len())
            .map_err(protocol)?;
        partial.bytes.extend_from_slice(&chunk.bytes);
        self.buffered_bytes = next_total;
        if !partial.resume.is_complete() {
            return Ok(receipt(reference, partial.resume.next_offset(), false));
        }
        let partial = self
            .partial
            .remove(&reference.id)
            .expect("completed partial object exists");
        if let Err(error) = validate_bytes(reference, &partial.bytes) {
            self.buffered_bytes = self
                .buffered_bytes
                .saturating_sub(partial.bytes.len() as u64);
            return Err(error);
        }
        let length = partial.bytes.len() as u64;
        self.complete
            .insert(reference.id, (partial.reference, partial.bytes));
        Ok(receipt(reference, length, true))
    }

    pub(super) fn get(
        &self,
        reference: &ValueRef,
        offset: u64,
        maximum_bytes: u32,
        authorization_scope: &str,
    ) -> NativeExecutionResult<(ObjectChunk, bool)> {
        validate_reference(reference, authorization_scope)?;
        let (stored, bytes) = self
            .complete
            .get(&reference.id)
            .ok_or_else(|| protocol("remote object is unavailable or incomplete"))?;
        if stored != reference || offset > reference.encoded_length {
            return Err(protocol(
                "remote object download reference or offset differs",
            ));
        }
        let maximum = usize::try_from(maximum_bytes)
            .unwrap_or(usize::MAX)
            .min(MAX_REMOTE_OBJECT_CHUNK_BYTES);
        if maximum == 0 {
            return Err(protocol("remote object download chunk bound is zero"));
        }
        let start = usize::try_from(offset)
            .map_err(|_| protocol("remote object offset does not fit this host"))?;
        let end = start.saturating_add(maximum).min(bytes.len());
        Ok((
            ObjectChunk {
                offset,
                bytes: bytes[start..end].to_vec(),
            },
            end == bytes.len(),
        ))
    }
}

pub(super) fn handle_command(
    cache: &mut RemoteObjectCache,
    command: RemoteWorkerCommand,
    authorization_scope: &str,
) -> RemoteWorkerOutcome {
    let result = match command {
        RemoteWorkerCommand::ProbeObject { reference } => cache
            .probe(&reference, authorization_scope)
            .map(|receipt| RemoteWorkerOutcome::ObjectPosition { receipt }),
        RemoteWorkerCommand::PutObjectChunk { reference, chunk } => cache
            .put(&reference, chunk, authorization_scope)
            .map(|receipt| RemoteWorkerOutcome::ObjectPosition { receipt }),
        RemoteWorkerCommand::GetObjectChunk {
            reference,
            offset,
            maximum_bytes,
        } => cache
            .get(&reference, offset, maximum_bytes, authorization_scope)
            .map(|(chunk, complete)| RemoteWorkerOutcome::ObjectChunk { chunk, complete }),
        _ => Err(protocol(
            "non-object command reached the object transfer owner",
        )),
    };
    result.unwrap_or_else(|error| RemoteWorkerOutcome::Rejected {
        message: error.to_string(),
    })
}

fn validate_reference(
    reference: &ValueRef,
    authorization_scope: &str,
) -> NativeExecutionResult<()> {
    ValuePayload::Object(Box::new(reference.clone()))
        .validate(ValueLimits::default())
        .map_err(protocol)?;
    if reference.authorization_scope != authorization_scope
        || reference.encoded_length == 0
        || reference.encoded_length > MAX_REMOTE_OBJECT_BYTES
    {
        return Err(protocol(
            "remote object reference is outside transfer authority or bounds",
        ));
    }
    Ok(())
}

fn validate_bytes(reference: &ValueRef, bytes: &[u8]) -> NativeExecutionResult<()> {
    if bytes.len() as u64 != reference.encoded_length
        || Digest::sha256(bytes) != reference.logical_digest
    {
        return Err(protocol(
            "remote object bytes differ from their logical identity",
        ));
    }
    Ok(())
}

fn receipt(reference: &ValueRef, next_offset: u64, complete: bool) -> RemoteObjectReceipt {
    RemoteObjectReceipt {
        value_id: reference.id,
        next_offset,
        complete,
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::ValueId;
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
    use runmat_execution::value::ValueRefKind;

    use super::*;

    fn reference(bytes: &[u8]) -> ValueRef {
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
            id: ValueId::derive(&[b"remote-object", bytes]),
            logical_digest: Digest::sha256(bytes),
            encoded_length: bytes.len() as u64,
            media_type: "application/vnd.runmat.test".into(),
            value_schema: "runmat.test.v1".into(),
            encryption_context: Digest::sha256(b"remote-object-context"),
            kind: ValueRefKind::ResultObject,
            authorization_scope: "run-object".into(),
            resident_fence: None,
        }
    }

    #[test]
    fn object_cache_resumes_and_rejects_corruption() {
        let bytes = b"canonical object bytes";
        let reference = reference(bytes);
        let mut cache = RemoteObjectCache::default();
        assert_eq!(
            cache.probe(&reference, "run-object").unwrap().next_offset,
            0
        );
        cache
            .put(
                &reference,
                ObjectChunk {
                    offset: 0,
                    bytes: bytes[..9].to_vec(),
                },
                "run-object",
            )
            .unwrap();
        assert_eq!(
            cache.probe(&reference, "run-object").unwrap().next_offset,
            9
        );
        assert!(cache
            .put(
                &reference,
                ObjectChunk {
                    offset: 8,
                    bytes: bytes[9..].to_vec(),
                },
                "run-object",
            )
            .is_err());
        let receipt = cache
            .put(
                &reference,
                ObjectChunk {
                    offset: 9,
                    bytes: bytes[9..].to_vec(),
                },
                "run-object",
            )
            .unwrap();
        assert!(receipt.complete);
        let (chunk, complete) = cache.get(&reference, 0, 10, "run-object").unwrap();
        assert_eq!(chunk.bytes, bytes[..10]);
        assert!(!complete);

        let mut corrupt = reference.clone();
        corrupt.logical_digest = Digest::sha256(b"substituted");
        assert!(cache.probe(&corrupt, "run-object").is_err());
        assert!(cache.probe(&reference, "another-run").is_err());
    }
}
