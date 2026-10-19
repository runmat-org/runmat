use std::collections::{HashMap, HashSet};
use std::io::{Seek, Write};
use std::sync::{Arc, Mutex};

use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::{CacheExport, CacheImport, FilesystemObjectStore};
use runmat_execution_artifact::{
    ArtifactError, ArtifactResult, LogicalObject, ObjectNamespace, ProgramExecutionObject,
};
use runmat_execution_transport_native::transfer::{ObjectChunk, ResumeState};

use super::protocol::{RemoteWorkerCommand, RemoteWorkerOutcome};
use super::RemoteObjectReceipt;
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) const MAX_REMOTE_OBJECT_BYTES: u64 =
    runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_BYTES;
pub(super) const MAX_REMOTE_OBJECT_CHUNK_BYTES: usize = 256 * 1024;
pub(super) const MAX_REMOTE_OBJECTS: usize =
    runmat_execution_artifact::MAX_PROGRAM_EXECUTION_RESULT_OBJECTS;

struct PartialObject {
    reference: ValueRef,
    resume: ResumeState,
    file: tempfile::NamedTempFile,
}

pub(super) struct RemoteObjectCache {
    complete: HashMap<runmat_execution::identity::ValueId, ValueRef>,
    partial: HashMap<runmat_execution::identity::ValueId, PartialObject>,
    generated: HashSet<Digest>,
    stored_bytes: u64,
    store: FilesystemObjectStore,
    _directory: tempfile::TempDir,
}

impl RemoteObjectCache {
    fn new() -> NativeExecutionResult<Self> {
        let directory = tempfile::Builder::new()
            .prefix("runmat-remote-worker-objects-")
            .tempdir()
            .map_err(protocol)?;
        let store =
            FilesystemObjectStore::open(directory.path().join("objects"), MAX_REMOTE_OBJECT_BYTES)
                .map_err(protocol)?;
        Ok(Self {
            complete: HashMap::new(),
            partial: HashMap::new(),
            generated: HashSet::new(),
            stored_bytes: 0,
            store,
            _directory: directory,
        })
    }
}

#[derive(Clone)]
pub(super) struct RemoteObjectStore(Arc<Mutex<RemoteObjectCache>>);

impl RemoteObjectStore {
    pub(super) fn new() -> NativeExecutionResult<Self> {
        Ok(Self(Arc::new(Mutex::new(RemoteObjectCache::new()?))))
    }

    pub(super) fn register(&self, object: ProgramExecutionObject) -> NativeExecutionResult<()> {
        object.validate().map_err(protocol)?;
        self.0
            .lock()
            .expect("remote object store poisoned")
            .register(object.reference, object.bytes)
    }

    pub(super) fn object(
        &self,
        reference: &ValueRef,
        authorization_scope: &str,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        validate_reference(reference, authorization_scope)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
        self.read_verified(reference.logical_digest)
    }

    pub(super) fn handle(
        &self,
        command: RemoteWorkerCommand,
        authorization_scope: &str,
    ) -> RemoteWorkerOutcome {
        handle_command(
            &mut self.0.lock().expect("remote object store poisoned"),
            command,
            authorization_scope,
        )
    }
}

impl CacheImport for RemoteObjectStore {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> runmat_execution_artifact::ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.0
            .lock()
            .expect("remote object store poisoned")
            .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.0
            .lock()
            .expect("remote object store poisoned")
            .read_verified_bounded(digest, maximum_bytes)
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.0
            .lock()
            .expect("remote object store poisoned")
            .read_verified(digest)
    }
}

impl CacheExport for RemoteObjectStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        self.0
            .lock()
            .expect("remote object store poisoned")
            .write_verified(object)
    }
}

impl RemoteObjectCache {
    fn register(&mut self, reference: ValueRef, bytes: Vec<u8>) -> NativeExecutionResult<()> {
        validate_reference(&reference, &reference.authorization_scope)?;
        validate_bytes(&reference, &bytes)?;
        if let Some(stored) = self.complete.get(&reference.id) {
            if !references_share_content_identity(stored, &reference) {
                return Err(protocol(
                    "remote object id was reused for different content",
                ));
            }
            return store_reference(&mut self.store, &reference, bytes);
        }
        let next_total = self.extended_total(reference.encoded_length)?;
        store_reference(&mut self.store, &reference, bytes)?;
        self.complete.insert(reference.id, reference);
        self.stored_bytes = next_total;
        Ok(())
    }

    pub(super) fn probe(
        &mut self,
        reference: &ValueRef,
        authorization_scope: &str,
    ) -> NativeExecutionResult<RemoteObjectReceipt> {
        validate_reference(reference, authorization_scope)?;
        if let Some(stored) = self.complete.get(&reference.id) {
            if !references_share_content_identity(stored, reference) {
                return Err(protocol(
                    "remote object id was reused for a different reference",
                ));
            }
            return Ok(receipt(reference, reference.encoded_length, true));
        }
        if let Some(partial) = self.partial.get(&reference.id) {
            if !references_share_content_identity(&partial.reference, reference) {
                return Err(protocol("remote object id was reused during transfer"));
            }
            return Ok(receipt(reference, partial.resume.next_offset(), false));
        }
        let next_total = self.extended_total(reference.encoded_length)?;
        let resume = ResumeState::new(reference.encoded_length).map_err(protocol)?;
        let file = tempfile::NamedTempFile::new_in(self.store.root()).map_err(protocol)?;
        self.partial.insert(
            reference.id,
            PartialObject {
                reference: reference.clone(),
                resume,
                file,
            },
        );
        self.stored_bytes = next_total;
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
        partial
            .resume
            .accept(chunk.offset, chunk.bytes.len())
            .map_err(protocol)?;
        partial.file.write_all(&chunk.bytes).map_err(protocol)?;
        if !partial.resume.is_complete() {
            return Ok(receipt(reference, partial.resume.next_offset(), false));
        }
        let mut partial = self
            .partial
            .remove(&reference.id)
            .expect("completed partial object exists");
        partial.file.as_file_mut().sync_all().map_err(protocol)?;
        partial.file.rewind().map_err(protocol)?;
        let mut bytes = Vec::with_capacity(
            usize::try_from(reference.encoded_length)
                .map_err(|_| protocol("remote object length does not fit this host"))?,
        );
        std::io::Read::read_to_end(partial.file.as_file_mut(), &mut bytes).map_err(protocol)?;
        if let Err(error) = validate_bytes(reference, &bytes) {
            self.stored_bytes = self.stored_bytes.saturating_sub(reference.encoded_length);
            return Err(error);
        }
        if let Err(error) = store_reference(&mut self.store, reference, bytes) {
            self.stored_bytes = self.stored_bytes.saturating_sub(reference.encoded_length);
            return Err(error);
        }
        self.complete.insert(reference.id, partial.reference);
        Ok(receipt(reference, reference.encoded_length, true))
    }

    pub(super) fn get(
        &self,
        reference: &ValueRef,
        offset: u64,
        maximum_bytes: u32,
        authorization_scope: &str,
    ) -> NativeExecutionResult<(ObjectChunk, bool)> {
        validate_reference(reference, authorization_scope)?;
        match self.complete.get(&reference.id) {
            Some(stored) if references_share_content_identity(stored, reference) => {}
            Some(_) => {
                return Err(protocol(
                    "remote object download reference differs from stored content",
                ))
            }
            None if self.generated.contains(&reference.logical_digest) => {}
            None => return Err(protocol("remote object is unavailable or incomplete")),
        }
        let bytes = self
            .store
            .read_verified(reference.logical_digest)
            .map_err(protocol)?
            .ok_or_else(|| protocol("remote object bytes are absent"))?;
        if bytes.len() as u64 != reference.encoded_length
            || Digest::sha256(&bytes) != reference.logical_digest
            || offset > reference.encoded_length
        {
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

impl CacheImport for RemoteObjectCache {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> runmat_execution_artifact::ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.store
            .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified_bounded(digest, maximum_bytes)
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified(digest)
    }
}

impl CacheExport for RemoteObjectCache {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        if object.descriptor.encoded_length == 0
            || object.descriptor.encoded_length > MAX_REMOTE_OBJECT_BYTES
        {
            return Err(ArtifactError::Limit(
                "remote generated object exceeds its per-object bound".into(),
            ));
        }
        if self.generated.contains(&object.descriptor.digest)
            || self
                .complete
                .values()
                .any(|reference| reference.logical_digest == object.descriptor.digest)
        {
            return self.store.write_verified(object);
        }
        let next_total = self
            .extended_total(object.descriptor.encoded_length)
            .map_err(|error| ArtifactError::Limit(error.to_string()))?;
        self.store.write_verified(object)?;
        self.generated.insert(object.descriptor.digest);
        self.stored_bytes = next_total;
        Ok(())
    }
}

impl RemoteObjectCache {
    fn extended_total(&self, bytes: u64) -> NativeExecutionResult<u64> {
        if self
            .complete
            .len()
            .saturating_add(self.partial.len())
            .saturating_add(self.generated.len())
            >= MAX_REMOTE_OBJECTS
        {
            return Err(protocol("remote object inventory exceeds its count bound"));
        }
        let next_total = self
            .stored_bytes
            .checked_add(bytes)
            .ok_or_else(|| protocol("remote object inventory byte total overflowed"))?;
        if next_total > runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES {
            return Err(protocol("remote object inventory exceeds its byte bound"));
        }
        Ok(next_total)
    }
}

fn store_reference(
    store: &mut FilesystemObjectStore,
    reference: &ValueRef,
    bytes: Vec<u8>,
) -> NativeExecutionResult<()> {
    let namespace = match reference.kind {
        runmat_execution::value::ValueRefKind::DriverObject => ObjectNamespace::InputValue,
        runmat_execution::value::ValueRefKind::ResultObject => ObjectNamespace::ResultValue,
        _ => return Err(protocol("remote execution object has a non-portable kind")),
    };
    let object = LogicalObject::new(
        namespace,
        reference.value_schema.clone(),
        reference.media_type.clone(),
        bytes,
    )
    .map_err(protocol)?;
    if object.descriptor.digest != reference.logical_digest
        || object.descriptor.encoded_length != reference.encoded_length
    {
        return Err(protocol(
            "remote execution object differs from its logical descriptor",
        ));
    }
    store.write_verified(&object).map_err(protocol)
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
        .validate_for_transport(
            ValueLimits::default(),
            runmat_execution::value::ValueTransportContext::Portable,
        )
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

pub(super) fn references_share_content_identity(left: &ValueRef, right: &ValueRef) -> bool {
    use runmat_execution::value::ValueRefKind::{DriverObject, ResultObject};

    matches!(left.kind, DriverObject | ResultObject)
        && matches!(right.kind, DriverObject | ResultObject)
        && left.schema_version == right.schema_version
        && left.id == right.id
        && left.logical_digest == right.logical_digest
        && left.encoded_length == right.encoded_length
        && left.media_type == right.media_type
        && left.value_schema == right.value_schema
        && left.encryption_context == right.encryption_context
        && left.authorization_scope == right.authorization_scope
        && left.resident_fence == right.resident_fence
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
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
    use runmat_execution::value::ValueRefKind;

    use super::*;

    fn reference(bytes: &[u8]) -> ValueRef {
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
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
    fn object_cache_resumes_from_disk_and_rejects_corruption() {
        let bytes = b"canonical object bytes";
        let reference = reference(bytes);
        let mut cache = RemoteObjectCache::new().unwrap();
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
        assert_eq!(
            cache
                .partial
                .get(&reference.id)
                .unwrap()
                .file
                .as_file()
                .metadata()
                .unwrap()
                .len(),
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
        assert!(cache.partial.is_empty());
        assert_eq!(cache.complete.get(&reference.id), Some(&reference));
        let (chunk, complete) = cache.get(&reference, 0, 10, "run-object").unwrap();
        assert_eq!(chunk.bytes, bytes[..10]);
        assert!(!complete);

        let mut driver_alias = reference.clone();
        driver_alias.kind = ValueRefKind::DriverObject;
        assert!(cache.probe(&driver_alias, "run-object").unwrap().complete);
        assert_eq!(
            cache
                .get(&driver_alias, 0, bytes.len() as u32, "run-object")
                .unwrap()
                .0
                .bytes,
            bytes
        );

        let mut corrupt = reference.clone();
        corrupt.logical_digest = Digest::sha256(b"substituted");
        assert!(cache.probe(&corrupt, "run-object").is_err());
        let mut substituted_schema = driver_alias;
        substituted_schema.value_schema = "runmat.substituted.v1".into();
        assert!(cache.probe(&substituted_schema, "run-object").is_err());
        assert!(cache.probe(&reference, "another-run").is_err());
    }

    #[test]
    fn generated_objects_use_the_same_verified_download_store() {
        let bytes = b"generated canonical bytes".to_vec();
        let object = LogicalObject::new(
            runmat_execution_artifact::ObjectNamespace::ResultValue,
            "remote/generated",
            "application/vnd.runmat.test",
            bytes.clone(),
        )
        .unwrap();
        let reference = reference(&bytes);
        let mut store = RemoteObjectStore::new().unwrap();
        store.write_verified(&object).unwrap();
        assert_eq!(
            store.read_verified(object.descriptor.digest).unwrap(),
            Some(bytes.clone())
        );
        let outcome = store.handle(
            RemoteWorkerCommand::GetObjectChunk {
                reference,
                offset: 0,
                maximum_bytes: 1024,
            },
            "run-object",
        );
        assert!(matches!(
            outcome,
            RemoteWorkerOutcome::ObjectChunk { chunk, complete }
                if chunk.bytes == bytes && complete
        ));
    }

    #[test]
    fn encrypted_input_inventory_seeds_the_verified_execution_store() {
        let bytes = b"transported analysis manifest".to_vec();
        let mut reference = reference(&bytes);
        reference.kind = ValueRefKind::DriverObject;
        let store = RemoteObjectStore::new().unwrap();
        store
            .register(ProgramExecutionObject {
                reference: reference.clone(),
                bytes: bytes.clone(),
            })
            .unwrap();
        assert_eq!(
            store.read_verified(reference.logical_digest).unwrap(),
            Some(bytes)
        );
        assert!(store
            .register(ProgramExecutionObject {
                reference,
                bytes: b"substituted".to_vec(),
            })
            .is_err());
    }
}

#[cfg(test)]
mod read_admission_tests;
