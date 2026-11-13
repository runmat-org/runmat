//! Cooperative binary MAT records with no workspace or object restoration.

use super::{
    authority::MatFsPath,
    cooperation::MatCodecCooperation,
    encoding::{MatEncodingOwner, MatVar},
    format::{MatArray, MatClass, MatData},
};
use crate::{
    context::RuntimeContext,
    execution::{staged_file, CpuCooperationBudget},
    runtime_error::RuntimeErrorSourceBound,
    BuiltinResult, RuntimeError,
};
use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
use runmat_value::NumericStorage;
use std::io::BufReader;
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Clone, Copy)]
pub(crate) struct Policy {
    pub identity: &'static BuiltinCatalogIdentity,
    pub error: &'static BuiltinErrorDescriptor,
    pub record: &'static str,
}

pub(crate) struct RecordPath {
    source: Rc<MatFsPath>,
    policy: Policy,
}

/// An exclusive unpublished output tied to its exact captured path owner.
pub(crate) struct ReservedRecord {
    source: Rc<MatFsPath>,
    stage: Box<dyn runmat_filesystem::FsStagedOutput>,
}

impl RecordPath {
    /// Capture under the explicitly supplied invocation before any await.
    pub(crate) fn capture(
        mut path: PathBuf,
        runtime: &RuntimeContext,
        policy: Policy,
    ) -> BuiltinResult<Self> {
        crate::context::host_allocation::require_unbounded_workspace(
            Some(runtime),
            policy.identity.name,
            "binary MAT record encoding and parsing has no complete finite workspace bound",
        )?;
        let _entry = runtime.enter();
        if path.extension().is_none() {
            path.set_extension("mat");
        }
        let source = MatFsPath::capture(path)
            .map_err(|source| policy.with_source("cannot capture MAT record path", source))?;
        Ok(Self {
            source: Rc::new(source),
            policy,
        })
    }

    pub(crate) async fn reserve(
        &self,
        runtime: &RuntimeContext,
        budget: &mut CpuCooperationBudget,
    ) -> BuiltinResult<ReservedRecord> {
        runtime
            .scope(self.reserve_in(Some(runtime), Some(budget)))
            .await
    }

    async fn reserve_in(
        &self,
        runtime: Option<&RuntimeContext>,
        budget: Option<&mut CpuCooperationBudget>,
    ) -> BuiltinResult<ReservedRecord> {
        let mut cooperation = MatCodecCooperation::from_original_budget(
            runtime,
            budget,
            Some(&self.source),
            self.policy.identity.name,
            self.policy.error,
        )?;
        let stage = staged_file::begin_stage(
            &mut cooperation,
            || self.source.check_current(),
            || {
                self.source
                    .provider()
                    .begin_staged_replace(self.source.path())
            },
            "selected filesystem cannot stage an exclusive MAT record replacement",
        )
        .await
        .map_err(|source| self.staged_error(source))?;
        Ok(ReservedRecord {
            source: Rc::clone(&self.source),
            stage,
        })
    }

    pub(crate) async fn read(
        &self,
        runtime: &RuntimeContext,
        budget: &mut CpuCooperationBudget,
    ) -> BuiltinResult<Vec<u8>> {
        runtime
            .scope(self.read_in(Some(runtime), Some(budget)))
            .await
    }

    async fn read_in(
        &self,
        runtime: Option<&RuntimeContext>,
        budget: Option<&mut CpuCooperationBudget>,
    ) -> BuiltinResult<Vec<u8>> {
        let mut cooperation = MatCodecCooperation::from_original_budget(
            runtime,
            budget,
            Some(&self.source),
            self.policy.identity.name,
            self.policy.error,
        )?;
        let file = cooperation
            .await_io(self.source.open())
            .await?
            .map_err(|source| self.policy.with_source("cannot open MAT record", source))?;
        let mut reader = BufReader::new(file);
        let mut arrays =
            super::load::read_mat_arrays_cooperative(&mut reader, &mut cooperation).await?;
        cooperation.completed(0).await?;
        // Never run loadobj, legacy object-marker restoration or workspace
        // assignment while deciding whether this is a selected binary record.
        if arrays.len() != 1 {
            return Err(self
                .policy
                .error("MAT record must contain exactly one variable"));
        }
        let (name, array) = arrays.pop().expect("checked sole raw MAT array");
        if name != self.policy.record || array.class != MatClass::UInt8 {
            return Err(self
                .policy
                .error("MAT record has the wrong name or numeric class"));
        }
        let MatData::Numeric {
            real: NumericStorage::U8(bytes),
            imag: None,
        } = array.data
        else {
            return Err(self.policy.error("MAT record must have real uint8 storage"));
        };
        if array.dims.len() != 2 || array.dims[0] != 1 || array.dims[1] != bytes.len() {
            return Err(self.policy.error("MAT record must be a uint8 row vector"));
        }
        Ok(bytes)
    }

    pub(crate) async fn write_reserved(
        &self,
        reserved: ReservedRecord,
        record: Vec<u8>,
        runtime: &RuntimeContext,
        budget: &mut CpuCooperationBudget,
    ) -> BuiltinResult<()> {
        runtime
            .scope(self.write_reserved_in(reserved, record, Some(runtime), Some(budget)))
            .await
    }

    async fn write_reserved_in(
        &self,
        reserved: ReservedRecord,
        record: Vec<u8>,
        runtime: Option<&RuntimeContext>,
        budget: Option<&mut CpuCooperationBudget>,
    ) -> BuiltinResult<()> {
        if !Rc::ptr_eq(&self.source, &reserved.source) {
            return Err(self
                .policy
                .error("reserved MAT record belongs to another selected path"));
        }
        let mut cooperation = MatCodecCooperation::from_original_budget(
            runtime,
            budget,
            Some(&self.source),
            self.policy.identity.name,
            self.policy.error,
        )?;
        let bytes = {
            let vars = [MatVar {
                name: self.policy.record.to_owned(),
                array: MatArray {
                    class: MatClass::UInt8,
                    dims: vec![1, record.len()],
                    data: MatData::Numeric {
                        real: NumericStorage::U8(record),
                        imag: None,
                    },
                },
            }];
            super::encoding::write_mat_bytes_cooperative(
                &vars,
                MatEncodingOwner::BinaryRecord {
                    identity: self.policy.identity,
                    error: self.policy.error,
                },
                Some(&mut cooperation),
            )
            .await?
        };
        let mut stage = reserved.stage;
        staged_file::write_stage_bytes(
            runtime,
            &mut cooperation,
            self.policy.identity.name,
            || self.source.check_current(),
            stage.as_mut(),
            &bytes,
            staged_file::StagedWriteMessages {
                no_progress: "MAT record write made no progress",
                excess_progress: "MAT record writer reported excess byte progress",
            },
        )
        .await
        .map_err(|source| self.staged_error(source))?;
        staged_file::finish_stage(
            runtime,
            &mut cooperation,
            self.policy.identity.name,
            || self.source.check_current(),
            stage,
        )
        .await
        .map_err(|source| self.staged_error(source))
    }

    fn staged_error(&self, source: staged_file::StagedFileError) -> RuntimeError {
        match source {
            staged_file::StagedFileError::Control(error) => error,
            staged_file::StagedFileError::PublicationInterrupted(error, disposition) => {
                staged_file::interruption_error(error, disposition)
            }
            staged_file::StagedFileError::Io(source) => {
                self.policy.with_source("MAT record I/O failed", source)
            }
            staged_file::StagedFileError::Publication(source) => self
                .policy
                .with_source("MAT record publication failed", source),
        }
    }
}

impl Policy {
    fn error(self, message: &str) -> RuntimeError {
        let mut builder = crate::build_runtime_error(message)
            .with_builtin(self.identity.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
        if let Some(identifier) = self.error.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
    fn with_source(self, message: &str, source: impl RuntimeErrorSourceBound) -> RuntimeError {
        let mut builder = crate::build_runtime_error(message)
            .with_builtin(self.identity.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source);
        if let Some(identifier) = self.error.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

impl staged_file::StagedWriteCooperation for MatCodecCooperation<'_> {
    async fn completed(&mut self, work: usize) -> BuiltinResult<()> {
        MatCodecCooperation::completed(self, work).await
    }
}
