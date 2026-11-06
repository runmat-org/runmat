//! Owned kernel output forms and the bridge's effective output policy.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisPayloadPartition, DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_execution::Digest;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisOutputPolicy {
    chunk_bytes: usize,
    inventory_limits: ObjectInventoryLimits,
}

impl AnalysisOutputPolicy {
    pub fn validate(self) -> AnalysisBridgeResult<()> {
        if Self::new(self.chunk_bytes, self.inventory_limits)? != self {
            return Err(AnalysisBridgeError::Invalid(
                "noncanonical analysis output policy".into(),
            ));
        }
        Ok(())
    }

    /// The invocation chooses transport; placement supplies only upper limits. The chunk size
    /// is part of the transport identity and must fit the worker. The inventory is the
    /// intersection: a study-wide output policy is admitted by the run budget, while a reserved
    /// worker holds only what its task reservation admits, so the worker enforces the smaller
    /// of the two and an output past it fails with a typed capacity error.
    pub fn bounded_by(self, ceiling: Self) -> AnalysisBridgeResult<Self> {
        self.validate()?;
        ceiling.validate()?;
        if self.chunk_bytes > ceiling.chunk_bytes {
            return Err(AnalysisBridgeError::Invalid(
                "analysis invocation chunk size exceeds worker limits".into(),
            ));
        }
        let actual = self.inventory_limits;
        let maximum = ceiling.inventory_limits;
        Self::new(
            self.chunk_bytes,
            ObjectInventoryLimits {
                max_objects: actual.max_objects.min(maximum.max_objects),
                max_object_bytes: actual.max_object_bytes.min(maximum.max_object_bytes),
                max_total_bytes: actual.max_total_bytes.min(maximum.max_total_bytes),
            },
        )
    }
    pub fn new(
        chunk_bytes: usize,
        inventory_limits: ObjectInventoryLimits,
    ) -> AnalysisBridgeResult<Self> {
        if chunk_bytes == 0
            || inventory_limits.max_objects == 0
            || inventory_limits.max_object_bytes == 0
            || inventory_limits.max_total_bytes == 0
        {
            return Err(AnalysisBridgeError::Invalid(
                "analysis execution limits must be non-zero".into(),
            ));
        }
        Ok(Self {
            chunk_bytes: chunk_bytes.min(DEFAULT_ANALYSIS_CHUNK_BYTES),
            inventory_limits,
        })
    }
    pub fn chunk_bytes(self) -> usize {
        self.chunk_bytes
    }
    pub fn inventory_limits(self) -> ObjectInventoryLimits {
        self.inventory_limits
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnalysisKernelPayload {
    Whole(Vec<u8>),
    Partitioned(Vec<AnalysisKernelOutputPartition>),
    Prepared(Box<PreparedAnalysisObjects>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisKernelOutput {
    pub kind: AnalysisNumericalObjectKind,
    pub logical_identity: Digest,
    pub payload: AnalysisKernelPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisKernelOutputPartition {
    pub logical_identity: Digest,
    pub payload: Vec<u8>,
}

impl AnalysisKernelOutput {
    pub fn whole(
        kind: AnalysisNumericalObjectKind,
        logical_identity: Digest,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            kind,
            logical_identity,
            payload: AnalysisKernelPayload::Whole(payload),
        }
    }
    pub fn partitioned(
        kind: AnalysisNumericalObjectKind,
        logical_identity: Digest,
        partitions: Vec<AnalysisKernelOutputPartition>,
    ) -> Self {
        Self {
            kind,
            logical_identity,
            payload: AnalysisKernelPayload::Partitioned(partitions),
        }
    }
    pub fn prepared(objects: PreparedAnalysisObjects) -> Self {
        Self {
            kind: objects.manifest.kind,
            logical_identity: objects.manifest.logical_identity,
            payload: AnalysisKernelPayload::Prepared(Box::new(objects)),
        }
    }
    /// Borrow only an explicitly whole result; never gather a partitioned or prepared field.
    pub fn whole_payload(&self) -> AnalysisBridgeResult<&[u8]> {
        match &self.payload {
            AnalysisKernelPayload::Whole(bytes) => Ok(bytes),
            _ => Err(AnalysisBridgeError::Invalid(
                "kernel output is not a whole payload".into(),
            )),
        }
    }
    pub(super) fn is_valid(&self) -> bool {
        match &self.payload {
            AnalysisKernelPayload::Whole(bytes) => !bytes.is_empty(),
            AnalysisKernelPayload::Partitioned(partitions) => {
                !partitions.is_empty() && partitions.iter().all(|p| !p.payload.is_empty())
            }
            AnalysisKernelPayload::Prepared(objects) => {
                self.kind == objects.manifest.kind
                    && self.logical_identity == objects.manifest.logical_identity
            }
        }
    }
    /// Consume prepared buffers without copying, gathering or rechunking; every closure is
    /// validated before the serial bridge can publish any output.
    pub fn into_prepared(
        self,
        chunk_bytes: usize,
        limits: ObjectInventoryLimits,
    ) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        let policy = AnalysisOutputPolicy::new(chunk_bytes, limits)?;
        if !self.is_valid() {
            return Err(AnalysisBridgeError::Invalid(
                "invalid kernel output payload".into(),
            ));
        }
        match self.payload {
            AnalysisKernelPayload::Whole(bytes) => crate::prepare_analysis_objects(
                self.kind,
                self.logical_identity,
                &bytes,
                policy.chunk_bytes(),
                ObjectNamespace::ResultValue,
                limits,
            ),
            AnalysisKernelPayload::Partitioned(partitions) => {
                let borrowed = partitions
                    .iter()
                    .map(|p| AnalysisPayloadPartition {
                        logical_identity: p.logical_identity,
                        payload: &p.payload,
                    })
                    .collect::<Vec<_>>();
                crate::prepare_partitioned_analysis_objects(
                    self.kind,
                    self.logical_identity,
                    &borrowed,
                    policy.chunk_bytes(),
                    ObjectNamespace::ResultValue,
                    limits,
                )
            }
            AnalysisKernelPayload::Prepared(objects) => {
                objects.validate_result_output(policy.chunk_bytes(), limits)?;
                Ok(*objects)
            }
        }
    }
}

#[cfg(test)]
mod tests;
