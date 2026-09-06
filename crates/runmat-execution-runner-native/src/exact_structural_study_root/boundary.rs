use std::collections::BTreeMap;

use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::value::ValueRefKind;
use runmat_execution::Digest;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_geometry_core::PersistentEntityId;
use serde::{Deserialize, Serialize};

use super::{protocol, BOUNDARY_DOMAIN, BOUNDARY_PREFIX, CONTRACT_LIMITS};
use crate::NativeExecutionResult;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryRegionResolution {
    pub schema_version: u16,
    pub geometry_source_digest: Digest,
    pub solver_mesh_digest: Digest,
    pub aliases: BTreeMap<String, PersistentEntityId>,
}

impl BoundaryRegionResolution {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        if self.schema_version != 1
            || self.geometry_source_digest.bytes() == &[0; 32]
            || self.solver_mesh_digest.bytes() == &[0; 32]
        {
            return Err(protocol("boundary region resolution is invalid"));
        }
        runmat_analysis_core::BoundaryRegionAliases::new(self.aliases.clone()).map_err(protocol)?;
        Ok(())
    }

    pub fn canonical_bytes(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate()?;
        encode_contract(BOUNDARY_PREFIX, BOUNDARY_DOMAIN, self, CONTRACT_LIMITS).map_err(protocol)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let resolution = decode_contract(BOUNDARY_PREFIX, BOUNDARY_DOMAIN, bytes, CONTRACT_LIMITS)
            .map_err(protocol)?;
        Self::validate(&resolution)?;
        Ok(resolution)
    }
}

pub(crate) fn prepare_boundary_resolution_objects(
    resolution: &BoundaryRegionResolution,
    access: &ArtifactAccess,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<runmat_analysis_execution::PreparedAnalysisObjects> {
    let bytes = resolution.canonical_bytes()?;
    let identity = Digest::sha256(
        [
            b"runmat-boundary-region-resolution/v1\0".as_slice(),
            resolution.geometry_source_digest.bytes(),
            resolution.solver_mesh_digest.bytes(),
        ]
        .concat(),
    );
    let objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::BoundaryResolution,
        identity,
        &bytes,
        chunk_bytes,
        ObjectNamespace::ResultValue,
        limits,
    )
    .map_err(protocol)?;
    for reference in objects
        .references(access, ValueRefKind::ResultObject)
        .map_err(protocol)?
    {
        access
            .validate_reference(
                &reference,
                ValueRefKind::ResultObject,
                &reference.media_type,
                &reference.value_schema,
            )
            .map_err(protocol)?;
    }
    Ok(objects)
}
