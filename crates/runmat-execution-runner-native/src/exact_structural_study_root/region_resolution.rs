use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::value::ValueRefKind;
use runmat_execution::Digest;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use serde::{Deserialize, Serialize};

use super::{protocol, CONTRACT_LIMITS, REGION_RESOLUTION_DOMAIN, REGION_RESOLUTION_PREFIX};
use crate::NativeExecutionResult;

/// Binds a whole-study result to the exact geometry source and solver mesh it ran on.
///
/// Model regions already name canonical persistent identities, which the solver matches against
/// the mesh's own provenance; this record carries no second region identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyRegionResolution {
    pub schema_version: u16,
    pub geometry_source_digest: Digest,
    pub solver_mesh_digest: Digest,
}

impl TopologyRegionResolution {
    pub const SCHEMA_VERSION: u16 = 2;
}

impl TopologyRegionResolution {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        if self.schema_version != Self::SCHEMA_VERSION
            || self.geometry_source_digest.bytes() == &[0; 32]
            || self.solver_mesh_digest.bytes() == &[0; 32]
        {
            return Err(protocol("topology region resolution is invalid"));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate()?;
        encode_contract(
            REGION_RESOLUTION_PREFIX,
            REGION_RESOLUTION_DOMAIN,
            self,
            CONTRACT_LIMITS,
        )
        .map_err(protocol)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let resolution = decode_contract(
            REGION_RESOLUTION_PREFIX,
            REGION_RESOLUTION_DOMAIN,
            bytes,
            CONTRACT_LIMITS,
        )
        .map_err(protocol)?;
        Self::validate(&resolution)?;
        Ok(resolution)
    }
}

pub(crate) fn prepare_topology_resolution_objects(
    resolution: &TopologyRegionResolution,
    access: &ArtifactAccess,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<runmat_analysis_execution::PreparedAnalysisObjects> {
    let bytes = resolution.canonical_bytes()?;
    let identity = Digest::sha256(
        [
            b"runmat-topology-region-resolution/v1\0".as_slice(),
            resolution.geometry_source_digest.bytes(),
            resolution.solver_mesh_digest.bytes(),
        ]
        .concat(),
    );
    let objects = runmat_analysis_execution::prepare_analysis_objects(
        AnalysisNumericalObjectKind::TopologyResolution,
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
