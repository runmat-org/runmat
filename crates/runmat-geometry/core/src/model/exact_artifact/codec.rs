use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest as _, Sha256};

use super::ExactGeometryManifest;
use crate::{
    ExactBRepModel, ExactBRepTopology, ExactEvaluatorRegistry, GeometryContractError,
    GeometryDigest, GeometryHealingReport,
};

const GEOMETRY_CODEC_PREFIX: &[u8] = b"runmat-geometry-canonical-cbor/v1\0";
const MANIFEST_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 100_000, 1 << 20, 64);
const COMPONENT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 8 * 1024 * 1024, 128);

impl ExactGeometryManifest {
    pub fn canonical_encode(&self) -> Result<Vec<u8>, GeometryContractError> {
        self.validate()?;
        encode("analysis.geometry.exact-manifest/v2", self, MANIFEST_LIMITS)
    }

    pub fn canonical_decode(bytes: &[u8]) -> Result<Self, GeometryContractError> {
        let value = decode(
            "analysis.geometry.exact-manifest/v2",
            bytes,
            MANIFEST_LIMITS,
        )?;
        Self::validate(&value)?;
        Ok(value)
    }

    pub fn canonical_digest(&self) -> Result<GeometryDigest, GeometryContractError> {
        digest(&self.canonical_encode()?)
    }
}

pub fn encode_exact_topology(
    topology: &ExactBRepTopology,
    model: &ExactBRepModel,
) -> Result<Vec<u8>, GeometryContractError> {
    topology.validate_against(model)?;
    encode(
        "analysis.geometry.exact-topology/v2",
        topology,
        COMPONENT_LIMITS,
    )
}

pub fn decode_exact_topology(
    bytes: &[u8],
    model: &ExactBRepModel,
) -> Result<ExactBRepTopology, GeometryContractError> {
    let topology: ExactBRepTopology = decode(
        "analysis.geometry.exact-topology/v2",
        bytes,
        COMPONENT_LIMITS,
    )?;
    topology.validate_against(model)?;
    Ok(topology)
}

pub fn encode_exact_evaluators(
    evaluators: &ExactEvaluatorRegistry,
    topology: &ExactBRepTopology,
    model: &ExactBRepModel,
) -> Result<Vec<u8>, GeometryContractError> {
    evaluators.validate_against(topology, model)?;
    encode(
        "analysis.geometry.exact-evaluators/v2",
        evaluators,
        COMPONENT_LIMITS,
    )
}

pub fn decode_exact_evaluators(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    model: &ExactBRepModel,
) -> Result<ExactEvaluatorRegistry, GeometryContractError> {
    let evaluators: ExactEvaluatorRegistry = decode(
        "analysis.geometry.exact-evaluators/v2",
        bytes,
        COMPONENT_LIMITS,
    )?;
    evaluators.validate_against(topology, model)?;
    Ok(evaluators)
}

pub fn encode_geometry_healing_report(
    report: &GeometryHealingReport,
) -> Result<Vec<u8>, GeometryContractError> {
    report.validate()?;
    encode(
        "analysis.geometry.healing-report/v2",
        report,
        COMPONENT_LIMITS,
    )
}

pub fn decode_geometry_healing_report(
    bytes: &[u8],
) -> Result<GeometryHealingReport, GeometryContractError> {
    let report: GeometryHealingReport = decode(
        "analysis.geometry.healing-report/v2",
        bytes,
        COMPONENT_LIMITS,
    )?;
    report.validate()?;
    Ok(report)
}

pub(super) fn digest(bytes: &[u8]) -> Result<GeometryDigest, GeometryContractError> {
    if bytes.is_empty() {
        return Err(GeometryContractError::invalid(
            "geometry component",
            "encoded component must not be empty",
        ));
    }
    Ok(GeometryDigest::from_bytes(Sha256::digest(bytes).into()))
}

fn encode<T: Serialize>(
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
) -> Result<Vec<u8>, GeometryContractError> {
    runmat_canonical_codec::encode_contract(GEOMETRY_CODEC_PREFIX, domain, value, limits)
        .map_err(map_error)
}

fn decode<T: DeserializeOwned>(
    domain: &str,
    bytes: &[u8],
    limits: CanonicalLimits,
) -> Result<T, GeometryContractError> {
    runmat_canonical_codec::decode_contract(GEOMETRY_CODEC_PREFIX, domain, bytes, limits)
        .map_err(map_error)
}

fn map_error(error: CanonicalCodecError) -> GeometryContractError {
    GeometryContractError::invalid(error.field, error.reason)
}
