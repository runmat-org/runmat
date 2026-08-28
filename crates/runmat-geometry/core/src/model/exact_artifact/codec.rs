use runmat_canonical_codec::CanonicalLimits;

use super::ExactGeometryManifest;
use crate::{
    ExactBRepModel, ExactBRepTopology, ExactEvaluatorRegistry, GeometryContractError,
    GeometryDigest, GeometryHealingReport,
};

const MANIFEST_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 100_000, 1 << 20, 64);
const COMPONENT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 8 * 1024 * 1024, 128);

impl ExactGeometryManifest {
    pub fn canonical_encode(&self) -> Result<Vec<u8>, GeometryContractError> {
        self.validate()?;
        crate::model::canonical::encode(
            "analysis.geometry.exact-manifest/v2",
            self,
            MANIFEST_LIMITS,
        )
    }

    pub fn canonical_decode(bytes: &[u8]) -> Result<Self, GeometryContractError> {
        let value = crate::model::canonical::decode(
            "analysis.geometry.exact-manifest/v2",
            bytes,
            MANIFEST_LIMITS,
        )?;
        Self::validate(&value)?;
        Ok(value)
    }

    pub fn canonical_digest(&self) -> Result<GeometryDigest, GeometryContractError> {
        crate::model::canonical::digest(&self.canonical_encode()?)
    }
}

pub fn encode_exact_topology(
    topology: &ExactBRepTopology,
    model: &ExactBRepModel,
) -> Result<Vec<u8>, GeometryContractError> {
    topology.validate_against(model)?;
    crate::model::canonical::encode(
        "analysis.geometry.exact-topology/v2",
        topology,
        COMPONENT_LIMITS,
    )
}

pub fn decode_exact_topology(
    bytes: &[u8],
    model: &ExactBRepModel,
) -> Result<ExactBRepTopology, GeometryContractError> {
    let topology: ExactBRepTopology = crate::model::canonical::decode(
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
    crate::model::canonical::encode(
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
    let evaluators: ExactEvaluatorRegistry = crate::model::canonical::decode(
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
    crate::model::canonical::encode(
        "analysis.geometry.healing-report/v2",
        report,
        COMPONENT_LIMITS,
    )
}

pub fn decode_geometry_healing_report(
    bytes: &[u8],
) -> Result<GeometryHealingReport, GeometryContractError> {
    let report: GeometryHealingReport = crate::model::canonical::decode(
        "analysis.geometry.healing-report/v2",
        bytes,
        COMPONENT_LIMITS,
    )?;
    report.validate()?;
    Ok(report)
}
