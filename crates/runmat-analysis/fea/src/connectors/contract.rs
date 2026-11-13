use runmat_analysis_core::{AnalysisConnector, ConnectorKind};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;

const CONNECTOR_PREFIX: &[u8] = b"runmat-analysis-structural-connector/v1\0";
/// Connector sets grow with the model; the run's admitted budget governs their size.
const CONNECTOR_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(4_000_000, 64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalConnectorNode {
    pub node_index: usize,
    pub node_identity: StableDigest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceFoundationQuadrature {
    pub nodes: Vec<CanonicalConnectorNode>,
    pub shape_values: Vec<f64>,
    pub unit_normal: [f64; 3],
    pub area_weight_m2: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CanonicalConnectorEntry {
    AxialSpring {
        connector_id: String,
        source_digest: Digest,
        primary: CanonicalConnectorNode,
        secondary: CanonicalConnectorNode,
        unit_axis: [f64; 3],
        stiffness_newtons_per_meter: f64,
    },
    AxialDashpot {
        connector_id: String,
        source_digest: Digest,
        primary: CanonicalConnectorNode,
        secondary: CanonicalConnectorNode,
        unit_axis: [f64; 3],
        damping_newton_seconds_per_meter: f64,
    },
    IsotropicGroundSpring {
        connector_id: String,
        source_digest: Digest,
        target: CanonicalConnectorNode,
        stiffness_newtons_per_meter: f64,
    },
    SurfaceNormalFoundation {
        connector_id: String,
        source_digest: Digest,
        modulus_newtons_per_cubic_meter: f64,
        quadrature: Vec<SurfaceFoundationQuadrature>,
    },
}

impl CanonicalConnectorEntry {
    pub fn connector_id(&self) -> &str {
        match self {
            Self::AxialSpring { connector_id, .. }
            | Self::AxialDashpot { connector_id, .. }
            | Self::IsotropicGroundSpring { connector_id, .. }
            | Self::SurfaceNormalFoundation { connector_id, .. } => connector_id,
        }
    }

    pub const fn source_digest(&self) -> Digest {
        match self {
            Self::AxialSpring { source_digest, .. }
            | Self::AxialDashpot { source_digest, .. }
            | Self::IsotropicGroundSpring { source_digest, .. }
            | Self::SurfaceNormalFoundation { source_digest, .. } => *source_digest,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStructuralConnectorPlan {
    pub schema_version: u16,
    pub topology: CanonicalStructuralTopologyIdentity,
    pub node_count: usize,
    pub authored_connectors_digest: Digest,
    pub entries: Vec<CanonicalConnectorEntry>,
}

impl CanonicalStructuralConnectorPlan {
    pub fn new(
        topology: CanonicalStructuralTopologyIdentity,
        node_count: usize,
        authored_connectors_digest: Digest,
        mut entries: Vec<CanonicalConnectorEntry>,
    ) -> Result<Self, ConnectorContractError> {
        entries.sort_by(|left, right| left.connector_id().cmp(right.connector_id()));
        let plan = Self {
            schema_version: 1,
            topology,
            node_count,
            authored_connectors_digest,
            entries,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn empty(
        topology: CanonicalStructuralTopologyIdentity,
        node_count: usize,
    ) -> Result<Self, ConnectorContractError> {
        Self::new(
            topology,
            node_count,
            authored_connectors_digest(&[])?,
            Vec::new(),
        )
    }

    pub fn validate(&self) -> Result<(), ConnectorContractError> {
        if self.schema_version != 1 || !self.topology.has_valid_identity() {
            return Err(ConnectorContractError::Invalid(
                "connector plan schema or topology is invalid".into(),
            ));
        }
        if !self
            .entries
            .windows(2)
            .all(|pair| pair[0].connector_id() < pair[1].connector_id())
        {
            return Err(ConnectorContractError::Invalid(
                "connector entries must be uniquely ordered by identity".into(),
            ));
        }
        for entry in &self.entries {
            validate_entry(entry, self.node_count)?;
        }
        Ok(())
    }

    pub fn content_digest(&self) -> Result<Digest, ConnectorContractError> {
        self.validate()?;
        canonical_digest("connector-plan", self)
    }

    pub fn validate_against_authored(
        &self,
        connectors: &[AnalysisConnector],
    ) -> Result<(), ConnectorContractError> {
        if self.authored_connectors_digest != authored_connectors_digest(connectors)?
            || self.entries.len() != connectors.len()
        {
            return Err(ConnectorContractError::AuthoredConnectorsMismatch);
        }
        for entry in &self.entries {
            let Some(connector) = connectors
                .iter()
                .find(|connector| connector.connector_id == entry.connector_id())
            else {
                return Err(ConnectorContractError::AuthoredConnectorsMismatch);
            };
            if entry.source_digest() != source_digest(connector)?
                || !entry_matches_authored(entry, connector)
            {
                return Err(ConnectorContractError::AuthoredConnectorsMismatch);
            }
        }
        Ok(())
    }
}

fn entry_matches_authored(entry: &CanonicalConnectorEntry, connector: &AnalysisConnector) -> bool {
    match (entry, &connector.kind) {
        (
            CanonicalConnectorEntry::AxialSpring {
                stiffness_newtons_per_meter,
                ..
            },
            ConnectorKind::AxialSpring { stiffness, .. },
        ) => *stiffness_newtons_per_meter == stiffness.canonical_value(),
        (
            CanonicalConnectorEntry::AxialDashpot {
                damping_newton_seconds_per_meter,
                ..
            },
            ConnectorKind::AxialDashpot { damping, .. },
        ) => *damping_newton_seconds_per_meter == damping.canonical_value(),
        (
            CanonicalConnectorEntry::IsotropicGroundSpring {
                stiffness_newtons_per_meter,
                ..
            },
            ConnectorKind::IsotropicGroundSpring { stiffness, .. },
        ) => *stiffness_newtons_per_meter == stiffness.canonical_value(),
        (
            CanonicalConnectorEntry::SurfaceNormalFoundation {
                modulus_newtons_per_cubic_meter,
                ..
            },
            ConnectorKind::SurfaceNormalFoundation { modulus, .. },
        ) => *modulus_newtons_per_cubic_meter == modulus.canonical_value(),
        _ => false,
    }
}

pub fn authored_connectors_digest(
    connectors: &[AnalysisConnector],
) -> Result<Digest, ConnectorContractError> {
    let mut sources = connectors
        .iter()
        .map(source_digest)
        .collect::<Result<Vec<_>, _>>()?;
    sources.sort_unstable();
    canonical_digest("authored-connectors", &sources)
}

pub(crate) fn source_digest(
    connector: &AnalysisConnector,
) -> Result<Digest, ConnectorContractError> {
    canonical_digest("authored-connector", connector)
}

fn validate_entry(
    entry: &CanonicalConnectorEntry,
    node_count: usize,
) -> Result<(), ConnectorContractError> {
    if entry.connector_id().trim().is_empty() {
        return Err(ConnectorContractError::Invalid(
            "connector identity is empty".into(),
        ));
    }
    match entry {
        CanonicalConnectorEntry::AxialSpring {
            primary,
            secondary,
            unit_axis,
            stiffness_newtons_per_meter,
            ..
        } => {
            validate_axial(
                *primary,
                *secondary,
                *unit_axis,
                *stiffness_newtons_per_meter,
                node_count,
            )?;
        }
        CanonicalConnectorEntry::AxialDashpot {
            primary,
            secondary,
            unit_axis,
            damping_newton_seconds_per_meter,
            ..
        } => {
            validate_axial(
                *primary,
                *secondary,
                *unit_axis,
                *damping_newton_seconds_per_meter,
                node_count,
            )?;
        }
        CanonicalConnectorEntry::IsotropicGroundSpring {
            target,
            stiffness_newtons_per_meter,
            ..
        } => {
            validate_node(*target, node_count)?;
            validate_positive(*stiffness_newtons_per_meter)?;
        }
        CanonicalConnectorEntry::SurfaceNormalFoundation {
            modulus_newtons_per_cubic_meter,
            quadrature,
            ..
        } => {
            validate_positive(*modulus_newtons_per_cubic_meter)?;
            if quadrature.is_empty() {
                return Err(ConnectorContractError::Invalid(
                    "surface foundation has no physical quadrature".into(),
                ));
            }
            for point in quadrature {
                if point.nodes.is_empty()
                    || point.nodes.len() != point.shape_values.len()
                    || !point.area_weight_m2.is_finite()
                    || point.area_weight_m2 <= 0.0
                    || point.shape_values.iter().any(|value| !value.is_finite())
                    || (point.shape_values.iter().sum::<f64>() - 1.0).abs() > 1.0e-10
                    || !is_unit(point.unit_normal)
                {
                    return Err(ConnectorContractError::Invalid(
                        "surface foundation quadrature is invalid".into(),
                    ));
                }
                for node in &point.nodes {
                    validate_node(*node, node_count)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_axial(
    primary: CanonicalConnectorNode,
    secondary: CanonicalConnectorNode,
    unit_axis: [f64; 3],
    coefficient: f64,
    node_count: usize,
) -> Result<(), ConnectorContractError> {
    validate_node(primary, node_count)?;
    validate_node(secondary, node_count)?;
    validate_positive(coefficient)?;
    if primary == secondary || !is_unit(unit_axis) {
        return Err(ConnectorContractError::Invalid(
            "axial connector endpoints or axis are invalid".into(),
        ));
    }
    Ok(())
}

fn validate_node(
    node: CanonicalConnectorNode,
    node_count: usize,
) -> Result<(), ConnectorContractError> {
    if node.node_index >= node_count || node.node_identity == StableDigest::ZERO {
        return Err(ConnectorContractError::Invalid(
            "connector node reference is outside the canonical topology".into(),
        ));
    }
    Ok(())
}

fn validate_positive(value: f64) -> Result<(), ConnectorContractError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ConnectorContractError::Invalid(
            "connector coefficient must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn is_unit(vector: [f64; 3]) -> bool {
    vector.iter().all(|value| value.is_finite())
        && (vector.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() <= 1.0e-10
}

fn canonical_digest<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> Result<Digest, ConnectorContractError> {
    encode_contract(CONNECTOR_PREFIX, domain, value, CONNECTOR_LIMITS)
        .map(|bytes| Digest::sha256(&bytes))
        .map_err(|error| ConnectorContractError::Encoding(error.to_string()))
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ConnectorContractError {
    #[error("connector contract is malformed: {0}")]
    Invalid(String),
    #[error("connector contract does not match the authored connectors")]
    AuthoredConnectorsMismatch,
    #[error("connector contract encoding failed: {0}")]
    Encoding(String),
}
