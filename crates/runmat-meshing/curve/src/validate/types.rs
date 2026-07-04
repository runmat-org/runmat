use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurveValidationOptions {
    pub max_endpoint_error_m: f64,
    pub max_growth_ratio: f64,
}

impl Default for CurveValidationOptions {
    fn default() -> Self {
        Self {
            max_endpoint_error_m: 1.0e-8,
            max_growth_ratio: 2.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveValidationReport {
    pub source_edge_count: usize,
    pub curve_node_count: usize,
    pub curve_element_count: usize,
    pub max_endpoint_error_m: f64,
    pub max_segment_length_m: f64,
    pub max_adjacent_length_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CurveValidationError {
    InvalidOptions,
    MissingCurveEndpoint {
        source_edge_id: u32,
        parameter: f64,
    },
    EndpointDrift {
        source_edge_id: u32,
        parameter: f64,
        error_m: f64,
        max_error_m: f64,
    },
    UnknownNode {
        element_id: u32,
        node_id: u32,
    },
    ElementEdgeMismatch {
        element_id: u32,
        source_edge_id: u32,
        left_source_edge_id: u32,
        right_source_edge_id: u32,
    },
    InvalidElementLength {
        element_id: u32,
        length_m: f64,
    },
    ExcessiveGrowth {
        source_edge_id: u32,
        left_element_id: u32,
        right_element_id: u32,
        ratio: f64,
        max_ratio: f64,
    },
}

impl std::fmt::Display for CurveValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions => write!(
                formatter,
                "curve validation options must use finite non-negative endpoint tolerance and growth ratio >= 1"
            ),
            Self::MissingCurveEndpoint {
                source_edge_id,
                parameter,
            } => write!(
                formatter,
                "source edge {source_edge_id} is missing a recovered curve endpoint at parameter {parameter:.6}"
            ),
            Self::EndpointDrift {
                source_edge_id,
                parameter,
                error_m,
                max_error_m,
            } => write!(
                formatter,
                "source edge {source_edge_id} curve endpoint at parameter {parameter:.6} drifted {error_m:.6e} m, exceeding {max_error_m:.6e} m"
            ),
            Self::UnknownNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "curve element {element_id} references unknown curve node {node_id}"
            ),
            Self::ElementEdgeMismatch {
                element_id,
                source_edge_id,
                left_source_edge_id,
                right_source_edge_id,
            } => write!(
                formatter,
                "curve element {element_id} on source edge {source_edge_id} connects nodes from source edges {left_source_edge_id} and {right_source_edge_id}"
            ),
            Self::InvalidElementLength {
                element_id,
                length_m,
            } => write!(
                formatter,
                "curve element {element_id} has invalid length {length_m:.6e} m"
            ),
            Self::ExcessiveGrowth {
                source_edge_id,
                left_element_id,
                right_element_id,
                ratio,
                max_ratio,
            } => write!(
                formatter,
                "source edge {source_edge_id} curve elements {left_element_id} and {right_element_id} have adjacent length ratio {ratio:.6}, exceeding {max_ratio:.6}"
            ),
        }
    }
}

impl std::error::Error for CurveValidationError {}
