use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurveValidationOptions {
    pub max_endpoint_error_m: f64,
    pub max_projection_error_m: f64,
    pub max_length_error_m: f64,
    pub max_growth_ratio: f64,
    pub require_source_edge_projection: bool,
}

impl Default for CurveValidationOptions {
    fn default() -> Self {
        Self {
            max_endpoint_error_m: 1.0e-8,
            max_projection_error_m: 1.0e-8,
            max_length_error_m: 1.0e-8,
            max_growth_ratio: 2.0,
            require_source_edge_projection: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveValidationReport {
    pub source_edge_count: usize,
    pub curve_node_count: usize,
    pub curve_element_count: usize,
    pub max_endpoint_error_m: f64,
    pub max_projection_error_m: f64,
    pub max_length_error_m: f64,
    pub max_segment_length_m: f64,
    pub max_parameter_gap: f64,
    pub max_adjacent_length_ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CurveValidationError {
    InvalidOptions,
    MissingCurveEndpoint {
        source_edge_id: u32,
        parameter: f64,
    },
    MissingSourceEdge {
        source_edge_id: u32,
    },
    EndpointDrift {
        source_edge_id: u32,
        parameter: f64,
        error_m: f64,
        max_error_m: f64,
    },
    InvalidNodeParameter {
        node_id: u32,
        parameter: f64,
    },
    NodeProjectionDrift {
        node_id: u32,
        source_edge_id: u32,
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
    ElementLengthMismatch {
        element_id: u32,
        reported_length_m: f64,
        measured_length_m: f64,
        error_m: f64,
        max_error_m: f64,
    },
    MissingElementChain {
        source_edge_id: u32,
    },
    NonIncreasingElementParameter {
        source_edge_id: u32,
        element_id: u32,
        left_parameter: f64,
        right_parameter: f64,
    },
    ElementParameterGap {
        source_edge_id: u32,
        left_element_id: Option<u32>,
        right_element_id: Option<u32>,
        expected_parameter: f64,
        actual_parameter: f64,
    },
    ElementParameterOverlap {
        source_edge_id: u32,
        left_element_id: u32,
        right_element_id: u32,
        expected_parameter: f64,
        actual_parameter: f64,
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
                "curve validation options must use finite non-negative endpoint/projection/length tolerances and growth ratio >= 1"
            ),
            Self::MissingCurveEndpoint {
                source_edge_id,
                parameter,
            } => write!(
                formatter,
                "source edge {source_edge_id} is missing a recovered curve endpoint at parameter {parameter:.6}"
            ),
            Self::MissingSourceEdge { source_edge_id } => {
                write!(formatter, "source edge {source_edge_id} is missing")
            }
            Self::EndpointDrift {
                source_edge_id,
                parameter,
                error_m,
                max_error_m,
            } => write!(
                formatter,
                "source edge {source_edge_id} curve endpoint at parameter {parameter:.6} drifted {error_m:.6e} m, exceeding {max_error_m:.6e} m"
            ),
            Self::InvalidNodeParameter { node_id, parameter } => write!(
                formatter,
                "curve node {node_id} has invalid source-edge parameter {parameter:.6e}"
            ),
            Self::NodeProjectionDrift {
                node_id,
                source_edge_id,
                error_m,
                max_error_m,
            } => write!(
                formatter,
                "curve node {node_id} on source edge {source_edge_id} projects {error_m:.6e} m from its source-edge parameter point, exceeding {max_error_m:.6e} m"
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
            Self::ElementLengthMismatch {
                element_id,
                reported_length_m,
                measured_length_m,
                error_m,
                max_error_m,
            } => write!(
                formatter,
                "curve element {element_id} reports length {reported_length_m:.6e} m but node coordinates measure {measured_length_m:.6e} m; error {error_m:.6e} m exceeds {max_error_m:.6e} m"
            ),
            Self::MissingElementChain { source_edge_id } => write!(
                formatter,
                "source edge {source_edge_id} has no recovered curve element chain"
            ),
            Self::NonIncreasingElementParameter {
                source_edge_id,
                element_id,
                left_parameter,
                right_parameter,
            } => write!(
                formatter,
                "curve element {element_id} on source edge {source_edge_id} has non-increasing parameters {left_parameter:.6} -> {right_parameter:.6}"
            ),
            Self::ElementParameterGap {
                source_edge_id,
                left_element_id,
                right_element_id,
                expected_parameter,
                actual_parameter,
            } => write!(
                formatter,
                "source edge {source_edge_id} curve element chain has a parameter gap from {expected_parameter:.6} after element {left_element_id:?} to {actual_parameter:.6} before element {right_element_id:?}"
            ),
            Self::ElementParameterOverlap {
                source_edge_id,
                left_element_id,
                right_element_id,
                expected_parameter,
                actual_parameter,
            } => write!(
                formatter,
                "source edge {source_edge_id} curve element chain overlaps: element {right_element_id} starts at {actual_parameter:.6} before element {left_element_id} ends at {expected_parameter:.6}"
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
