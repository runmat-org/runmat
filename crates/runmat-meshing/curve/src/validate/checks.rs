use std::collections::BTreeMap;

use runmat_meshing_cad::{SourceTopologyEdge, SourceTopologyModel};

use crate::{CurveDiscretization, CurveElement, CurveNode};

use super::types::{CurveValidationError, CurveValidationOptions, CurveValidationReport};

const PARAMETER_TOLERANCE: f64 = 1.0e-12;

pub fn validate_curve_discretization(
    topology: &SourceTopologyModel,
    curves: &CurveDiscretization,
    options: CurveValidationOptions,
) -> Result<CurveValidationReport, CurveValidationError> {
    validate_options(options)?;
    let nodes_by_id = nodes_by_id(curves);
    let source_edges = topology
        .edges
        .iter()
        .map(|edge| (edge.edge_id, edge))
        .collect::<BTreeMap<_, _>>();
    let mut elements_by_edge = BTreeMap::<u32, Vec<&CurveElement>>::new();
    let mut max_projection_error_m = 0.0_f64;
    let mut max_length_error_m = 0.0_f64;
    let mut max_segment_length_m = 0.0_f64;

    for node in &curves.nodes {
        let source_edge = source_edges.get(&node.source_edge_id).ok_or(
            CurveValidationError::MissingSourceEdge {
                source_edge_id: node.source_edge_id,
            },
        )?;
        if !node.parameter.is_finite() || !(0.0..=1.0).contains(&node.parameter) {
            return Err(CurveValidationError::InvalidNodeParameter {
                node_id: node.node_id,
                parameter: node.parameter,
            });
        }
        if options.require_source_edge_projection {
            let expected = source_edge_point(topology, source_edge, node.parameter)?;
            let projection_error_m = distance(node.coordinates_m, expected);
            max_projection_error_m = max_projection_error_m.max(projection_error_m);
            if projection_error_m > options.max_projection_error_m {
                return Err(CurveValidationError::NodeProjectionDrift {
                    node_id: node.node_id,
                    source_edge_id: node.source_edge_id,
                    error_m: projection_error_m,
                    max_error_m: options.max_projection_error_m,
                });
            }
        }
    }

    for element in &curves.elements {
        source_edges.get(&element.source_edge_id).ok_or(
            CurveValidationError::MissingSourceEdge {
                source_edge_id: element.source_edge_id,
            },
        )?;
        let left_node = nodes_by_id.get(&element.node_ids[0]).copied().ok_or(
            CurveValidationError::UnknownNode {
                element_id: element.element_id,
                node_id: element.node_ids[0],
            },
        )?;
        let right_node = nodes_by_id.get(&element.node_ids[1]).copied().ok_or(
            CurveValidationError::UnknownNode {
                element_id: element.element_id,
                node_id: element.node_ids[1],
            },
        )?;
        if left_node.source_edge_id != element.source_edge_id
            || right_node.source_edge_id != element.source_edge_id
        {
            return Err(CurveValidationError::ElementEdgeMismatch {
                element_id: element.element_id,
                source_edge_id: element.source_edge_id,
                left_source_edge_id: left_node.source_edge_id,
                right_source_edge_id: right_node.source_edge_id,
            });
        }
        if !element.length_m.is_finite() || element.length_m <= 0.0 {
            return Err(CurveValidationError::InvalidElementLength {
                element_id: element.element_id,
                length_m: element.length_m,
            });
        }
        let measured_length_m = distance(left_node.coordinates_m, right_node.coordinates_m);
        let length_error_m = (element.length_m - measured_length_m).abs();
        max_length_error_m = max_length_error_m.max(length_error_m);
        if length_error_m > options.max_length_error_m {
            return Err(CurveValidationError::ElementLengthMismatch {
                element_id: element.element_id,
                reported_length_m: element.length_m,
                measured_length_m,
                error_m: length_error_m,
                max_error_m: options.max_length_error_m,
            });
        }
        max_segment_length_m = max_segment_length_m.max(element.length_m);
        elements_by_edge
            .entry(element.source_edge_id)
            .or_default()
            .push(element);
    }

    let mut max_endpoint_error_m = 0.0_f64;
    for edge in &topology.edges {
        max_endpoint_error_m = max_endpoint_error_m.max(validate_edge_endpoint(
            topology,
            curves,
            edge,
            0,
            0.0,
            options.max_endpoint_error_m,
        )?);
        max_endpoint_error_m = max_endpoint_error_m.max(validate_edge_endpoint(
            topology,
            curves,
            edge,
            1,
            1.0,
            options.max_endpoint_error_m,
        )?);
    }

    let max_parameter_gap = validate_parameter_chains(topology, &nodes_by_id, &elements_by_edge)?;
    let max_adjacent_length_ratio = validate_growth(
        &nodes_by_id,
        &mut elements_by_edge,
        options.max_growth_ratio,
    )?;

    Ok(CurveValidationReport {
        source_edge_count: topology.edges.len(),
        curve_node_count: curves.nodes.len(),
        curve_element_count: curves.elements.len(),
        max_endpoint_error_m,
        max_projection_error_m,
        max_length_error_m,
        max_segment_length_m,
        max_parameter_gap,
        max_adjacent_length_ratio,
    })
}

fn validate_options(options: CurveValidationOptions) -> Result<(), CurveValidationError> {
    if !options.max_endpoint_error_m.is_finite()
        || options.max_endpoint_error_m < 0.0
        || !options.max_projection_error_m.is_finite()
        || options.max_projection_error_m < 0.0
        || !options.max_length_error_m.is_finite()
        || options.max_length_error_m < 0.0
        || !options.max_growth_ratio.is_finite()
        || options.max_growth_ratio < 1.0
    {
        return Err(CurveValidationError::InvalidOptions);
    }
    Ok(())
}

fn nodes_by_id(curves: &CurveDiscretization) -> BTreeMap<u32, &CurveNode> {
    curves
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect()
}

fn validate_edge_endpoint(
    topology: &SourceTopologyModel,
    curves: &CurveDiscretization,
    edge: &SourceTopologyEdge,
    edge_endpoint_index: usize,
    parameter: f64,
    max_error_m: f64,
) -> Result<f64, CurveValidationError> {
    let source_vertex = topology
        .vertices
        .get(edge.node_ids[edge_endpoint_index] as usize);
    let source_coordinates_m = source_vertex
        .filter(|vertex| vertex.vertex_id == edge.node_ids[edge_endpoint_index])
        .map(|vertex| vertex.coordinates_m)
        .ok_or(CurveValidationError::MissingCurveEndpoint {
            source_edge_id: edge.edge_id,
            parameter,
        })?;
    let endpoint = curves
        .nodes
        .iter()
        .filter(|node| {
            node.source_edge_id == edge.edge_id && (node.parameter - parameter).abs() <= 1.0e-12
        })
        .min_by(|left, right| {
            distance(left.coordinates_m, source_coordinates_m)
                .total_cmp(&distance(right.coordinates_m, source_coordinates_m))
        })
        .ok_or(CurveValidationError::MissingCurveEndpoint {
            source_edge_id: edge.edge_id,
            parameter,
        })?;
    let error_m = distance(endpoint.coordinates_m, source_coordinates_m);
    if error_m > max_error_m {
        return Err(CurveValidationError::EndpointDrift {
            source_edge_id: edge.edge_id,
            parameter,
            error_m,
            max_error_m,
        });
    }
    Ok(error_m)
}

fn source_edge_point(
    topology: &SourceTopologyModel,
    edge: &SourceTopologyEdge,
    parameter: f64,
) -> Result<[f64; 3], CurveValidationError> {
    let left = topology
        .vertices
        .get(edge.node_ids[0] as usize)
        .filter(|vertex| vertex.vertex_id == edge.node_ids[0])
        .map(|vertex| vertex.coordinates_m)
        .ok_or(CurveValidationError::MissingCurveEndpoint {
            source_edge_id: edge.edge_id,
            parameter: 0.0,
        })?;
    let right = topology
        .vertices
        .get(edge.node_ids[1] as usize)
        .filter(|vertex| vertex.vertex_id == edge.node_ids[1])
        .map(|vertex| vertex.coordinates_m)
        .ok_or(CurveValidationError::MissingCurveEndpoint {
            source_edge_id: edge.edge_id,
            parameter: 1.0,
        })?;
    Ok([
        left[0] + (right[0] - left[0]) * parameter,
        left[1] + (right[1] - left[1]) * parameter,
        left[2] + (right[2] - left[2]) * parameter,
    ])
}

#[derive(Debug, Clone, Copy)]
struct ChainElement<'a> {
    element: &'a CurveElement,
    start_parameter: f64,
    end_parameter: f64,
}

fn validate_parameter_chains(
    topology: &SourceTopologyModel,
    nodes_by_id: &BTreeMap<u32, &CurveNode>,
    elements_by_edge: &BTreeMap<u32, Vec<&CurveElement>>,
) -> Result<f64, CurveValidationError> {
    for source_edge in &topology.edges {
        let Some(elements) = elements_by_edge.get(&source_edge.edge_id) else {
            return Err(CurveValidationError::MissingElementChain {
                source_edge_id: source_edge.edge_id,
            });
        };
        let mut chain = Vec::<ChainElement<'_>>::with_capacity(elements.len());
        for element in elements {
            let left_node = nodes_by_id.get(&element.node_ids[0]).copied().ok_or(
                CurveValidationError::UnknownNode {
                    element_id: element.element_id,
                    node_id: element.node_ids[0],
                },
            )?;
            let right_node = nodes_by_id.get(&element.node_ids[1]).copied().ok_or(
                CurveValidationError::UnknownNode {
                    element_id: element.element_id,
                    node_id: element.node_ids[1],
                },
            )?;
            if right_node.parameter <= left_node.parameter + PARAMETER_TOLERANCE {
                return Err(CurveValidationError::NonIncreasingElementParameter {
                    source_edge_id: source_edge.edge_id,
                    element_id: element.element_id,
                    left_parameter: left_node.parameter,
                    right_parameter: right_node.parameter,
                });
            }
            chain.push(ChainElement {
                element,
                start_parameter: left_node.parameter,
                end_parameter: right_node.parameter,
            });
        }
        chain.sort_by(|left, right| left.start_parameter.total_cmp(&right.start_parameter));

        let mut expected_parameter = 0.0_f64;
        let mut previous_element_id = None::<u32>;
        for chain_element in chain {
            if chain_element.start_parameter > expected_parameter + PARAMETER_TOLERANCE {
                return Err(CurveValidationError::ElementParameterGap {
                    source_edge_id: source_edge.edge_id,
                    left_element_id: previous_element_id,
                    right_element_id: Some(chain_element.element.element_id),
                    expected_parameter,
                    actual_parameter: chain_element.start_parameter,
                });
            }
            if chain_element.start_parameter < expected_parameter - PARAMETER_TOLERANCE {
                return Err(CurveValidationError::ElementParameterOverlap {
                    source_edge_id: source_edge.edge_id,
                    left_element_id: previous_element_id
                        .unwrap_or(chain_element.element.element_id),
                    right_element_id: chain_element.element.element_id,
                    expected_parameter,
                    actual_parameter: chain_element.start_parameter,
                });
            }
            expected_parameter = expected_parameter.max(chain_element.end_parameter);
            previous_element_id = Some(chain_element.element.element_id);
        }
        if expected_parameter < 1.0 - PARAMETER_TOLERANCE {
            return Err(CurveValidationError::ElementParameterGap {
                source_edge_id: source_edge.edge_id,
                left_element_id: previous_element_id,
                right_element_id: None,
                expected_parameter,
                actual_parameter: 1.0,
            });
        }
    }
    Ok(0.0)
}

fn validate_growth(
    nodes_by_id: &BTreeMap<u32, &CurveNode>,
    elements_by_edge: &mut BTreeMap<u32, Vec<&CurveElement>>,
    max_growth_ratio: f64,
) -> Result<f64, CurveValidationError> {
    let mut max_ratio = 1.0_f64;
    for (source_edge_id, elements) in elements_by_edge {
        elements.sort_by(|left, right| {
            let left_parameter = nodes_by_id
                .get(&left.node_ids[0])
                .map(|node| node.parameter)
                .unwrap_or(f64::INFINITY);
            let right_parameter = nodes_by_id
                .get(&right.node_ids[0])
                .map(|node| node.parameter)
                .unwrap_or(f64::INFINITY);
            left_parameter.total_cmp(&right_parameter)
        });
        for pair in elements.windows(2) {
            let left = pair[0];
            let right = pair[1];
            let ratio = adjacent_length_ratio(left.length_m, right.length_m);
            max_ratio = max_ratio.max(ratio);
            if ratio > max_growth_ratio {
                return Err(CurveValidationError::ExcessiveGrowth {
                    source_edge_id: *source_edge_id,
                    left_element_id: left.element_id,
                    right_element_id: right.element_id,
                    ratio,
                    max_ratio: max_growth_ratio,
                });
            }
        }
    }
    Ok(max_ratio)
}

fn adjacent_length_ratio(left: f64, right: f64) -> f64 {
    let min = left.min(right);
    let max = left.max(right);
    if min <= 0.0 {
        f64::INFINITY
    } else {
        max / min
    }
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    if !left.iter().all(|coordinate| coordinate.is_finite())
        || !right.iter().all(|coordinate| coordinate.is_finite())
    {
        return f64::INFINITY;
    }
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}
