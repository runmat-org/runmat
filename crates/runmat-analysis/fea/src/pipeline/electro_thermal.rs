use runmat_analysis_core::AnalysisField;

use crate::{
    assembly::ElectroThermalAssemblySummary,
    contracts::{
        fea_electro_thermal_temperature_field_id, fea_electro_thermal_thermal_residual_field_id,
        FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY, FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
        FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL, FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT,
    },
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const MIN_CONDUCTIVITY: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
struct ConductanceEdge {
    from: usize,
    to: usize,
    conductance: f64,
}

#[derive(Debug)]
struct ConductanceDomainGraph {
    edges: Vec<ConductanceEdge>,
    topology: ElectroThermalDomainTopology,
    node_degrees: Vec<usize>,
    boundary_node_count: usize,
    max_node_degree: usize,
    mean_node_degree: f64,
    conductance_span_ratio: f64,
    topology_coverage_ratio: f64,
}

#[derive(Debug, Clone)]
struct ElectroThermalDomainTopology {
    conductive_node_count: usize,
    conductive_edge_count: usize,
    mapped_voltage_boundary_count: usize,
    mapped_current_source_count: usize,
    material_region_count: usize,
    topology_component_count: usize,
    source_boundary_alignment_ratio: f64,
    domain_conductance_coverage_ratio: f64,
    material_region_coverage_ratio: f64,
}

#[derive(Default)]
pub(crate) struct ElectroThermalFields {
    pub(crate) static_fields: Vec<AnalysisField>,
    pub(crate) temperature_snapshots: Vec<AnalysisField>,
    pub(crate) thermal_residual_snapshots: Vec<AnalysisField>,
    pub(crate) diagnostics: Vec<FeaDiagnostic>,
}

#[derive(Debug)]
struct ElectroThermalPotentialSolve {
    potential: Vec<f64>,
    graph: ConductanceDomainGraph,
    edge_current: Vec<f64>,
    nodal_unscaled_joule_heat: Vec<f64>,
    residual_norm: f64,
    equation_scale: f64,
    current_balance_residual: f64,
    potential_span_v: f64,
    integrated_joule_heat_w: f64,
    condition_estimate: f64,
    source_current_a: f64,
}

pub(crate) fn recover_electro_thermal_fields(
    summary: Option<&ElectroThermalAssemblySummary>,
    progress_factors: &[f64],
    residual_norms: &[f64],
    dof_count: usize,
) -> ElectroThermalFields {
    let Some(summary) = summary.filter(|summary| summary.enabled) else {
        return ElectroThermalFields::default();
    };

    let node_count = dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1);
    let solve = solve_conductance_domain_graph(summary, node_count);
    let mut electric_field = Vec::with_capacity(node_count * VECTOR_COMPONENT_COUNT);
    let mut current_density = Vec::with_capacity(node_count * VECTOR_COMPONENT_COUNT);
    let mut nodal_field_sum = vec![0.0_f64; node_count];
    let mut nodal_current_sum = vec![0.0_f64; node_count];
    let signed_span = solve
        .potential
        .first()
        .zip(solve.potential.last())
        .map(|(first, last)| first - last)
        .unwrap_or(0.0)
        .signum();
    for (edge, current) in solve.graph.edges.iter().zip(solve.edge_current.iter()) {
        let voltage_drop = solve.potential[edge.from] - solve.potential[edge.to];
        let local_field = voltage_drop;
        let oriented_current = current.abs() * signed_span;
        nodal_field_sum[edge.from] += local_field;
        nodal_field_sum[edge.to] += local_field;
        nodal_current_sum[edge.from] += oriented_current;
        nodal_current_sum[edge.to] += oriented_current;
    }
    for index in 0..node_count {
        let degree = solve
            .graph
            .node_degrees
            .get(index)
            .copied()
            .unwrap_or_default()
            .max(1) as f64;
        electric_field.extend_from_slice(&[nodal_field_sum[index] / degree, 0.0, 0.0]);
        current_density.extend_from_slice(&[nodal_current_sum[index] / degree, 0.0, 0.0]);
    }
    let unscaled_joule_integral = solve.nodal_unscaled_joule_heat.iter().sum::<f64>();
    let heating_scale = if unscaled_joule_integral > 1.0e-12 {
        summary.joule_heating_scale.max(0.0) / unscaled_joule_integral
    } else {
        0.0
    };
    let joule_heat = solve
        .nodal_unscaled_joule_heat
        .iter()
        .map(|value| value * heating_scale)
        .collect::<Vec<_>>();
    let mean_joule_heat = if joule_heat.is_empty() {
        0.0
    } else {
        joule_heat.iter().sum::<f64>() / joule_heat.len() as f64
    };

    let diagnostics = vec![
        electro_thermal_domain_topology_diagnostic(&solve),
        electro_thermal_potential_solve_diagnostic(node_count, &solve),
        electro_thermal_conduction_conservation_diagnostic(node_count, &solve),
    ];

    let static_fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL,
            vec![node_count],
            solve.potential,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            electric_field,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            current_density,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT,
            vec![node_count],
            joule_heat.clone(),
        ),
    ];

    let mut temperature_snapshots = Vec::with_capacity(progress_factors.len());
    let mut thermal_residual_snapshots = Vec::with_capacity(progress_factors.len());
    let mut final_temperature = vec![summary.reference_temperature_k; node_count];
    for (index, progress_factor) in progress_factors.iter().copied().enumerate() {
        let current_factor = progress_factor.clamp(0.0, 1.0);
        let mean_temperature_rise = summary.joule_heating_scale.max(0.0)
            * current_factor
            * (1.0 + summary.temporal_profile_variation);
        let temperature = joule_heat
            .iter()
            .map(|source| {
                let source_scale = if mean_joule_heat > 1.0e-12 {
                    source / mean_joule_heat
                } else {
                    0.0
                };
                summary.reference_temperature_k + mean_temperature_rise * source_scale
            })
            .collect::<Vec<_>>();
        final_temperature = temperature.clone();
        temperature_snapshots.push(AnalysisField::host_f64(
            fea_electro_thermal_temperature_field_id(index),
            vec![node_count],
            temperature,
        ));

        let residual = residual_norms
            .get(index)
            .copied()
            .filter(|value| value.is_finite())
            .unwrap_or(0.0)
            * (1.0 + summary.conductivity_spread_ratio.ln().max(0.0));
        thermal_residual_snapshots.push(AnalysisField::host_f64(
            fea_electro_thermal_thermal_residual_field_id(index),
            vec![1],
            vec![residual],
        ));
    }
    let mut diagnostics = diagnostics;
    diagnostics.push(electro_thermal_source_coupling_diagnostic(
        summary,
        &joule_heat,
        &final_temperature,
    ));

    ElectroThermalFields {
        static_fields,
        temperature_snapshots,
        thermal_residual_snapshots,
        diagnostics,
    }
}

fn solve_conductance_domain_graph(
    summary: &ElectroThermalAssemblySummary,
    node_count: usize,
) -> ElectroThermalPotentialSolve {
    let node_count = node_count.max(1);
    let potential_span_v = summary.applied_voltage_v.abs();
    let graph = build_conductance_domain_graph(summary, node_count);
    if node_count == 1 {
        return ElectroThermalPotentialSolve {
            potential: vec![summary.applied_voltage_v],
            graph,
            edge_current: Vec::new(),
            nodal_unscaled_joule_heat: vec![0.0],
            residual_norm: 0.0,
            equation_scale: potential_span_v.max(1.0),
            current_balance_residual: 0.0,
            potential_span_v,
            integrated_joule_heat_w: 0.0,
            condition_estimate: 1.0,
            source_current_a: 0.0,
        };
    }

    let mut potential = vec![0.0_f64; node_count];
    potential[0] = summary.applied_voltage_v;
    potential[node_count - 1] = 0.0;
    let interior_count = node_count.saturating_sub(2);
    if interior_count > 0 {
        let mut matrix = vec![vec![0.0_f64; interior_count]; interior_count];
        let mut rhs = vec![0.0_f64; interior_count];
        for edge in &graph.edges {
            let conductance = edge.conductance.max(MIN_CONDUCTIVITY);
            accumulate_graph_equation(
                edge.from,
                edge.to,
                conductance,
                node_count,
                &mut matrix,
                &mut rhs,
                &potential,
            );
        }
        let interior = solve_dense_system(matrix, rhs);
        for (offset, value) in interior.into_iter().enumerate() {
            potential[offset + 1] = value;
        }
    }

    let edge_current = graph
        .edges
        .iter()
        .map(|edge| {
            let voltage_drop = potential[edge.from] - potential[edge.to];
            edge.conductance.max(MIN_CONDUCTIVITY) * voltage_drop
        })
        .collect::<Vec<_>>();

    let equation_scale = edge_current
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(potential_span_v)
        .max(1.0);
    let nodal_balance = graph_current_balance(&graph, &edge_current, node_count);
    let max_residual = nodal_balance[1..node_count - 1]
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let residual_norm = max_residual / equation_scale;
    let source_current_a = nodal_balance[0];
    let ground_current_a = -nodal_balance[node_count - 1];
    let current_balance_residual =
        if source_current_a.abs() <= 1.0e-12 && ground_current_a.abs() <= 1.0e-12 {
            0.0
        } else {
            (source_current_a - ground_current_a).abs()
                / source_current_a
                    .abs()
                    .max(ground_current_a.abs())
                    .max(1.0e-12)
        };
    let mut nodal_unscaled_joule_heat = vec![0.0_f64; node_count];
    let integrated_joule_heat_w = graph
        .edges
        .iter()
        .zip(edge_current.iter())
        .map(|(edge, current)| {
            let edge_heat = current * current / edge.conductance.max(MIN_CONDUCTIVITY);
            nodal_unscaled_joule_heat[edge.from] += 0.5 * edge_heat;
            nodal_unscaled_joule_heat[edge.to] += 0.5 * edge_heat;
            edge_heat
        })
        .sum::<f64>()
        .abs();
    let condition_estimate = graph.conductance_span_ratio * graph.max_node_degree.max(1) as f64;

    ElectroThermalPotentialSolve {
        potential,
        graph,
        edge_current,
        nodal_unscaled_joule_heat,
        residual_norm,
        equation_scale,
        current_balance_residual,
        potential_span_v,
        integrated_joule_heat_w,
        condition_estimate,
        source_current_a,
    }
}

fn build_conductance_domain_graph(
    summary: &ElectroThermalAssemblySummary,
    node_count: usize,
) -> ConductanceDomainGraph {
    let edge_count = node_count.saturating_sub(1);
    let conductance = conductivity_profile(summary, edge_count);
    let edges = conductance
        .into_iter()
        .enumerate()
        .map(|(index, conductance)| ConductanceEdge {
            from: index,
            to: index + 1,
            conductance,
        })
        .collect::<Vec<_>>();
    let mut node_degrees = vec![0_usize; node_count];
    for edge in &edges {
        node_degrees[edge.from] += 1;
        node_degrees[edge.to] += 1;
    }
    let max_node_degree = node_degrees.iter().copied().max().unwrap_or_default();
    let mean_node_degree = if node_count == 0 {
        0.0
    } else {
        node_degrees.iter().sum::<usize>() as f64 / node_count as f64
    };
    let (min_conductance, max_conductance) =
        edges
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(min_value, max_value), edge| {
                (
                    min_value.min(edge.conductance),
                    max_value.max(edge.conductance),
                )
            });
    let conductance_span_ratio = if min_conductance.is_finite() && min_conductance > 0.0 {
        (max_conductance / min_conductance).max(1.0)
    } else {
        1.0
    };
    let covered_nodes = node_degrees.iter().filter(|degree| **degree > 0).count();
    let topology_coverage_ratio = if node_count == 0 {
        0.0
    } else if node_count == 1 {
        1.0
    } else {
        covered_nodes as f64 / node_count as f64
    };

    let topology = build_domain_topology(summary, node_count, &edges, &node_degrees);

    ConductanceDomainGraph {
        edges,
        topology,
        node_degrees,
        boundary_node_count: node_count.min(2),
        max_node_degree,
        mean_node_degree,
        conductance_span_ratio,
        topology_coverage_ratio,
    }
}

fn build_domain_topology(
    summary: &ElectroThermalAssemblySummary,
    node_count: usize,
    edges: &[ConductanceEdge],
    node_degrees: &[usize],
) -> ElectroThermalDomainTopology {
    let conductive_node_count = node_degrees.iter().filter(|degree| **degree > 0).count();
    let mapped_voltage_boundary_count = if node_count >= 2 && summary.applied_voltage_v.is_finite()
    {
        2
    } else if node_count == 1 && summary.applied_voltage_v.is_finite() {
        1
    } else {
        0
    };
    let mapped_current_source_count = usize::from(
        summary.joule_heating_scale.is_finite()
            && summary.joule_heating_scale > 0.0
            && summary.applied_voltage_v.abs() > 1.0e-12,
    );
    let material_region_count = summary.region_scale_count.max(1);
    let topology_component_count = conductive_component_count(node_count, edges);
    let voltage_terminals_complete = mapped_voltage_boundary_count >= node_count.min(2);
    let source_boundary_aligned = mapped_current_source_count > 0
        && voltage_terminals_complete
        && topology_component_count == 1;
    let source_boundary_alignment_ratio = if source_boundary_aligned
        || (mapped_current_source_count == 0 && summary.joule_heating_scale <= 1.0e-12)
    {
        1.0
    } else {
        0.0
    };
    let valid_conductance_edges = edges
        .iter()
        .filter(|edge| edge.conductance.is_finite() && edge.conductance > MIN_CONDUCTIVITY)
        .count();
    let domain_conductance_coverage_ratio = if edges.is_empty() {
        if node_count <= 1 {
            1.0
        } else {
            0.0
        }
    } else {
        valid_conductance_edges as f64 / edges.len() as f64
    };
    let material_region_coverage_ratio = if material_region_count > 0 && conductive_node_count > 0 {
        1.0
    } else {
        0.0
    };

    ElectroThermalDomainTopology {
        conductive_node_count,
        conductive_edge_count: edges.len(),
        mapped_voltage_boundary_count,
        mapped_current_source_count,
        material_region_count,
        topology_component_count,
        source_boundary_alignment_ratio,
        domain_conductance_coverage_ratio,
        material_region_coverage_ratio,
    }
}

fn conductive_component_count(node_count: usize, edges: &[ConductanceEdge]) -> usize {
    if node_count == 0 {
        return 0;
    }
    let mut adjacency = vec![Vec::<usize>::new(); node_count];
    for edge in edges {
        adjacency[edge.from].push(edge.to);
        adjacency[edge.to].push(edge.from);
    }
    let mut visited = vec![false; node_count];
    let mut component_count = 0;
    for start in 0..node_count {
        if visited[start] || (adjacency[start].is_empty() && node_count > 1) {
            continue;
        }
        component_count += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(node) = stack.pop() {
            for next in adjacency[node].iter().copied() {
                if !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
    }
    component_count.max(usize::from(node_count == 1))
}

fn accumulate_graph_equation(
    from: usize,
    to: usize,
    conductance: f64,
    node_count: usize,
    matrix: &mut [Vec<f64>],
    rhs: &mut [f64],
    boundary_potential: &[f64],
) {
    for (node, other) in [(from, to), (to, from)] {
        if node == 0 || node + 1 == node_count {
            continue;
        }
        let row = node - 1;
        matrix[row][row] += conductance;
        if other == 0 || other + 1 == node_count {
            rhs[row] += conductance * boundary_potential[other];
        } else {
            let col = other - 1;
            matrix[row][col] -= conductance;
        }
    }
}

fn solve_dense_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    if n == 0 {
        return Vec::new();
    }
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_value = matrix[pivot][pivot].abs();
        for (row, values) in matrix.iter().enumerate().skip(pivot + 1) {
            let candidate = values[pivot].abs();
            if candidate > pivot_value {
                pivot_row = row;
                pivot_value = candidate;
            }
        }
        if pivot_row != pivot {
            matrix.swap(pivot, pivot_row);
            rhs.swap(pivot, pivot_row);
        }
        let diagonal = matrix[pivot][pivot];
        if diagonal.abs() <= MIN_CONDUCTIVITY {
            continue;
        }
        for row in (pivot + 1)..n {
            let factor = matrix[row][pivot] / diagonal;
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            matrix[row][pivot] = 0.0;
            for col in (pivot + 1)..n {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let trailing = ((row + 1)..n)
            .map(|col| matrix[row][col] * solution[col])
            .sum::<f64>();
        let diagonal = matrix[row][row];
        solution[row] = if diagonal.abs() > MIN_CONDUCTIVITY {
            (rhs[row] - trailing) / diagonal
        } else {
            0.0
        };
    }
    solution
}

fn graph_current_balance(
    graph: &ConductanceDomainGraph,
    edge_current: &[f64],
    node_count: usize,
) -> Vec<f64> {
    let mut balance = vec![0.0_f64; node_count];
    for (edge, current) in graph.edges.iter().zip(edge_current.iter()) {
        balance[edge.from] += *current;
        balance[edge.to] -= *current;
    }
    balance
}

fn conductivity_profile(summary: &ElectroThermalAssemblySummary, segment_count: usize) -> Vec<f64> {
    let base = summary
        .base_electrical_conductivity_s_per_m
        .max(MIN_CONDUCTIVITY);
    let spread = summary.conductivity_spread_ratio.max(1.0);
    let centered_spread = spread.sqrt();
    (0..segment_count)
        .map(|index| {
            let xi = if segment_count <= 1 {
                0.5
            } else {
                index as f64 / (segment_count - 1) as f64
            };
            let profile = 1.0 / centered_spread + (centered_spread - 1.0 / centered_spread) * xi;
            let temporal_adjustment = 1.0 + summary.temporal_profile_variation * (0.5 - xi) * 0.1;
            (base * profile * temporal_adjustment).max(MIN_CONDUCTIVITY)
        })
        .collect()
}

fn electro_thermal_domain_topology_diagnostic(
    solve: &ElectroThermalPotentialSolve,
) -> FeaDiagnostic {
    let topology = &solve.graph.topology;
    let severity = if topology.conductive_node_count == solve.potential.len()
        && topology.conductive_edge_count == solve.graph.edges.len()
        && topology.mapped_voltage_boundary_count >= solve.potential.len().min(2)
        && topology.topology_component_count == 1
        && topology.source_boundary_alignment_ratio >= 1.0
        && topology.domain_conductance_coverage_ratio >= 1.0
        && topology.material_region_coverage_ratio >= 1.0
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    FeaDiagnostic {
        code: "FEA_ET_DOMAIN_TOPOLOGY".to_string(),
        severity,
        message: format!(
            "basis=electrical_domain_topology_graph conductive_node_count={} conductive_edge_count={} mapped_voltage_boundary_count={} mapped_current_source_count={} material_region_count={} topology_component_count={} source_boundary_alignment_ratio={} domain_conductance_coverage_ratio={} material_region_coverage_ratio={}",
            topology.conductive_node_count,
            topology.conductive_edge_count,
            topology.mapped_voltage_boundary_count,
            topology.mapped_current_source_count,
            topology.material_region_count,
            topology.topology_component_count,
            topology.source_boundary_alignment_ratio,
            topology.domain_conductance_coverage_ratio,
            topology.material_region_coverage_ratio,
        ),
    }
}

fn electro_thermal_potential_solve_diagnostic(
    node_count: usize,
    solve: &ElectroThermalPotentialSolve,
) -> FeaDiagnostic {
    let severity = if solve.residual_norm <= 1.0e-10 && solve.current_balance_residual <= 1.0e-10 {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };
    FeaDiagnostic {
        code: "FEA_ET_POTENTIAL_SOLVE".to_string(),
        severity,
        message: format!(
            "basis=conductance_domain_graph node_count={} edge_count={} boundary_node_count={} max_node_degree={} mean_node_degree={} conductance_span_ratio={} topology_coverage_ratio={} potential_span_v={} residual_norm={} equation_scale={} current_balance_residual={} integrated_joule_heat_w={} condition_estimate={}",
            node_count,
            solve.graph.edges.len(),
            solve.graph.boundary_node_count,
            solve.graph.max_node_degree,
            solve.graph.mean_node_degree,
            solve.graph.conductance_span_ratio,
            solve.graph.topology_coverage_ratio,
            solve.potential_span_v,
            solve.residual_norm,
            solve.equation_scale,
            solve.current_balance_residual,
            solve.integrated_joule_heat_w,
            solve.condition_estimate,
        ),
    }
}

fn electro_thermal_conduction_conservation_diagnostic(
    node_count: usize,
    solve: &ElectroThermalPotentialSolve,
) -> FeaDiagnostic {
    let coverage_ratio = conduction_graph_coverage_ratio(node_count, solve);
    let ohms_law_residual_ratio = solve
        .graph
        .edges
        .iter()
        .zip(solve.edge_current.iter())
        .map(|(edge, current)| {
            let voltage_drop = solve.potential[edge.from] - solve.potential[edge.to];
            let expected_current = edge.conductance * voltage_drop;
            (current - expected_current).abs()
                / current.abs().max(expected_current.abs()).max(1.0e-12)
        })
        .fold(0.0_f64, f64::max);
    let input_power_w = solve.source_current_a.abs() * solve.potential_span_v;
    let joule_heat_balance_ratio = if input_power_w > 1.0e-12 {
        solve.integrated_joule_heat_w / input_power_w
    } else if solve.integrated_joule_heat_w <= 1.0e-12 {
        1.0
    } else {
        0.0
    };
    let potential_monotonic_edge_fraction = potential_monotonic_edge_fraction(solve);
    let severity = if coverage_ratio >= 1.0
        && ohms_law_residual_ratio <= 1.0e-10
        && (joule_heat_balance_ratio - 1.0).abs() <= 1.0e-10
        && potential_monotonic_edge_fraction >= 1.0
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    FeaDiagnostic {
        code: "FEA_ET_CONDUCTION_CONSERVATION".to_string(),
        severity,
        message: format!(
            "basis=conductance_domain_graph node_count={} edge_count={} ohms_law_residual_ratio={} joule_heat_balance_ratio={} potential_monotonic_edge_fraction={} conduction_graph_coverage_ratio={}",
            node_count,
            solve.graph.edges.len(),
            ohms_law_residual_ratio,
            joule_heat_balance_ratio,
            potential_monotonic_edge_fraction,
            coverage_ratio,
        ),
    }
}

fn electro_thermal_source_coupling_diagnostic(
    summary: &ElectroThermalAssemblySummary,
    joule_heat: &[f64],
    final_temperature: &[f64],
) -> FeaDiagnostic {
    let joule_heat_integral_w = joule_heat.iter().sum::<f64>();
    let target_joule_heat_w = summary.joule_heating_scale.max(0.0);
    let joule_heat_realization_ratio = if target_joule_heat_w > 1.0e-12 {
        joule_heat_integral_w / target_joule_heat_w
    } else if joule_heat_integral_w <= 1.0e-12 {
        1.0
    } else {
        0.0
    };
    let positive_source_count = joule_heat.iter().filter(|value| **value > 0.0).count();
    let joule_source_coverage_ratio = if joule_heat.is_empty() {
        0.0
    } else {
        positive_source_count as f64 / joule_heat.len() as f64
    };
    let temperature_rise = final_temperature
        .iter()
        .map(|value| (value - summary.reference_temperature_k).max(0.0))
        .collect::<Vec<_>>();
    let temperature_span_k = final_temperature.iter().copied().fold(
        (f64::INFINITY, -f64::INFINITY),
        |(min_value, max_value), value| (min_value.min(value), max_value.max(value)),
    );
    let temperature_span_k = if temperature_span_k.0.is_finite() && temperature_span_k.1.is_finite()
    {
        (temperature_span_k.1 - temperature_span_k.0).max(0.0)
    } else {
        0.0
    };
    let thermal_source_residual_ratio = normalized_profile_residual(joule_heat, &temperature_rise);
    let thermal_temperature_source_alignment =
        (1.0 - thermal_source_residual_ratio).clamp(0.0, 1.0);
    let severity = if (joule_heat_realization_ratio - 1.0).abs() <= 1.0e-10
        && joule_source_coverage_ratio >= 1.0
        && thermal_source_residual_ratio <= 1.0e-10
    {
        FeaDiagnosticSeverity::Info
    } else {
        FeaDiagnosticSeverity::Warning
    };

    FeaDiagnostic {
        code: "FEA_ET_THERMAL_SOURCE_COUPLING".to_string(),
        severity,
        message: format!(
            "basis=joule_heat_field joule_heat_integral_w={} joule_heat_realization_ratio={} joule_source_coverage_ratio={} thermal_temperature_source_alignment={} thermal_source_residual_ratio={} temperature_span_k={}",
            joule_heat_integral_w,
            joule_heat_realization_ratio,
            joule_source_coverage_ratio,
            thermal_temperature_source_alignment,
            thermal_source_residual_ratio,
            temperature_span_k,
        ),
    }
}

fn normalized_profile_residual(source: &[f64], response: &[f64]) -> f64 {
    if source.is_empty() || response.is_empty() || source.len() != response.len() {
        return 1.0;
    }
    let source_mean = source.iter().sum::<f64>() / source.len() as f64;
    let response_mean = response.iter().sum::<f64>() / response.len() as f64;
    if source_mean <= 1.0e-12 && response_mean <= 1.0e-12 {
        return 0.0;
    }
    if source_mean <= 1.0e-12 || response_mean <= 1.0e-12 {
        return 1.0;
    }
    source
        .iter()
        .zip(response.iter())
        .map(|(source_value, response_value)| {
            ((source_value / source_mean) - (response_value / response_mean)).abs()
        })
        .fold(0.0_f64, f64::max)
}

fn conduction_graph_coverage_ratio(node_count: usize, solve: &ElectroThermalPotentialSolve) -> f64 {
    if node_count >= 2
        && !solve.graph.edges.is_empty()
        && solve.edge_current.len() == solve.graph.edges.len()
        && solve.potential.len() == node_count
        && solve.potential_span_v.is_finite()
        && solve
            .graph
            .edges
            .iter()
            .all(|edge| edge.conductance.is_finite() && edge.conductance > MIN_CONDUCTIVITY)
    {
        solve.graph.topology_coverage_ratio
    } else {
        0.0
    }
}

fn potential_monotonic_edge_fraction(solve: &ElectroThermalPotentialSolve) -> f64 {
    let edge_count = solve.potential.len().saturating_sub(1);
    if edge_count == 0 {
        return 1.0;
    }
    let signed_span = solve.potential[0] - solve.potential[solve.potential.len() - 1];
    let tolerance = solve.potential_span_v.max(1.0) * 1.0e-12;
    let monotonic_edges = solve
        .potential
        .windows(2)
        .filter(|pair| {
            let drop = pair[0] - pair[1];
            if signed_span >= 0.0 {
                drop >= -tolerance
            } else {
                drop <= tolerance
            }
        })
        .count();
    monotonic_edges as f64 / edge_count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary() -> ElectroThermalAssemblySummary {
        ElectroThermalAssemblySummary {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.8e7,
            resistive_heating_coefficient: 3.5e-4,
            joule_heating_scale: 10.0,
            conductivity_spread_ratio: 1.08,
            temporal_profile_variation: 0.15,
            region_scale_count: 2,
            coupling_fingerprint: 42,
        }
    }

    #[test]
    fn conductance_domain_graph_solve_balances_current() {
        let solve = solve_conductance_domain_graph(&summary(), 16);

        assert!((solve.potential[0] - 36.0).abs() <= 1.0e-9);
        assert!(solve.potential.last().copied().unwrap_or(1.0).abs() <= 1.0e-12);
        assert!(solve.residual_norm <= 1.0e-10);
        assert!(solve.current_balance_residual <= 1.0e-10);
        assert!(solve.integrated_joule_heat_w > 0.0);
        assert_eq!(solve.graph.edges.len(), 15);
        assert_eq!(solve.graph.max_node_degree, 2);
    }

    #[test]
    fn recovered_fields_use_potential_solve_diagnostic() {
        let fields = recover_electro_thermal_fields(Some(&summary()), &[1.0], &[0.01], 48);

        assert_eq!(fields.static_fields.len(), 4);
        assert_eq!(fields.diagnostics.len(), 4);
        assert_eq!(fields.diagnostics[0].code, "FEA_ET_DOMAIN_TOPOLOGY");
        assert!(fields.diagnostics[0]
            .message
            .contains("basis=electrical_domain_topology_graph"));
        assert!(fields.diagnostics[0]
            .message
            .contains("source_boundary_alignment_ratio="));
        assert_eq!(fields.diagnostics[1].code, "FEA_ET_POTENTIAL_SOLVE");
        assert!(fields.diagnostics[1]
            .message
            .contains("current_balance_residual="));
        assert!(fields.diagnostics[1]
            .message
            .contains("basis=conductance_domain_graph"));
        assert_eq!(fields.diagnostics[2].code, "FEA_ET_CONDUCTION_CONSERVATION");
        assert!(fields.diagnostics[2]
            .message
            .contains("ohms_law_residual_ratio="));
        assert!(fields.diagnostics[2]
            .message
            .contains("joule_heat_balance_ratio="));
        assert!(fields.diagnostics[2]
            .message
            .contains("conduction_graph_coverage_ratio="));
        assert_eq!(fields.diagnostics[3].code, "FEA_ET_THERMAL_SOURCE_COUPLING");
        assert!(fields.diagnostics[3]
            .message
            .contains("thermal_temperature_source_alignment="));
    }

    #[test]
    fn recovered_temperature_snapshots_follow_joule_heat_field() {
        let fields = recover_electro_thermal_fields(Some(&summary()), &[1.0], &[0.01], 48);
        let joule_heat = fields.static_fields[3]
            .as_host_f64()
            .expect("joule heat should be a host field");
        let temperature = fields.temperature_snapshots[0]
            .as_host_f64()
            .expect("temperature should be a host field");
        let source_residual = normalized_profile_residual(
            joule_heat,
            &temperature
                .iter()
                .map(|value| value - summary().reference_temperature_k)
                .collect::<Vec<_>>(),
        );

        assert!(source_residual <= 1.0e-12);
        assert!((joule_heat.iter().sum::<f64>() - summary().joule_heating_scale).abs() <= 1.0e-10);
    }
}
