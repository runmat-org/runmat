use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkSuiteGatePolicy {
    #[serde(default)]
    pub require_all_solve_ready: bool,
    #[serde(default)]
    pub require_no_budget_exceeded: bool,
    #[serde(default)]
    pub require_no_missing_surface_source_edges: bool,
    #[serde(default)]
    pub require_all_surface_source_edge_loops_closed: bool,
    #[serde(default)]
    pub require_no_missing_cad_exact_queries: bool,
    #[serde(default)]
    pub require_no_missing_cad_derivative_queries: bool,
    #[serde(default)]
    pub require_no_missing_cad_curvature_queries: bool,
    #[serde(default)]
    pub require_no_rejected_requested_refinement_points: bool,
    #[serde(default)]
    pub require_no_dropped_requested_refinement_points: bool,
    #[serde(default)]
    pub require_no_unrepaired_exact_quality_cavities: bool,
    #[serde(default)]
    pub require_no_fan_fallback_components: bool,
    #[serde(default)]
    pub require_full_boundary_face_recovery: bool,
    #[serde(default)]
    pub require_full_boundary_edge_recovery: bool,
    #[serde(default)]
    pub max_generation_failure_count: Option<usize>,
    #[serde(default)]
    pub max_failed_count: Option<usize>,
    #[serde(default)]
    pub max_total_ms: Option<f64>,
    #[serde(default)]
    pub max_analysis_mesh_json_bytes: Option<usize>,
    #[serde(default)]
    pub max_mesh_evidence_json_bytes: Option<usize>,
}

impl Default for MeshBenchmarkSuiteGatePolicy {
    fn default() -> Self {
        Self {
            require_all_solve_ready: true,
            require_no_budget_exceeded: true,
            require_no_missing_surface_source_edges: true,
            require_all_surface_source_edge_loops_closed: true,
            require_no_missing_cad_exact_queries: true,
            require_no_missing_cad_derivative_queries: true,
            require_no_missing_cad_curvature_queries: true,
            require_no_rejected_requested_refinement_points: true,
            require_no_dropped_requested_refinement_points: true,
            require_no_unrepaired_exact_quality_cavities: true,
            require_no_fan_fallback_components: true,
            require_full_boundary_face_recovery: true,
            require_full_boundary_edge_recovery: true,
            max_generation_failure_count: Some(0),
            max_failed_count: Some(0),
            max_total_ms: None,
            max_analysis_mesh_json_bytes: None,
            max_mesh_evidence_json_bytes: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshBenchmarkSuiteGateViolation {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshBenchmarkSuiteGateResult {
    pub passed: bool,
    pub violation_count: usize,
    pub violations: Vec<MeshBenchmarkSuiteGateViolation>,
}

pub fn evaluate_mesh_benchmark_suite_gate(
    suite: &MeshBenchmarkSuiteReport,
    policy: &MeshBenchmarkSuiteGatePolicy,
) -> MeshBenchmarkSuiteGateResult {
    let mut violations = Vec::<MeshBenchmarkSuiteGateViolation>::new();
    if let Some(max_generation_failure_count) = policy.max_generation_failure_count {
        if suite.summary.generation_failure_count > max_generation_failure_count {
            violations.push(gate_violation(
                "generation_failure_count_exceeded",
                format!(
                    "generation failures {} exceed limit {}",
                    suite.summary.generation_failure_count, max_generation_failure_count
                ),
            ));
        }
    }
    if let Some(max_failed_count) = policy.max_failed_count {
        if suite.summary.failed_count > max_failed_count {
            violations.push(gate_violation(
                "failed_count_exceeded",
                format!(
                    "failed benchmark count {} exceeds limit {}",
                    suite.summary.failed_count, max_failed_count
                ),
            ));
        }
    }
    if policy.require_all_solve_ready
        && suite.summary.solve_ready_count != suite.summary.report_count
    {
        violations.push(gate_violation(
            "not_all_reports_solve_ready",
            format!(
                "solve-ready reports {} do not match report count {}",
                suite.summary.solve_ready_count, suite.summary.report_count
            ),
        ));
    }
    if policy.require_no_budget_exceeded && suite.summary.budget_exceeded_count > 0 {
        violations.push(gate_violation(
            "element_budget_exceeded",
            format!(
                "{} benchmark reports exceeded element budget",
                suite.summary.budget_exceeded_count
            ),
        ));
    }
    if policy.require_no_missing_surface_source_edges {
        let missing_source_edge_count = suite
            .reports
            .iter()
            .map(|report| report.cad.surface_missing_source_edge_count)
            .sum::<usize>();
        if missing_source_edge_count > 0 {
            violations.push(gate_violation(
                "surface_source_edges_missing",
                format!(
                    "{missing_source_edge_count} surface source edges are missing from benchmark reports"
                ),
            ));
        }
    }
    if policy.require_all_surface_source_edge_loops_closed {
        let open_source_edge_loop_count = suite
            .reports
            .iter()
            .map(|report| {
                report
                    .cad
                    .surface_source_edge_loop_count
                    .saturating_sub(report.cad.surface_closed_edge_loop_count)
            })
            .sum::<usize>();
        if open_source_edge_loop_count > 0 {
            violations.push(gate_violation(
                "surface_source_edge_loops_open",
                format!(
                    "{open_source_edge_loop_count} surface source-edge loops are not closed in benchmark reports"
                ),
            ));
        }
    }
    if policy.require_no_missing_cad_exact_queries {
        let missing_exact_query_count = suite
            .reports
            .iter()
            .map(|report| report.cad.missing_exact_query_face_count)
            .sum::<usize>();
        if missing_exact_query_count > 0 {
            violations.push(gate_violation(
                "cad_exact_queries_missing",
                format!(
                    "{missing_exact_query_count} CAD evaluator faces are missing exact query-backed frames in benchmark reports"
                ),
            ));
        }
    }
    if policy.require_no_missing_cad_derivative_queries {
        let missing_derivative_query_count = suite
            .reports
            .iter()
            .map(|report| report.cad.missing_derivative_query_face_count)
            .sum::<usize>();
        if missing_derivative_query_count > 0 {
            violations.push(gate_violation(
                "cad_derivative_queries_missing",
                format!(
                    "{missing_derivative_query_count} CAD evaluator faces are missing derivative queries in benchmark reports"
                ),
            ));
        }
    }
    if policy.require_no_missing_cad_curvature_queries {
        let missing_curvature_query_count = suite
            .reports
            .iter()
            .map(|report| report.cad.missing_curvature_query_face_count)
            .sum::<usize>();
        if missing_curvature_query_count > 0 {
            violations.push(gate_violation(
                "cad_curvature_queries_missing",
                format!(
                    "{missing_curvature_query_count} CAD evaluator faces are missing curvature queries in benchmark reports"
                ),
            ));
        }
    }
    if policy.require_no_rejected_requested_refinement_points {
        let rejected_requested_refinement_count = suite
            .reports
            .iter()
            .map(|report| {
                report
                    .sizing
                    .rejected_requested_tetrahedron_refinement_point_count
            })
            .sum::<usize>();
        if rejected_requested_refinement_count > 0 {
            violations.push(gate_violation(
                "requested_refinement_points_rejected",
                format!(
                    "{rejected_requested_refinement_count} requested refinement points were rejected in benchmark reports"
                ),
            ));
        }
    }
    if policy.require_no_dropped_requested_refinement_points {
        let dropped_requested_refinement_count = suite
            .reports
            .iter()
            .map(|report| {
                report
                    .sizing
                    .dropped_requested_tetrahedron_refinement_point_count
            })
            .sum::<usize>();
        if dropped_requested_refinement_count > 0 {
            violations.push(gate_violation(
                "requested_refinement_points_dropped",
                format!(
                    "{dropped_requested_refinement_count} requested refinement points were dropped after recovery in benchmark reports"
                ),
            ));
        }
    }
    if policy.require_no_unrepaired_exact_quality_cavities {
        let unrepaired_exact_quality_count = suite
            .reports
            .iter()
            .map(|report| report.solve_readiness.unrepaired_exact_quality_total_count)
            .sum::<usize>();
        if unrepaired_exact_quality_count > 0 {
            violations.push(gate_violation(
                "unrepaired_exact_quality_cavities",
                format!(
                    "{unrepaired_exact_quality_count} exact-quality recovery cavities remain unrepaired in benchmark reports"
                ),
            ));
        }
    }
    if policy.require_no_fan_fallback_components {
        let fan_fallback_component_count = suite
            .reports
            .iter()
            .map(|report| report.solve_readiness.fan_fallback_component_count)
            .sum::<usize>();
        if fan_fallback_component_count > 0 {
            violations.push(gate_violation(
                "fan_fallback_components_present",
                format!(
                    "{fan_fallback_component_count} fan fallback components are present in benchmark reports"
                ),
            ));
        }
    }
    if policy.require_full_boundary_face_recovery {
        let incomplete_face_recovery_count = suite
            .reports
            .iter()
            .filter(|report| report.coverage.boundary_face_recovery_ratio + 1.0e-9 < 1.0)
            .count();
        if incomplete_face_recovery_count > 0 {
            violations.push(gate_violation(
                "boundary_face_recovery_incomplete",
                format!(
                    "{incomplete_face_recovery_count} benchmark reports have incomplete boundary face recovery"
                ),
            ));
        }
    }
    if policy.require_full_boundary_edge_recovery {
        let incomplete_edge_recovery_count = suite
            .reports
            .iter()
            .filter(|report| report.coverage.boundary_edge_recovery_ratio + 1.0e-9 < 1.0)
            .count();
        if incomplete_edge_recovery_count > 0 {
            violations.push(gate_violation(
                "boundary_edge_recovery_incomplete",
                format!(
                    "{incomplete_edge_recovery_count} benchmark reports have incomplete boundary edge recovery"
                ),
            ));
        }
    }
    if let (Some(total_ms), Some(max_total_ms)) = (suite.summary.total_ms, policy.max_total_ms) {
        if total_ms > max_total_ms {
            violations.push(gate_violation(
                "total_runtime_exceeded",
                format!("total runtime {total_ms:.3} ms exceeds limit {max_total_ms:.3} ms"),
            ));
        }
    }
    if let (Some(bytes), Some(max_bytes)) = (
        suite.summary.largest_analysis_mesh_json_bytes,
        policy.max_analysis_mesh_json_bytes,
    ) {
        if bytes > max_bytes {
            violations.push(gate_violation(
                "analysis_mesh_artifact_size_exceeded",
                format!("analysis mesh artifact {bytes} bytes exceeds limit {max_bytes} bytes"),
            ));
        }
    }
    if let (Some(bytes), Some(max_bytes)) = (
        suite.summary.largest_mesh_evidence_json_bytes,
        policy.max_mesh_evidence_json_bytes,
    ) {
        if bytes > max_bytes {
            violations.push(gate_violation(
                "mesh_evidence_artifact_size_exceeded",
                format!("mesh evidence artifact {bytes} bytes exceeds limit {max_bytes} bytes"),
            ));
        }
    }
    MeshBenchmarkSuiteGateResult {
        passed: violations.is_empty(),
        violation_count: violations.len(),
        violations,
    }
}

fn gate_violation(
    code: impl Into<String>,
    message: impl Into<String>,
) -> MeshBenchmarkSuiteGateViolation {
    MeshBenchmarkSuiteGateViolation {
        code: code.into(),
        message: message.into(),
    }
}
