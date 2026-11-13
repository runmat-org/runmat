use std::path::Path;

use runmat_execution_runner_native::{
    mesh_source_with_native_execution, NativeMeshingContactDefinition,
    NativeMeshingPartitionPolicy, NativeMeshingSourceError, NativeMeshingSourceRequest,
};
use runmat_meshing_core::{
    predicate::{
        tetrahedron_circumsphere, tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian,
        tetrahedron_volume,
    },
    tolerance::MeshingTolerance,
    SolverMeshArtifact,
};
use runmat_meshing_core::{ElementOrder as CanonicalElementOrder, MeshingRequestSettings};

use super::admission::{classify as classify_admission, import_options};
use super::contract::{
    AdmissionScenario, ElementOrder, FailureExpectation, MeshExpectation, MeshScenario, SourceUnits,
};
use super::report::{MeshQualityReport, MeshReport, ObservedFailure};

#[derive(Clone, Debug, PartialEq)]
struct SuccessfulExecution {
    canonical_digest: String,
    node_count: usize,
    element_count: usize,
    boundary_face_count: usize,
    region_count: usize,
    quality: MeshQualityReport,
}

#[derive(Debug)]
enum MeshRunError {
    Source(NativeMeshingSourceError),
    Accounting(runmat_simulation_usage::LedgerError),
    InvalidResult(String),
}

impl From<NativeMeshingSourceError> for MeshRunError {
    fn from(error: NativeMeshingSourceError) -> Self {
        Self::Source(error)
    }
}

impl std::fmt::Display for MeshRunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(error) => error.fmt(formatter),
            Self::Accounting(error) => write!(formatter, "mesh quality accounting failed: {error}"),
            Self::InvalidResult(error) => write!(
                formatter,
                "completed meshing result failed artifact/evidence validation: {error}"
            ),
        }
    }
}

pub fn execute(
    source: &Path,
    source_sha256: [u8; 32],
    units: SourceUnits,
    admission: &AdmissionScenario,
    scenario: &MeshScenario,
    request: &super::contract::MeshRequest,
) -> MeshReport {
    let mut baseline = None;
    let mut actual_failure = None;
    let mut detail = None;
    let mut executions = 0;

    'layouts: for layout in &request.partition_layouts {
        for _ in 0..request.repeats {
            executions += 1;
            match run_once(
                source,
                source_sha256,
                units,
                admission,
                scenario,
                request,
                *layout,
            ) {
                Ok(execution) => match &scenario.expected {
                    MeshExpectation::Success { regions } => {
                        if execution.region_count != *regions {
                            detail = Some(format!(
                                "expected {regions} mesh regions, observed {}",
                                execution.region_count
                            ));
                            break 'layouts;
                        }
                        if let Some(expected) = &baseline {
                            if expected != &execution {
                                detail = Some(format!(
                                    "repeat or partition layout changed the canonical mesh: expected {expected:?}, observed {execution:?}"
                                ));
                                break 'layouts;
                            }
                        } else {
                            baseline = Some(execution);
                        }
                    }
                    MeshExpectation::Failure { .. } => {
                        detail = Some("meshing succeeded but a typed failure was required".into());
                        baseline = Some(execution);
                        break 'layouts;
                    }
                },
                Err(error) => {
                    let classified = classify_run_error(&error);
                    match &scenario.expected {
                        MeshExpectation::Failure { failure }
                            if classified
                                .as_ref()
                                .is_some_and(|actual| matches_expectation(actual, failure)) =>
                        {
                            if actual_failure
                                .as_ref()
                                .is_some_and(|previous| previous != classified.as_ref().unwrap())
                            {
                                detail = Some(
                                    "repeat executions returned different typed failures".into(),
                                );
                                break 'layouts;
                            }
                            actual_failure = classified;
                        }
                        _ => {
                            actual_failure = classified;
                            detail = Some(error.to_string());
                            break 'layouts;
                        }
                    }
                }
            }
        }
    }

    let passed = detail.is_none()
        && match &scenario.expected {
            MeshExpectation::Success { .. } => baseline.is_some() && actual_failure.is_none(),
            MeshExpectation::Failure { failure } => {
                baseline.is_none()
                    && actual_failure
                        .as_ref()
                        .is_some_and(|actual| matches_expectation(actual, failure))
            }
        };
    let (canonical_digest, node_count, element_count, boundary_face_count, region_count, quality) =
        baseline
            .map(|execution| {
                (
                    Some(execution.canonical_digest),
                    Some(execution.node_count),
                    Some(execution.element_count),
                    Some(execution.boundary_face_count),
                    Some(execution.region_count),
                    Some(execution.quality),
                )
            })
            .unwrap_or((None, None, None, None, None, None));

    MeshReport {
        id: scenario.id.clone(),
        outcome: if canonical_digest.is_some() {
            "success".into()
        } else {
            "failure".into()
        },
        canonical_digest,
        node_count,
        element_count,
        boundary_face_count,
        region_count,
        quality,
        failure: actual_failure,
        executions,
        passed,
        detail,
    }
}

fn run_once(
    source: &Path,
    source_sha256: [u8; 32],
    units: SourceUnits,
    admission: &AdmissionScenario,
    scenario: &MeshScenario,
    request: &super::contract::MeshRequest,
    layout: super::contract::PartitionLayout,
) -> Result<SuccessfulExecution, MeshRunError> {
    let mut settings = MeshingRequestSettings {
        target_edge_length_m: runmat_types::physical::Length::new(request.target_size_m),
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(
            request.maximum_deviation_m,
        ),
        element_order: match request.element_order {
            ElementOrder::Tet4 => CanonicalElementOrder::Tet4,
            ElementOrder::Tet10 => CanonicalElementOrder::Tet10,
        },
        deterministic_seed: request.seed,
        ..MeshingRequestSettings::default()
    };
    settings.limits.maximum_elements = request.maximum_elements;
    let result = mesh_source_with_native_execution(NativeMeshingSourceRequest {
        source: source.to_path_buf(),
        import_options: import_options(units, &admission.healing),
        contact_definitions: scenario
            .contacts
            .iter()
            .map(|contact| NativeMeshingContactDefinition {
                side_a_face_selectors: contact.side_a_faces.clone(),
                side_b_face_selectors: contact.side_b_faces.clone(),
            })
            .collect(),
        expected_source_sha256: Some(source_sha256),
        settings,
        partition_policy: NativeMeshingPartitionPolicy {
            preferred_edges_per_partition: layout.edges,
            preferred_faces_per_partition: layout.faces,
        },
    })?;
    result
        .dag_result
        .evidence
        .validate(&result.dag_result.artifact)
        .map_err(|error| MeshRunError::InvalidResult(error.to_string()))?;
    let artifact = result.dag_result.artifact;
    let quality = measure_quality(&artifact)?;
    Ok(SuccessfulExecution {
        canonical_digest: hex_digest(artifact.canonical_digest.bytes()),
        node_count: artifact.topology.nodes.len(),
        element_count: artifact.topology.volume_elements.len(),
        boundary_face_count: artifact.topology.boundary_faces.len(),
        region_count: artifact.topology.regions.len(),
        quality,
    })
}

fn measure_quality(artifact: &SolverMeshArtifact) -> Result<MeshQualityReport, MeshRunError> {
    let nodes = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut total_volume_m3 = 0.0;
    let mut minimum_scaled_jacobian = f64::INFINITY;
    let mut maximum_edge_aspect_ratio = 0.0_f64;
    let mut maximum_radius_edge_ratio = 0.0_f64;

    for element in &artifact.topology.volume_elements {
        let corner_ids = element.node_ids.get(..4).ok_or_else(|| {
            MeshRunError::InvalidResult(format!(
                "volume element {} has fewer than four corner nodes",
                element.element_id
            ))
        })?;
        let point = |index: usize| {
            nodes.get(&corner_ids[index]).copied().ok_or_else(|| {
                MeshRunError::InvalidResult(format!(
                    "volume element {} refers to missing node {}",
                    element.element_id, corner_ids[index]
                ))
            })
        };
        let points = [point(0)?, point(1)?, point(2)?, point(3)?];
        let volume = tetrahedron_volume(points);
        let scaled_jacobian =
            tetrahedron_scaled_jacobian(points).map_err(MeshRunError::Accounting)?;
        let edge_aspect_ratio = tetrahedron_edge_aspect_ratio(points);
        let (_, radius_squared) = tetrahedron_circumsphere(
            points,
            MeshingTolerance {
                absolute_m: runmat_types::physical::Length::new(0.0),
                relative: 0.0,
            },
        )
        .ok_or_else(|| {
            MeshRunError::InvalidResult(format!(
                "volume element {} has no finite circumsphere",
                element.element_id
            ))
        })?;
        let minimum_edge = minimum_edge_length(points);
        let radius_edge_ratio = radius_squared.sqrt() / minimum_edge;
        if !volume.is_finite()
            || volume <= 0.0
            || !scaled_jacobian.is_finite()
            || scaled_jacobian <= 0.0
            || !edge_aspect_ratio.is_finite()
            || !radius_edge_ratio.is_finite()
        {
            return Err(MeshRunError::InvalidResult(format!(
                "volume element {} has invalid measured quality",
                element.element_id
            )));
        }
        total_volume_m3 += volume;
        minimum_scaled_jacobian = minimum_scaled_jacobian.min(scaled_jacobian);
        maximum_edge_aspect_ratio = maximum_edge_aspect_ratio.max(edge_aspect_ratio);
        maximum_radius_edge_ratio = maximum_radius_edge_ratio.max(radius_edge_ratio);
    }
    if !total_volume_m3.is_finite() || total_volume_m3 <= 0.0 {
        return Err(MeshRunError::InvalidResult(
            "solver mesh has no positive finite volume".into(),
        ));
    }
    Ok(MeshQualityReport {
        total_volume_m3,
        minimum_scaled_jacobian,
        maximum_edge_aspect_ratio,
        maximum_radius_edge_ratio,
    })
}

fn minimum_edge_length(points: [[f64; 3]; 4]) -> f64 {
    let mut minimum = f64::INFINITY;
    for left in 0..4 {
        for right in (left + 1)..4 {
            let squared = (0..3)
                .map(|axis| (points[left][axis] - points[right][axis]).powi(2))
                .sum::<f64>();
            minimum = minimum.min(squared.sqrt());
        }
    }
    minimum
}

fn classify_run_error(error: &MeshRunError) -> Option<ObservedFailure> {
    let error = match error {
        MeshRunError::Accounting(cause) => {
            return Some(ObservedFailure::Accounting { cause: *cause });
        }
        MeshRunError::InvalidResult(_) => return None,
        MeshRunError::Source(error) => error,
    };
    if let Some(admission) = error.admission_error() {
        return classify_admission(admission)
            .map(|category| ObservedFailure::Admission { category });
    }
    error
        .meshing_failure()
        .map(|failure| ObservedFailure::Meshing {
            category: failure.category,
            stage: failure.stage,
            operation: failure.operation,
            entity_ids: failure.entity_ids.clone(),
            detail: failure.detail().map(str::to_owned),
            remediation: failure.remediation.clone(),
        })
}

fn matches_expectation(actual: &ObservedFailure, expected: &FailureExpectation) -> bool {
    match (actual, expected) {
        (
            ObservedFailure::Admission { category: actual },
            FailureExpectation::Admission { category: expected },
        ) => actual == expected,
        (
            ObservedFailure::Meshing {
                category: actual_category,
                stage: actual_stage,
                ..
            },
            FailureExpectation::Meshing {
                category: expected_category,
                stage: expected_stage,
            },
        ) => actual_category == expected_category && actual_stage == expected_stage,
        _ => false,
    }
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    encoded
}
