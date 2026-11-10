use super::*;

fn digest(byte: u8) -> StableDigest {
    StableDigest::from_bytes([byte; 32])
}

fn algorithms() -> AlgorithmVersionSet {
    AlgorithmVersionSet {
        geometry: "geometry/v2".into(),
        curve: "curve-cdt/v2".into(),
        surface: "surface-cdt/v2".into(),
        plc: "plc/v2".into(),
        tetrahedron: "tetrahedron-cdt/v2".into(),
        optimization: "optimization/v2".into(),
        validation: "validation/v2".into(),
    }
}

fn request() -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: ElementOrder::Tet10,
        dimension: crate::MeshDimension::Solid,
        deterministic_seed: 7,
        algorithms: algorithms(),
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(1.0e-8),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-10),
            model_relative_term: 1.0e-9,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-5),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        },
        metric: MetricFieldRequest {
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.01).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: vec![MetricContribution {
                source: MetricSourceKind::Face,
                scope: MetricContributionScope::Entity {
                    entity_id: PersistentEntityId::new(
                        PersistentEntityKind::Face,
                        "face:17".into(),
                        vec!["root".into(), "instance:2".into()],
                    ),
                },
                metric: MetricTensor3::isotropic_length_m(0.002).unwrap(),
            }],
            cells_across_gap: None,
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-5),
                maximum_tangent_change_degrees: 5.0,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 1.5,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_edge_length: 0.1,
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-5),
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 2.0,
                minimum_scaled_jacobian: 0.05,
                maximum_metric_edge_length: 1.5,
            },
        },
        limits: MeshingSizeLimits {
            maximum_nodes: 2_000_000,
            maximum_elements: 10_000_000,
            maximum_recursion_depth: 256,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 100,
            maximum_work_units_between_checks: 4096,
        },
        size_controls: Vec::new(),
    }
}

#[test]
fn canonical_request_round_trips_and_rejects_unknown_fields() {
    let request = request();
    request.validate().unwrap();
    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: MeshingRequest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, request);

    let mut value = serde_json::to_value(&request).unwrap();
    value["backend"] = serde_json::json!("removed");
    assert!(serde_json::from_value::<MeshingRequest>(value).is_err());
}

#[test]
fn canonical_request_rejects_non_finite_and_unbounded_inputs() {
    let mut invalid = request();
    invalid.tolerance.source_tolerance_m = runmat_types::physical::Length::new(f64::NAN);
    assert_eq!(invalid.validate().unwrap_err().field, "source_tolerance_m");

    let mut invalid = request();
    invalid.limits.maximum_nodes = 0;
    assert_eq!(invalid.validate().unwrap_err().field, "meshing size limits");

    let mut invalid = request();
    invalid.quality.curve.minimum_metric_edge_length = 2.0;
    assert_eq!(
        invalid.validate().unwrap_err().field,
        "curve quality targets"
    );

    let mut invalid = request();
    invalid.quality.surface.minimum_metric_edge_length = 1.01;
    assert_eq!(
        invalid.validate().unwrap_err().field,
        "surface quality targets"
    );

    let mut invalid = request();
    let entity_id = PersistentEntityId::new(
        PersistentEntityKind::Face,
        "face:overflow".into(),
        vec!["root".into()],
    );
    invalid.metric.contributions = (0..=65_536)
        .map(|_| MetricContribution {
            source: MetricSourceKind::Face,
            scope: MetricContributionScope::Entity {
                entity_id: entity_id.clone(),
            },
            metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
        })
        .collect();
    assert_eq!(
        invalid.validate().unwrap_err().field,
        "metric contributions"
    );
}

#[test]
fn metric_requires_a_symmetric_positive_definite_tensor() {
    let indefinite = MetricTensor3 {
        xx: 1.0,
        yy: 1.0,
        zz: -1.0,
        xy: 0.0,
        xz: 0.0,
        yz: 0.0,
    };
    assert_eq!(indefinite.validate().unwrap_err().field, "metric tensor");
    assert!(MetricTensor3::isotropic_length_m(0.0).is_err());

    let overflowing = MetricTensor3 {
        xx: f64::MAX,
        yy: f64::MAX,
        zz: f64::MAX,
        xy: 0.0,
        xz: 0.0,
        yz: 0.0,
    };
    assert!(overflowing.validate().is_err());
}

#[test]
fn metric_contributions_are_typed_unique_and_canonical() {
    let mut metric = request().metric;
    let curve_id = PersistentEntityId::new(
        PersistentEntityKind::Edge,
        "edge:1".into(),
        vec!["root".into()],
    );
    let region_id = PersistentEntityId::new(
        PersistentEntityKind::Region,
        "region:1".into(),
        vec!["root".into()],
    );
    metric.contributions = vec![
        MetricContribution {
            source: MetricSourceKind::Region,
            scope: MetricContributionScope::Region { region_id },
            metric: MetricTensor3::isotropic_length_m(0.008).unwrap(),
        },
        MetricContribution {
            source: MetricSourceKind::Curve,
            scope: MetricContributionScope::Entity {
                entity_id: curve_id,
            },
            metric: MetricTensor3::isotropic_length_m(0.004).unwrap(),
        },
    ];
    metric.validate().unwrap();

    metric.contributions.swap(0, 1);
    assert_eq!(metric.validate().unwrap_err().field, "metric contributions");
    metric.contributions[0] = metric.contributions[1].clone();
    assert_eq!(metric.validate().unwrap_err().field, "metric contributions");

    let mut wrong_region = metric.contributions[0].clone();
    wrong_region.scope = MetricContributionScope::Region {
        region_id: PersistentEntityId::new(
            PersistentEntityKind::Face,
            "face:not-region".into(),
            vec!["root".into()],
        ),
    };
    assert_eq!(
        wrong_region.validate().unwrap_err().field,
        "metric contribution region"
    );
}

#[test]
fn geometry_revision_and_persistent_identity_are_bounded() {
    GeometryRevisionRef {
        source_unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
            runmat_geometry_core::LengthUnit::Meter,
            "canonical solver fixture",
        ),
        source_digest: digest(3),
        geometry_revision: 4,
        persistent_mapping_version: 2,
    }
    .validate()
    .unwrap();
    assert!(GeometryRevisionRef {
        source_unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
            runmat_geometry_core::LengthUnit::Meter,
            "canonical solver fixture"
        ),
        source_digest: StableDigest::ZERO,
        geometry_revision: 4,
        persistent_mapping_version: 2,
    }
    .validate()
    .is_err());

    let invalid = PersistentEntityId::new(PersistentEntityKind::Face, " face ".into(), Vec::new());
    assert!(invalid.validate().is_err());
}

#[test]
fn typed_failure_round_trips_and_requires_canonical_diagnostics() {
    let failure = MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category: MeshingFailureCategory::QualityTargetUnreachable,
        stage: MeshingStageKind::Optimization,
        operation: MeshingOperation::Optimize,
        entity_ids: vec![PersistentEntityId::new(
            PersistentEntityKind::Solid,
            "solid:1".into(),
            vec!["root".into()],
        )],
        witnesses: vec![GeometricWitness::Point {
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([0.0, 1.0, 2.0]),
        }],
        request_values: vec![MeshingDiagnosticEntry {
            name: "minimum_scaled_jacobian".into(),
            value: MeshingDiagnosticValue::Scalar(0.1),
            unit: None,
        }],
        achieved_values: vec![MeshingDiagnosticEntry {
            name: "minimum_scaled_jacobian".into(),
            value: MeshingDiagnosticValue::Scalar(0.02),
            unit: None,
        }],
        accounting_error: None,
        remediation: "relax the quality target or resolve the named geometric feature".into(),
    };
    failure.validate().unwrap();
    let encoded = serde_json::to_vec(&failure).unwrap();
    assert_eq!(
        serde_json::from_slice::<MeshingFailure>(&encoded).unwrap(),
        failure
    );

    let mut unsorted = failure;
    unsorted.request_values.insert(
        0,
        MeshingDiagnosticEntry {
            name: "z_value".into(),
            value: MeshingDiagnosticValue::Count(1),
            unit: None,
        },
    );
    assert!(unsorted.validate().is_err());

    let mut mismatched = unsorted;
    mismatched
        .request_values
        .sort_by(|left, right| left.name.cmp(&right.name));
    mismatched.stage = MeshingStageKind::CurveMesh;
    assert_eq!(
        mismatched.validate().unwrap_err().field,
        "meshing failure operation"
    );
}

#[test]
fn cancellation_policy_has_a_runtime_signal_boundary() {
    request().cancellation.validate().unwrap();
    assert!(!NeverCancelled.is_cancelled());
}
