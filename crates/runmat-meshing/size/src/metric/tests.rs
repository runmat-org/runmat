use super::*;

#[test]
fn resolver_conservatively_intersects_only_incident_contributions() {
    let edge = id(PersistentEntityKind::Edge, "edge");
    let face = id(PersistentEntityKind::Face, "face");
    let region = id(PersistentEntityKind::Region, "region");
    let unit = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let request = MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: unit,
        maximum_grading_ratio: 1.25,
        contributions: vec![
            MetricContribution {
                source: MetricSourceKind::Region,
                scope: MetricContributionScope::Region {
                    region_id: region.clone(),
                },
                metric: unit,
            },
            MetricContribution {
                source: MetricSourceKind::Curve,
                scope: MetricContributionScope::Entity {
                    entity_id: edge.clone(),
                },
                metric: MetricTensor3 {
                    xx: 4.0,
                    yy: 1.0,
                    zz: 1.0,
                    xy: 0.0,
                    xz: 0.0,
                    yz: 0.0,
                },
            },
            MetricContribution {
                source: MetricSourceKind::Face,
                scope: MetricContributionScope::Entity { entity_id: face },
                metric: unit,
            },
        ],
    };
    let resolver = ResolvedMetricField::new(&request).unwrap();
    let resolved = resolver.resolve(&BTreeSet::from([edge, region])).unwrap();
    assert_eq!(resolved.metric.xx, 6.0);
    assert_eq!(resolved.metric.yy, 3.0);
    assert_eq!(resolved.metric.zz, 3.0);
    assert_eq!(
        resolved.active_sources,
        vec![
            MetricSourceKind::Global,
            MetricSourceKind::Region,
            MetricSourceKind::Curve,
        ]
    );
    assert_eq!(resolved.clipped_contribution_count, 0);
    assert_eq!(resolved.rejected_contribution_count, 0);
    assert_eq!(resolved.applied_contribution_count, 2);
}

#[test]
fn contract_rejects_duplicate_and_ambiguous_global_contributions() {
    let unit = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let contribution = MetricContribution {
        source: MetricSourceKind::Curve,
        scope: MetricContributionScope::Entity {
            entity_id: id(PersistentEntityKind::Edge, "edge"),
        },
        metric: unit,
    };
    let mut request = MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: unit,
        maximum_grading_ratio: 1.25,
        contributions: vec![contribution.clone(), contribution],
    };
    assert_eq!(
        request.validate().unwrap_err().field,
        "metric contributions"
    );

    request.contributions = vec![MetricContribution {
        source: MetricSourceKind::Global,
        scope: MetricContributionScope::Entity {
            entity_id: id(PersistentEntityKind::Edge, "edge"),
        },
        metric: unit,
    }];
    assert_eq!(
        request.validate().unwrap_err().field,
        "metric contribution source"
    );
}

#[test]
fn request_intersects_and_canonicalizes_derived_contributions() {
    let edge_a = id(PersistentEntityKind::Edge, "edge-a");
    let edge_b = id(PersistentEntityKind::Edge, "edge-b");
    let unit = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let request = MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: unit,
        maximum_grading_ratio: 1.25,
        contributions: Vec::new(),
    };
    let contribution = |entity_id| MetricContribution {
        source: MetricSourceKind::Curve,
        scope: MetricContributionScope::Entity { entity_id },
        metric: unit,
    };

    let resolved = request
        .intersect_contributions([
            contribution(edge_b),
            contribution(edge_a.clone()),
            contribution(edge_a),
        ])
        .unwrap();

    assert_eq!(resolved.contributions.len(), 2);
    let MetricContributionScope::Entity { entity_id: first } = &resolved.contributions[0].scope
    else {
        panic!("derived curve contribution must remain entity-scoped")
    };
    let MetricContributionScope::Entity { entity_id: second } = &resolved.contributions[1].scope
    else {
        panic!("derived curve contribution must remain entity-scoped")
    };
    assert!(first < second);
}

fn id(kind: PersistentEntityKind, source: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source.into(),
        assembly_path: vec!["root".into()],
    }
}
