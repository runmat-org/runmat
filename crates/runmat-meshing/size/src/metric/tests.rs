use super::*;

#[test]
fn resolver_conservatively_intersects_only_incident_contributions() {
    let edge = id(PersistentEntityKind::Edge, "edge");
    let face = id(PersistentEntityKind::Face, "face");
    let region = id(PersistentEntityKind::Region, "region");
    let unit = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let request = MetricFieldRequest {
        entity_supports: Vec::new(),
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
    assert_eq!(resolved.metric.xx, 4.0);
    assert_eq!(resolved.metric.yy, 1.0);
    assert_eq!(resolved.metric.zz, 1.0);
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
        entity_supports: Vec::new(),
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
        entity_supports: Vec::new(),
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
    PersistentEntityId::new(kind, source.into(), vec!["root".into()])
}

#[test]
fn uniform_resolution_is_entity_independent_and_scoped_requests_cannot_claim_it() {
    let mut request = MetricFieldRequest {
        entity_supports: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3 {
            xx: 4.0,
            yy: 3.0,
            zz: 2.0,
            xy: 0.2,
            xz: 0.1,
            yz: -0.1,
        },
        maximum_grading_ratio: 1.0,
        contributions: Vec::new(),
    };
    let region = id(PersistentEntityKind::Region, "region");
    let field = ResolvedMetricField::new(&request).unwrap();
    assert_eq!(
        field.uniform_evaluation().unwrap().unwrap(),
        field.resolve(&BTreeSet::from([region.clone()])).unwrap()
    );
    request.contributions.push(MetricContribution {
        source: MetricSourceKind::Region,
        scope: MetricContributionScope::Region { region_id: region },
        metric: request.global_metric,
    });
    assert!(ResolvedMetricField::new(&request)
        .unwrap()
        .uniform_evaluation()
        .unwrap()
        .is_none());
    request.maximum_grading_ratio = 0.9;
    assert!(ResolvedMetricField::new(&request).is_err());
}

#[test]
fn topology_multiplicity_preserves_density_and_all_constraint_provenance() {
    for tensor in [
        MetricTensor3::isotropic_length_m(0.125).unwrap(),
        MetricTensor3 {
            xx: 4.0,
            yy: 3.0,
            zz: 2.0,
            xy: 0.5,
            xz: 0.25,
            yz: -0.125,
        },
    ] {
        let mut request = MetricFieldRequest {
            entity_supports: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
            maximum_grading_ratio: 1.5,
            contributions: Vec::new(),
        };
        let mut incident = BTreeSet::new();
        let mut reference = None;
        for count in [1, 16, 4096] {
            while request.contributions.len() < count {
                let entity = id(
                    PersistentEntityKind::Face,
                    &format!("face-{:05}", request.contributions.len()),
                );
                incident.insert(entity.clone());
                request.contributions.push(MetricContribution {
                    source: MetricSourceKind::Face,
                    scope: MetricContributionScope::Entity { entity_id: entity },
                    metric: tensor,
                });
            }
            let evaluation = ResolvedMetricField::new(&request)
                .unwrap()
                .resolve(&incident)
                .unwrap();
            assert_eq!(evaluation.applied_contribution_count, count as u32);
            if let Some(expected) = reference {
                assert_eq!(evaluation.metric, expected);
            } else {
                reference = Some(evaluation.metric);
            }
        }
    }
}

#[test]
fn tensor_intersection_preserves_each_direction_and_equal_cross_terms() {
    let a = MetricTensor3 {
        xx: 4.0,
        yy: 3.0,
        zz: 2.0,
        xy: 0.5,
        xz: 0.25,
        yz: -0.125,
    };
    for b in [
        a,
        MetricTensor3 {
            xx: 8.0,
            yy: 2.0,
            zz: 5.0,
            ..a
        },
        MetricTensor3 {
            xy: -0.5,
            xz: -0.25,
            yz: 0.125,
            ..a
        },
    ] {
        let combined = intersection::intersect(a, b).unwrap();
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    let v = [x as f64, y as f64, z as f64];
                    let density = |m: MetricTensor3| {
                        m.xx * v[0] * v[0]
                            + m.yy * v[1] * v[1]
                            + m.zz * v[2] * v[2]
                            + 2.0 * (m.xy * v[0] * v[1] + m.xz * v[0] * v[2] + m.yz * v[1] * v[2])
                    };
                    assert!(density(combined) >= density(a));
                    assert!(density(combined) >= density(b));
                }
            }
        }
        if a == b {
            assert_eq!(combined, a);
        }
    }
}

#[test]
fn compact_scope_index_matches_exhaustive_incident_selection() {
    let unit = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let mut request = MetricFieldRequest {
        entity_supports: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: unit,
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
    };
    for source in [
        MetricSourceKind::Region,
        MetricSourceKind::Curve,
        MetricSourceKind::Face,
    ] {
        for n in 0..128 {
            request.contributions.push(MetricContribution {
                source,
                scope: MetricContributionScope::Entity {
                    entity_id: id(PersistentEntityKind::Face, &format!("face-{n:03}")),
                },
                metric: unit,
            });
        }
    }
    let index = index::ContributionIndex::new(&request);
    for stride in [1, 3, 17, 200] {
        let incident = (0..150)
            .step_by(stride)
            .map(|n| id(PersistentEntityKind::Face, &format!("face-{n:03}")))
            .collect::<BTreeSet<_>>();
        let expected = request
            .contributions
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let entity = match &c.scope {
                    MetricContributionScope::Entity { entity_id } => entity_id,
                    MetricContributionScope::Region { region_id } => region_id,
                };
                incident.contains(entity).then_some(i)
            })
            .collect::<Vec<_>>();
        assert_eq!(index.applicable(&request, &incident), expected);
    }
}

fn graded_request(
    ratio: f64,
    supported: bool,
) -> (MetricFieldRequest, PersistentEntityId, PersistentEntityId) {
    let edge = id(PersistentEntityKind::Edge, "edge");
    let face = id(PersistentEntityKind::Face, "face");
    let request = MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(10.0).unwrap(),
        maximum_grading_ratio: ratio,
        contributions: vec![MetricContribution {
            source: MetricSourceKind::Proximity,
            scope: MetricContributionScope::Entity {
                entity_id: edge.clone(),
            },
            metric: MetricTensor3::isotropic_length_m(0.1).unwrap(),
        }],
        entity_supports: if supported {
            vec![MetricEntitySupport {
                entity_id: edge.clone(),
                points_m: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            }]
        } else {
            Vec::new()
        },
    };
    (request, edge, face)
}

fn length_at(
    resolver: &ResolvedMetricField,
    scope: &BTreeSet<PersistentEntityId>,
    point: [f64; 3],
) -> f64 {
    let constant = resolver.resolve_constant(scope).unwrap().metric;
    let local = resolver.local_sources(scope).unwrap();
    resolver
        .evaluate_graded(constant, &local, point)
        .unwrap()
        .xx
        .sqrt()
        .recip()
}

#[test]
fn supported_edge_sizing_is_fine_at_the_edge_and_coarsens_with_distance() {
    let (request, edge, face) = graded_request(1.3, true);
    let resolver = ResolvedMetricField::new(&request).unwrap();
    let scope = BTreeSet::from([edge, face]);
    assert_eq!(resolver.resolve_constant(&scope).unwrap().metric.xx, 0.01);
    let at_edge = length_at(&resolver, &scope, [0.5, 0.0, 0.0]);
    let near = length_at(&resolver, &scope, [0.5, 1.0, 0.0]);
    let far = length_at(&resolver, &scope, [0.5, 100.0, 0.0]);
    assert!((at_edge - 0.1).abs() < 1e-12);
    assert!((near - (0.1 + 0.3 * 1.0)).abs() < 1e-9);
    // Capped by the uniform part: never coarser than the global size.
    assert!((far - 10.0).abs() < 1e-9);
    // Distance to the support segment, including past its end.
    let past_end = length_at(&resolver, &scope, [3.0, 0.0, 0.0]);
    assert!((past_end - (0.1 + 0.3 * 2.0)).abs() < 1e-9);
}

#[test]
fn unsupported_or_ungraded_edge_sizing_applies_uniformly() {
    for (ratio, supported) in [(1.3, false), (1.0, true)] {
        let (request, edge, face) = graded_request(ratio, supported);
        let resolver = ResolvedMetricField::new(&request).unwrap();
        let scope = BTreeSet::from([edge, face]);
        let far = length_at(&resolver, &scope, [0.5, 100.0, 0.0]);
        assert!(
            (far - 0.1).abs() < 1e-12,
            "ratio {ratio} supported {supported}"
        );
    }
}

#[test]
fn entity_supports_must_be_ordered_finite_edges_or_vertices() {
    let (mut request, _, face) = graded_request(1.3, true);
    request.entity_supports[0].points_m[0][1] = f64::NAN;
    assert!(request.validate().is_err());
    let (mut request, _, _) = graded_request(1.3, true);
    request.entity_supports[0].entity_id = face;
    assert!(request.validate().is_err());
    let (mut request, _, _) = graded_request(1.3, true);
    request
        .entity_supports
        .push(request.entity_supports[0].clone());
    assert!(request.validate().is_err());
}
