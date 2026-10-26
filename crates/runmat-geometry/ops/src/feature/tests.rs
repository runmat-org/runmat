use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    DerivedGeometryDefinition, GeneratedGeometryDefinition, GeometryAuthoringValue,
    GeometryDefinition, GeometryDefinitionRef, GeometryDefinitionSource, GeometryDigest,
    GeometryFeature, GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef,
    GeometryParameterDefinition, GeometryParameterType, GeometryRegionRole,
    GeometryScalarExpression, GeometryScalarOperator, PersistentEntityId,
};

use super::provider::resolve_provider_selection;
use super::*;

fn scalar(value: f64) -> GeometryScalarExpression {
    GeometryScalarExpression::Literal { value }
}

fn parameter(name: &str) -> GeometryScalarExpression {
    GeometryScalarExpression::Parameter { name: name.into() }
}

fn capabilities(
    kinds: impl IntoIterator<Item = GeometryFeatureKind>,
) -> GeometryFeatureCapabilities {
    GeometryFeatureCapabilities {
        supported: kinds.into_iter().collect::<BTreeSet<_>>(),
        maximum_features: 16,
        maximum_instances_per_feature: 8,
        maximum_boolean_operands: 8,
    }
}

#[test]
fn parameters_are_bound_once_and_feature_values_are_resolved() {
    let definitions = vec![GeometryParameterDefinition {
        parameter_id: "width".into(),
        parameter_type: GeometryParameterType::Length,
        required: true,
        default: None,
    }];
    let bindings = BTreeMap::from([(
        "width".into(),
        GeometryAuthoringValue::Scalar(GeometryScalarExpression::Binary {
            operator: GeometryScalarOperator::Multiply,
            left: Box::new(scalar(2.0)),
            right: Box::new(scalar(0.5)),
        }),
    )]);
    let parameters = resolve_geometry_parameters(&definitions, &bindings).unwrap();
    let graph = GeometryFeatureGraph {
        features: vec![GeometryFeature {
            feature_id: "stock".into(),
            operation: GeometryFeatureOperation::Box {
                width_m: parameter("width"),
                depth_m: scalar(2.0),
                height_m: scalar(3.0),
            },
        }],
        result: GeometryFeatureRef("stock".into()),
    };
    let plan = plan_feature_graph(
        &graph,
        &parameters,
        &capabilities([GeometryFeatureKind::Box]),
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.resource_bounds.maximum_live_artifacts, 1);
    assert!(matches!(
        plan.steps[0].operation,
        ResolvedGeometryFeatureOperation::Box { width_m, .. } if width_m.canonical_value() == 1.0
    ));
}

#[test]
fn parameter_references_resolve_defaults_to_canonical_literal_values() {
    let definitions = vec![GeometryParameterDefinition {
        parameter_id: "width".into(),
        parameter_type: GeometryParameterType::Length,
        required: false,
        default: Some(GeometryAuthoringValue::Scalar(
            GeometryScalarExpression::Binary {
                operator: GeometryScalarOperator::Multiply,
                left: Box::new(scalar(2.0)),
                right: Box::new(scalar(0.5)),
            },
        )),
    }];
    let parameters = resolve_geometry_parameters(&definitions, &BTreeMap::new()).unwrap();

    assert_eq!(
        parameters
            .resolve_value(&GeometryAuthoringValue::Parameter("width".into()))
            .unwrap(),
        GeometryAuthoringValue::Scalar(GeometryScalarExpression::Canonical {
            value: runmat_types::physical::DimensionedScalar::try_from_descriptor(
                runmat_types::physical::DynamicDimension::quantity_type(
                    runmat_types::physical::QuantityKind::Length
                ),
                1.0
            )
            .unwrap(),
        })
    );
}

#[test]
fn live_artifact_bound_releases_unreferenced_intermediate_results() {
    let parameters = resolve_geometry_parameters(&[], &BTreeMap::new()).unwrap();
    let graph = GeometryFeatureGraph {
        features: vec![
            GeometryFeature {
                feature_id: "stock".into(),
                operation: GeometryFeatureOperation::Box {
                    width_m: scalar(1.0),
                    depth_m: scalar(1.0),
                    height_m: scalar(1.0),
                },
            },
            GeometryFeature {
                feature_id: "unused".into(),
                operation: GeometryFeatureOperation::Sphere {
                    radius_m: scalar(1.0),
                    center_m: [scalar(0.0), scalar(0.0), scalar(0.0)],
                },
            },
            GeometryFeature {
                feature_id: "result".into(),
                operation: GeometryFeatureOperation::Transform {
                    source: GeometryFeatureRef("stock".into()),
                    matrix: (0..16)
                        .map(|index| {
                            scalar(if matches!(index, 0 | 5 | 10 | 15) {
                                1.0
                            } else {
                                0.0
                            })
                        })
                        .collect(),
                },
            },
        ],
        result: GeometryFeatureRef("result".into()),
    };
    let plan = plan_feature_graph(
        &graph,
        &parameters,
        &capabilities([
            GeometryFeatureKind::Box,
            GeometryFeatureKind::Sphere,
            GeometryFeatureKind::Transform,
        ]),
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.resource_bounds.maximum_live_artifacts, 2);
}

// witness: FEA-AUTHORING-GEOMETRY-002/bounded_feature_graph @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Unsupported operations and provider bounds fail before execution; the full bounded
//   operation set is not enumerated.
#[test]
fn unsupported_operations_and_provider_bounds_fail_before_execution() {
    let parameters = resolve_geometry_parameters(&[], &BTreeMap::new()).unwrap();
    let graph = GeometryFeatureGraph {
        features: vec![GeometryFeature {
            feature_id: "ball".into(),
            operation: GeometryFeatureOperation::Sphere {
                radius_m: scalar(1.0),
                center_m: [scalar(0.0), scalar(0.0), scalar(0.0)],
            },
        }],
        result: GeometryFeatureRef("ball".into()),
    };
    let error = plan_feature_graph(
        &graph,
        &parameters,
        &capabilities([GeometryFeatureKind::Box]),
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        GeometryFeatureError::UnsupportedOperation { .. }
    ));
}

struct EdgeInventoryProvider {
    edges: Vec<PersistentEntityId>,
}

impl GeometryFeatureProvider for EdgeInventoryProvider {
    type Geometry = String;
    type Error = &'static str;

    fn capabilities(&self) -> GeometryFeatureCapabilities {
        capabilities([GeometryFeatureKind::Fillet, GeometryFeatureKind::Chamfer])
    }

    fn entity_inventory(
        &self,
        _geometry: &Self::Geometry,
    ) -> Result<BTreeMap<GeometryRegionRole, Vec<PersistentEntityId>>, Self::Error> {
        Ok(BTreeMap::from([(
            GeometryRegionRole::Edge,
            self.edges.clone(),
        )]))
    }

    fn execute(
        &self,
        _feature_id: &str,
        _operation: &ResolvedGeometryFeatureOperation,
        _selection: &GeometryFeatureProviderSelection,
        _dependencies: &[(&str, &Self::Geometry)],
        _source: Option<&Self::Geometry>,
    ) -> Result<GeometryFeatureProviderOutput<Self::Geometry>, Self::Error> {
        Err("selection-only provider")
    }
}

fn edge(name: &str) -> PersistentEntityId {
    PersistentEntityId::new(
        runmat_geometry_core::PersistentEntityKind::Edge,
        name.into(),
        vec!["root".into()],
    )
}

#[test]
fn edge_treatments_resolve_exact_target_membership_and_cardinality() {
    let first = edge("edge-a");
    let second = edge("edge-b");
    let provider = EdgeInventoryProvider {
        edges: vec![first.clone(), second],
    };
    let geometry = "stock".to_string();
    let selector = runmat_geometry_core::GeometryFeatureRegionSelection {
        role: GeometryRegionRole::Edge,
        selector: runmat_geometry_core::GeometryRegionSelector::PersistentId(
            first.analysis_selection_id().unwrap(),
        ),
        cardinality: runmat_geometry_core::GeometryRegionCardinality::ExactlyOne,
    };
    let operation = ResolvedGeometryFeatureOperation::Fillet {
        target: GeometryFeatureRef("stock".into()),
        edges: selector,
        radius_m: runmat_types::physical::Length::new(0.1),
    };
    assert_eq!(
        resolve_provider_selection(
            &provider,
            "rounded",
            &operation,
            &[("stock", &geometry)],
            &BTreeMap::new(),
        )
        .unwrap(),
        GeometryFeatureProviderSelection::Edges(vec![first])
    );

    let overbroad = ResolvedGeometryFeatureOperation::Chamfer {
        target: GeometryFeatureRef("stock".into()),
        edges: runmat_geometry_core::GeometryFeatureRegionSelection {
            role: GeometryRegionRole::Edge,
            selector: runmat_geometry_core::GeometryRegionSelector::Result,
            cardinality: runmat_geometry_core::GeometryRegionCardinality::ExactlyOne,
        },
        distance_m: runmat_types::physical::Length::new(0.1),
    };
    assert!(matches!(
        resolve_provider_selection(
            &provider,
            "beveled",
            &overbroad,
            &[("stock", &geometry)],
            &BTreeMap::new(),
        ),
        Err(GeometryFeatureError::InvalidPlan(_))
    ));
}

struct RecordingProvider;

impl GeometryFeatureProvider for RecordingProvider {
    type Geometry = String;
    type Error = &'static str;

    fn capabilities(&self) -> GeometryFeatureCapabilities {
        capabilities([GeometryFeatureKind::Box, GeometryFeatureKind::Transform])
    }

    fn entity_inventory(
        &self,
        _geometry: &Self::Geometry,
    ) -> Result<BTreeMap<GeometryRegionRole, Vec<PersistentEntityId>>, Self::Error> {
        Ok(BTreeMap::new())
    }

    fn execute(
        &self,
        feature_id: &str,
        _operation: &ResolvedGeometryFeatureOperation,
        _selection: &super::GeometryFeatureProviderSelection,
        dependencies: &[(&str, &Self::Geometry)],
        _source: Option<&Self::Geometry>,
    ) -> Result<GeometryFeatureProviderOutput<Self::Geometry>, Self::Error> {
        Ok(GeometryFeatureProviderOutput {
            geometry: format!(
                "{feature_id}({})",
                dependencies
                    .iter()
                    .map(|(_, value)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            history: GeometryTopologyHistory {
                inventory_complete: false,
                result_entities: BTreeMap::new(),
                introduced: Vec::new(),
                source_lineage: Vec::new(),
            },
        })
    }
}

struct RecordingHost;

impl GeometryMaterializationHost<String> for RecordingHost {
    type Error = &'static str;

    fn import_exact(
        &self,
        source: &runmat_geometry_core::ImportedGeometryDefinition,
    ) -> Result<String, Self::Error> {
        Ok(format!("import({})", source.logical_path))
    }

    fn generate(
        &self,
        _request: &ResolvedGeneratedGeometryRequest,
    ) -> Result<runmat_geometry_core::GeneratedGeometryProduct, Self::Error> {
        Ok(runmat_geometry_core::GeneratedGeometryProduct {
            graph: GeometryFeatureGraph {
                features: vec![GeometryFeature {
                    feature_id: "generated".into(),
                    operation: GeometryFeatureOperation::Box {
                        width_m: scalar(1.0),
                        depth_m: scalar(1.0),
                        height_m: scalar(1.0),
                    },
                }],
                result: GeometryFeatureRef("generated".into()),
            },
            region_outputs: BTreeMap::new(),
        })
    }
}

#[test]
fn execution_uses_only_planned_dependencies_and_returns_the_declared_result() {
    let parameters = resolve_geometry_parameters(&[], &BTreeMap::new()).unwrap();
    let graph = GeometryFeatureGraph {
        features: vec![
            GeometryFeature {
                feature_id: "stock".into(),
                operation: GeometryFeatureOperation::Box {
                    width_m: scalar(1.0),
                    depth_m: scalar(2.0),
                    height_m: scalar(3.0),
                },
            },
            GeometryFeature {
                feature_id: "moved".into(),
                operation: GeometryFeatureOperation::Transform {
                    source: GeometryFeatureRef("stock".into()),
                    matrix: (0..16)
                        .map(|index| {
                            scalar(if matches!(index, 0 | 5 | 10 | 15) {
                                1.0
                            } else {
                                0.0
                            })
                        })
                        .collect(),
                },
            },
        ],
        result: GeometryFeatureRef("moved".into()),
    };
    let plan = plan_feature_graph(
        &graph,
        &parameters,
        &RecordingProvider.capabilities(),
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    let executed = execute_feature_plan(&RecordingProvider, &plan, None).unwrap();
    assert_eq!(executed.geometry, "moved(stock())");
    assert_eq!(executed.history_by_feature.len(), 2);
}

#[test]
fn cyclic_parameter_expressions_fail_closed() {
    let definitions = vec![
        GeometryParameterDefinition {
            parameter_id: "a".into(),
            parameter_type: GeometryParameterType::Scalar,
            required: true,
            default: None,
        },
        GeometryParameterDefinition {
            parameter_id: "b".into(),
            parameter_type: GeometryParameterType::Scalar,
            required: true,
            default: None,
        },
    ];
    let bindings = BTreeMap::from([
        ("a".into(), GeometryAuthoringValue::Scalar(parameter("b"))),
        ("b".into(), GeometryAuthoringValue::Scalar(parameter("a"))),
    ]);
    assert!(matches!(
        resolve_geometry_parameters(&definitions, &bindings),
        Err(GeometryFeatureError::ParameterCycle(_))
    ));
}

// witness: FEA-AUTHORING-GEOMETRY-002/geometry_source_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: A derived geometry requires the exact declared source revision; the closed-form check
//   that imported, constructed and generated forms are mutually exclusive is not asserted here.
#[test]
fn derived_geometry_requires_the_exact_declared_source() {
    let source_ref = GeometryDefinitionRef {
        module_path: "parts.fea".into(),
        geometry_id: "stock".into(),
    };
    let definition = GeometryDefinition {
        module_path: "parts.fea".into(),
        geometry_id: "moved".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Derived(DerivedGeometryDefinition {
            source: source_ref.clone(),
            graph: GeometryFeatureGraph {
                features: vec![GeometryFeature {
                    feature_id: "result".into(),
                    operation: GeometryFeatureOperation::Transform {
                        source: GeometryFeatureRef("source.result".into()),
                        matrix: (0..16)
                            .map(|index| {
                                scalar(if matches!(index, 0 | 5 | 10 | 15) {
                                    1.0
                                } else {
                                    0.0
                                })
                            })
                            .collect(),
                    },
                }],
                result: GeometryFeatureRef("result".into()),
            },
        }),
        regions: Vec::new(),
        cad_parameters: Vec::new(),
    };
    let source = "source".to_string();
    let materialized = materialize_geometry_definition(
        &definition,
        &BTreeMap::new(),
        Some(GeometryDependency {
            reference: &source_ref,
            geometry: &source,
        }),
        &RecordingProvider,
        &RecordingHost,
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    assert_eq!(materialized.kind, GeometryMaterializationKind::Derived);
    assert_eq!(materialized.geometry, "result(source)");
}

// witness: FEA-AUTHORING-GEOMETRY-002/generated_geometry @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Generated geometry receives resolved arguments at the provider boundary;
//   materialization of the function result is covered by the OCCT tests.
#[test]
fn generated_geometry_receives_resolved_arguments() {
    let definition = GeometryDefinition {
        module_path: "main.fea".into(),
        geometry_id: "generated".into(),
        parameters: vec![GeometryParameterDefinition {
            parameter_id: "length".into(),
            parameter_type: GeometryParameterType::Length,
            required: true,
            default: None,
        }],
        source: GeometryDefinitionSource::Generated(GeneratedGeometryDefinition {
            module_path: "geometry.m".into(),
            function: "make_part".into(),
            source_digest: GeometryDigest::from_bytes([7; 32]),
            arguments: vec![runmat_geometry_core::GeneratedGeometryArgument {
                parameter: "length".into(),
                value: GeometryAuthoringValue::Scalar(parameter("length")),
            }],
        }),
        regions: Vec::new(),
        cad_parameters: Vec::new(),
    };
    let materialized = materialize_geometry_definition(
        &definition,
        &BTreeMap::from([(
            "length".into(),
            GeometryAuthoringValue::Scalar(scalar(0.25)),
        )]),
        None,
        &RecordingProvider,
        &RecordingHost,
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    assert_eq!(materialized.kind, GeometryMaterializationKind::Generated);
    assert_eq!(materialized.geometry, "generated()");
}

#[test]
fn source_geometry_expression_retains_units_through_parameter_arithmetic() {
    use runmat_types::physical::{parse_unit_expression, DynamicDimension, QuantityKind};
    let source = GeometryScalarExpression::Quantity {
        value: Box::new(scalar(25.0)),
        unit: parse_unit_expression("mm").unwrap(),
        source_unit_symbol: "mm".into(),
        semantic_kind: None,
    };
    let definitions = [GeometryParameterDefinition {
        parameter_id: "width".into(),
        parameter_type: GeometryParameterType::Length,
        required: false,
        default: Some(GeometryAuthoringValue::Scalar(source)),
    }];
    let parameters = resolve_geometry_parameters(&definitions, &BTreeMap::new()).unwrap();
    let resolved = parameters
        .resolve_value(&GeometryAuthoringValue::Parameter("width".into()))
        .unwrap();
    let GeometryAuthoringValue::Scalar(GeometryScalarExpression::Canonical { value }) = resolved
    else {
        panic!("descriptor was dropped")
    };
    assert!(value
        .descriptor()
        .is_compatible_with(DynamicDimension::quantity_type(QuantityKind::Length)));
    assert_eq!(value.canonical_value(), 0.025);
    let product = GeometryScalarExpression::Binary {
        operator: GeometryScalarOperator::Multiply,
        left: Box::new(parameter("width")),
        right: Box::new(scalar(2.0)),
    };
    assert_eq!(parameters.length(&product).unwrap(), 0.05);
    assert!(parameters.angle(&product).is_err());
    assert!(parameters.scalar(&product).is_err());
}

#[test]
fn angle_cannot_enter_a_direction_or_length_parameter() {
    use runmat_types::physical::parse_unit_expression;
    let angle = GeometryScalarExpression::Quantity {
        value: Box::new(scalar(90.0)),
        unit: parse_unit_expression("deg").unwrap(),
        source_unit_symbol: "deg".into(),
        semantic_kind: None,
    };
    let definitions = [GeometryParameterDefinition {
        parameter_id: "width".into(),
        parameter_type: GeometryParameterType::Length,
        required: false,
        default: Some(GeometryAuthoringValue::Scalar(angle.clone())),
    }];
    assert!(resolve_geometry_parameters(&definitions, &BTreeMap::new()).is_err());
    let parameters = resolve_geometry_parameters(&[], &BTreeMap::new()).unwrap();
    assert!(parameters
        .vector(&[angle.clone(), scalar(0.0), scalar(0.0)])
        .is_err());
    assert!((parameters.angle(&angle).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
}
