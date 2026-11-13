use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactMassPropertiesImplementation, GeometryDefinition, GeometryDefinitionSource,
    GeometryFeature, GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef,
    GeometryRegionCardinality, GeometryRegionDefinition, GeometryRegionRole,
    GeometryRegionSelector, PersistentEntityId,
};
use runmat_geometry_ops::{
    GeometryFeaturePlanLimits, GeometryFeatureProvider, GeometryMaterializationHost,
    GeometryMaterializationKind, GeometryRegionEvidence, ResolvedGeneratedGeometryRequest,
    ResolvedGeometryFeatureOperation,
};

use super::feature::{OcctExactFeatureError, OcctExactFeatureProvider};
use crate::{exact::ExactCadImportOptions, import::GeometryImportContext, ImportedExactCad};

#[test]
fn exact_primitives_use_the_canonical_import_path() {
    let provider = provider();
    let stock = execute(
        &provider,
        "stock",
        ResolvedGeometryFeatureOperation::Box {
            width_m: runmat_types::physical::Length::new(2.0),
            depth_m: runmat_types::physical::Length::new(3.0),
            height_m: runmat_types::physical::Length::new(4.0),
        },
        &[],
    );
    assert_eq!(stock.topology.faces.len(), 6);
    assert_close(volume(&stock), 24.0);

    let cylinder = execute(
        &provider,
        "cylinder",
        ResolvedGeometryFeatureOperation::Cylinder {
            radius_m: runmat_types::physical::Length::new(0.25),
            height_m: runmat_types::physical::Length::new(6.0),
            axis: [0.0, 0.0, 1.0],
            origin_m:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                    [1.0, 1.5, -1.0],
                ),
        },
        &[],
    );
    assert_close(
        volume(&cylinder),
        std::f64::consts::PI * 0.25_f64.powi(2) * 6.0,
    );

    let sphere = execute(
        &provider,
        "sphere",
        ResolvedGeometryFeatureOperation::Sphere {
            radius_m: runmat_types::physical::Length::new(0.5),
            center_m:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                    [2.0, -1.0, 0.25],
                ),
        },
        &[],
    );
    assert_close(
        volume(&sphere),
        4.0 * std::f64::consts::PI * 0.5_f64.powi(3) / 3.0,
    );
    stock.build_closure().unwrap();
    cylinder.build_closure().unwrap();
    sphere.build_closure().unwrap();
}

#[test]
fn exact_transform_and_instance_are_qualified_explicitly() {
    let provider = provider();
    let source = unit_box(&provider, "source");
    let moved = execute(
        &provider,
        "moved",
        ResolvedGeometryFeatureOperation::Transform {
            source: GeometryFeatureRef("source".into()),
            matrix: translation(3.0, -2.0, 1.0),
        },
        &[("source", &source)],
    );
    assert_close(volume(&moved), 1.0);
    assert_eq!(moved.analysis_options().revision.revision, 2);

    let instances = execute(
        &provider,
        "instances",
        ResolvedGeometryFeatureOperation::Instance {
            source: GeometryFeatureRef("source".into()),
            transforms: vec![translation(0.0, 0.0, 0.0), translation(2.0, 0.0, 0.0)],
        },
        &[("source", &source)],
    );
    assert_close(volume(&instances), 2.0);
    assert_eq!(instances.topology.solids.len(), 2);
    instances.build_closure().unwrap();
}

#[test]
fn exact_boolean_profile_is_qualified_operation_by_operation() {
    let provider = provider();
    let left = unit_box(&provider, "left");
    let right_source = unit_box(&provider, "right_source");
    let right = execute(
        &provider,
        "right",
        ResolvedGeometryFeatureOperation::Transform {
            source: GeometryFeatureRef("right_source".into()),
            matrix: translation(0.5, 0.0, 0.0),
        },
        &[("right_source", &right_source)],
    );
    let operands = [("left", &left), ("right", &right)];

    let union = execute(
        &provider,
        "union",
        ResolvedGeometryFeatureOperation::Union {
            operands: vec![
                GeometryFeatureRef("left".into()),
                GeometryFeatureRef("right".into()),
            ],
        },
        &operands,
    );
    assert_close(volume(&union), 1.5);

    let intersection = execute(
        &provider,
        "intersection",
        ResolvedGeometryFeatureOperation::Intersect {
            operands: vec![
                GeometryFeatureRef("left".into()),
                GeometryFeatureRef("right".into()),
            ],
        },
        &operands,
    );
    assert_close(volume(&intersection), 0.5);

    let subtraction = execute(
        &provider,
        "subtraction",
        ResolvedGeometryFeatureOperation::Subtract {
            base: GeometryFeatureRef("left".into()),
            tools: vec![GeometryFeatureRef("right".into())],
        },
        &operands,
    );
    assert_close(volume(&subtraction), 0.5);
    assert_eq!(subtraction.analysis_options().revision.revision, 3);
    assert!(subtraction
        .analysis_options()
        .revision
        .parent_document_digest
        .is_some());
    union.build_closure().unwrap();
    intersection.build_closure().unwrap();
    subtraction.build_closure().unwrap();
}

// witness: FEA-AUTHORING-GEOMETRY-002/exact_geometry_closure @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Constructed materialization and result regions share one exact artifact; derived and
//   generated closure are covered by sibling tests.
#[test]
fn constructed_materialization_and_result_regions_share_the_exact_artifact() {
    let provider = provider();
    let mut definition = box_definition();
    let materialized = runmat_geometry_ops::materialize_geometry_definition(
        &definition,
        &BTreeMap::new(),
        None,
        &provider,
        &RejectingSourceHost,
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    assert_eq!(materialized.kind, GeometryMaterializationKind::Constructed);
    assert_close(volume(&materialized.geometry), 24.0);
    let regions = runmat_geometry_ops::resolve_geometry_regions(
        &definition,
        &materialized.geometry.topology,
        &materialized.feature_history,
        &NoRegionEvidence,
    )
    .unwrap();
    assert_eq!(regions["body"].entity_ids.len(), 1);

    definition.regions[0].selector =
        GeometryRegionSelector::BoundaryIntroducedBy(GeometryFeatureRef("stock".into()));
    let introduced = runmat_geometry_ops::resolve_geometry_regions(
        &definition,
        &materialized.geometry.topology,
        &materialized.feature_history,
        &NoRegionEvidence,
    )
    .unwrap();
    assert_eq!(introduced["body"].entity_ids.len(), 1);

    let mut incomplete = materialized.feature_history.clone();
    incomplete.get_mut("stock").unwrap().inventory_complete = false;
    assert!(runmat_geometry_ops::resolve_geometry_regions(
        &definition,
        &materialized.geometry.topology,
        &incomplete,
        &NoRegionEvidence,
    )
    .is_err());
}

// witness: FEA-AUTHORING-GEOMETRY-002/feature_lineage_inventory @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Boolean history projects authored regions onto the final exact topology.
#[test]
fn boolean_history_projects_authored_regions_to_the_final_exact_topology() {
    let provider = provider();
    let definition = bored_block_definition();
    let materialized = runmat_geometry_ops::materialize_geometry_definition(
        &definition,
        &BTreeMap::new(),
        None,
        &provider,
        &RejectingSourceHost,
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    let regions = runmat_geometry_ops::resolve_geometry_regions(
        &definition,
        &materialized.geometry.topology,
        &materialized.feature_history,
        &NoRegionEvidence,
    )
    .unwrap();

    assert_eq!(regions["body"].role, GeometryRegionRole::Cell);
    assert_eq!(regions["body"].entity_ids.len(), 1);
    assert_eq!(regions["hole_wall"].role, GeometryRegionRole::Face);
    assert_eq!(regions["hole_wall"].entity_ids.len(), 1);
    assert!(regions["hole_wall"]
        .entity_ids
        .iter()
        .all(|entity| materialized
            .geometry
            .topology
            .faces
            .iter()
            .any(|face| face.id == *entity)));

    let bore = &materialized.feature_history["bore"];
    assert!(bore.inventory_complete);
    assert_eq!(
        bore.result_entities[&GeometryRegionRole::Face],
        regions["hole_wall"].entity_ids
    );
    let stock = &materialized.feature_history["stock"];
    assert_eq!(stock.result_entities[&GeometryRegionRole::Cell].len(), 1);
    materialized.geometry.build_closure().unwrap();
}

#[test]
fn boundary_introduced_by_an_instance_follows_its_copies_through_a_subtraction() {
    let provider = provider();
    let definition = pinned_block_definition();
    let materialized = runmat_geometry_ops::materialize_geometry_definition(
        &definition,
        &BTreeMap::new(),
        None,
        &provider,
        &RejectingSourceHost,
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    let regions = runmat_geometry_ops::resolve_geometry_regions(
        &definition,
        &materialized.geometry.topology,
        &materialized.feature_history,
        &NoRegionEvidence,
    )
    .unwrap();
    // Two pins cut through the block leave two bore walls.
    assert_eq!(regions["pin_bores"].entity_ids.len(), 2);
    materialized.geometry.build_closure().unwrap();
}

// witness: FEA-AUTHORING-GEOMETRY-002/feature_lineage_inventory @
//   shared_infrastructure/mesh_independent/source_check; features: occt-native; coverage:
//   partial;
//   boundary: Feature lineage merges shared ancestors across boolean branches; the full
//   per-operation entity inventory and canonical order are not asserted.
#[test]
fn feature_lineage_merges_shared_ancestors_across_boolean_branches() {
    let provider = provider();
    let definition = branched_union_definition();
    let materialized = runmat_geometry_ops::materialize_geometry_definition(
        &definition,
        &BTreeMap::new(),
        None,
        &provider,
        &RejectingSourceHost,
        GeometryFeaturePlanLimits::default(),
    )
    .unwrap();
    let regions = runmat_geometry_ops::resolve_geometry_regions(
        &definition,
        &materialized.geometry.topology,
        &materialized.feature_history,
        &NoRegionEvidence,
    )
    .unwrap();

    assert_eq!(regions["stock_descendants"].role, GeometryRegionRole::Solid);
    assert_eq!(regions["stock_descendants"].entity_ids.len(), 2);
    assert_eq!(
        materialized.feature_history["stock"].result_entities[&GeometryRegionRole::Solid],
        regions["stock_descendants"].entity_ids
    );
}

#[test]
fn unqualified_source_regeneration_fails_even_when_the_provider_is_called_directly() {
    let provider = provider();
    let source = unit_box(&provider, "source");
    let error = match provider.execute(
        "regeneration",
        &ResolvedGeometryFeatureOperation::RegenerateSource {
            parameters: BTreeMap::new(),
        },
        &runmat_geometry_ops::GeometryFeatureProviderSelection::None,
        &[("source", &source)],
        None,
    ) {
        Ok(_) => panic!("unqualified extrusion unexpectedly executed"),
        Err(error) => error,
    };
    assert!(matches!(error, OcctExactFeatureError::InvalidRequest(_)));

    let wrong_dependencies = match provider.execute(
        "moved",
        &ResolvedGeometryFeatureOperation::Transform {
            source: GeometryFeatureRef("source".into()),
            matrix: translation(1.0, 0.0, 0.0),
        },
        &runmat_geometry_ops::GeometryFeatureProviderSelection::None,
        &[("wrong", &source)],
        None,
    ) {
        Ok(_) => panic!("mismatched feature dependencies unexpectedly executed"),
        Err(error) => error,
    };
    assert!(matches!(
        wrong_dependencies,
        OcctExactFeatureError::InvalidRequest(_)
    ));
}

fn provider() -> OcctExactFeatureProvider {
    OcctExactFeatureProvider::new(
        ExactCadImportOptions::default(),
        GeometryImportContext::new(),
    )
}

fn unit_box(provider: &OcctExactFeatureProvider, id: &str) -> ImportedExactCad {
    execute(
        provider,
        id,
        ResolvedGeometryFeatureOperation::Box {
            width_m: runmat_types::physical::Length::new(1.0),
            depth_m: runmat_types::physical::Length::new(1.0),
            height_m: runmat_types::physical::Length::new(1.0),
        },
        &[],
    )
}

fn execute(
    provider: &OcctExactFeatureProvider,
    id: &str,
    operation: ResolvedGeometryFeatureOperation,
    dependencies: &[(&str, &ImportedExactCad)],
) -> ImportedExactCad {
    provider
        .execute(
            id,
            &operation,
            &runmat_geometry_ops::GeometryFeatureProviderSelection::None,
            dependencies,
            None,
        )
        .unwrap()
        .geometry
}

fn translation(x: f64, y: f64, z: f64) -> [f64; 16] {
    [
        1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn volume(geometry: &ImportedExactCad) -> f64 {
    geometry
        .evaluators
        .mass_properties
        .iter()
        .map(|record| match &record.implementation {
            ExactMassPropertiesImplementation::KernelValidated { properties, .. } => {
                properties.volume_m3.canonical_value()
            }
            ExactMassPropertiesImplementation::Kernel { .. } => 0.0,
        })
        .sum()
}

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-9 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:.16e} differs from expected {expected:.16e} by more than {tolerance:.16e}"
    );
}

fn box_definition() -> GeometryDefinition {
    GeometryDefinition {
        module_path: "main.fea".into(),
        geometry_id: "stock".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
            features: vec![GeometryFeature {
                feature_id: "stock".into(),
                operation: GeometryFeatureOperation::Box {
                    width_m: 2.0.into(),
                    depth_m: 3.0.into(),
                    height_m: 4.0.into(),
                },
            }],
            result: GeometryFeatureRef("stock".into()),
        }),
        regions: vec![GeometryRegionDefinition {
            region_id: "body".into(),
            role: GeometryRegionRole::Solid,
            selector: GeometryRegionSelector::Result,
            cardinality: GeometryRegionCardinality::ExactlyOne,
        }],
        cad_parameters: Vec::new(),
    }
}

fn bored_block_definition() -> GeometryDefinition {
    GeometryDefinition {
        module_path: "main.fea".into(),
        geometry_id: "bored_block".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
            features: vec![
                GeometryFeature {
                    feature_id: "stock".into(),
                    operation: GeometryFeatureOperation::Box {
                        width_m: 2.0.into(),
                        depth_m: 2.0.into(),
                        height_m: 2.0.into(),
                    },
                },
                GeometryFeature {
                    feature_id: "bore".into(),
                    operation: GeometryFeatureOperation::Cylinder {
                        radius_m: 0.25.into(),
                        height_m: 4.0.into(),
                        axis: [0.0.into(), 0.0.into(), 1.0.into()],
                        origin_m: [1.0.into(), 1.0.into(), (-1.0).into()],
                    },
                },
                GeometryFeature {
                    feature_id: "cut".into(),
                    operation: GeometryFeatureOperation::Subtract {
                        base: GeometryFeatureRef("stock".into()),
                        tools: vec![GeometryFeatureRef("bore".into())],
                    },
                },
            ],
            result: GeometryFeatureRef("cut".into()),
        }),
        regions: vec![
            GeometryRegionDefinition {
                region_id: "body".into(),
                role: GeometryRegionRole::Cell,
                selector: GeometryRegionSelector::FeatureResult(GeometryFeatureRef("stock".into())),
                cardinality: GeometryRegionCardinality::ExactlyOne,
            },
            GeometryRegionDefinition {
                region_id: "hole_wall".into(),
                role: GeometryRegionRole::Face,
                selector: GeometryRegionSelector::BoundaryIntroducedBy(GeometryFeatureRef(
                    "bore".into(),
                )),
                cardinality: GeometryRegionCardinality::ExactlyOne,
            },
        ],
        cad_parameters: Vec::new(),
    }
}

fn pinned_block_definition() -> GeometryDefinition {
    let translation = |x: f64| {
        [
            1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>()
    };
    GeometryDefinition {
        module_path: "main.fea".into(),
        geometry_id: "pinned_block".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
            features: vec![
                GeometryFeature {
                    feature_id: "stock".into(),
                    operation: GeometryFeatureOperation::Box {
                        width_m: 2.0.into(),
                        depth_m: 2.0.into(),
                        height_m: 2.0.into(),
                    },
                },
                GeometryFeature {
                    feature_id: "pin".into(),
                    operation: GeometryFeatureOperation::Cylinder {
                        radius_m: 0.2.into(),
                        height_m: 4.0.into(),
                        axis: [0.0.into(), 0.0.into(), 1.0.into()],
                        origin_m: [0.5.into(), 1.0.into(), (-1.0).into()],
                    },
                },
                GeometryFeature {
                    feature_id: "pins".into(),
                    operation: GeometryFeatureOperation::Instance {
                        source: GeometryFeatureRef("pin".into()),
                        transforms: vec![translation(0.0), translation(1.0)],
                    },
                },
                GeometryFeature {
                    feature_id: "cut".into(),
                    operation: GeometryFeatureOperation::Subtract {
                        base: GeometryFeatureRef("stock".into()),
                        tools: vec![GeometryFeatureRef("pins".into())],
                    },
                },
            ],
            result: GeometryFeatureRef("cut".into()),
        }),
        regions: vec![GeometryRegionDefinition {
            region_id: "pin_bores".into(),
            role: GeometryRegionRole::Face,
            selector: GeometryRegionSelector::BoundaryIntroducedBy(GeometryFeatureRef(
                "pins".into(),
            )),
            cardinality: GeometryRegionCardinality::OneOrMore,
        }],
        cad_parameters: Vec::new(),
    }
}

fn branched_union_definition() -> GeometryDefinition {
    GeometryDefinition {
        module_path: "main.fea".into(),
        geometry_id: "branched_union".into(),
        parameters: Vec::new(),
        source: GeometryDefinitionSource::Constructed(GeometryFeatureGraph {
            features: vec![
                GeometryFeature {
                    feature_id: "stock".into(),
                    operation: GeometryFeatureOperation::Box {
                        width_m: 1.0.into(),
                        depth_m: 1.0.into(),
                        height_m: 1.0.into(),
                    },
                },
                GeometryFeature {
                    feature_id: "moved".into(),
                    operation: GeometryFeatureOperation::Transform {
                        source: GeometryFeatureRef("stock".into()),
                        matrix: translation(2.0, 0.0, 0.0)
                            .into_iter()
                            .map(Into::into)
                            .collect(),
                    },
                },
                GeometryFeature {
                    feature_id: "result".into(),
                    operation: GeometryFeatureOperation::Union {
                        operands: vec![
                            GeometryFeatureRef("stock".into()),
                            GeometryFeatureRef("moved".into()),
                        ],
                    },
                },
            ],
            result: GeometryFeatureRef("result".into()),
        }),
        regions: vec![GeometryRegionDefinition {
            region_id: "stock_descendants".into(),
            role: GeometryRegionRole::Solid,
            selector: GeometryRegionSelector::FeatureResult(GeometryFeatureRef("stock".into())),
            cardinality: GeometryRegionCardinality::OneOrMore,
        }],
        cad_parameters: Vec::new(),
    }
}

struct RejectingSourceHost;

impl GeometryMaterializationHost<ImportedExactCad> for RejectingSourceHost {
    type Error = &'static str;

    fn import_exact(
        &self,
        _source: &runmat_geometry_core::ImportedGeometryDefinition,
    ) -> Result<ImportedExactCad, Self::Error> {
        Err("constructed geometry must not import")
    }

    fn generate(
        &self,
        _request: &ResolvedGeneratedGeometryRequest,
    ) -> Result<runmat_geometry_core::GeneratedGeometryProduct, Self::Error> {
        Err("constructed geometry must not invoke a generator")
    }
}

struct NoRegionEvidence;

impl GeometryRegionEvidence for NoRegionEvidence {
    type Error = std::convert::Infallible;

    fn by_name(
        &self,
        _role: GeometryRegionRole,
        _name: &str,
    ) -> Result<Vec<PersistentEntityId>, Self::Error> {
        Ok(Vec::new())
    }

    fn by_tag(
        &self,
        _role: GeometryRegionRole,
        _tag: &str,
    ) -> Result<Vec<PersistentEntityId>, Self::Error> {
        Ok(Vec::new())
    }

    fn generated_output(
        &self,
        _role: GeometryRegionRole,
        _output: &str,
    ) -> Result<Vec<PersistentEntityId>, Self::Error> {
        Ok(Vec::new())
    }
}
