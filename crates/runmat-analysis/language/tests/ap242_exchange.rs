pub mod support;

use runmat_analysis_language::*;
use runmat_geometry_core::{
    Ap242Association, Ap242AssociationTarget, Ap242Capability, Ap242CapabilityDisposition,
    Ap242CapabilityRecord, Ap242EngineeringProperty, Ap242EngineeringPropertyKind,
    Ap242EnvironmentCondition, Ap242ExchangeInventory, Ap242MaterialCandidate, Ap242MaterialClass,
    Ap242Measure, Ap242Parameter, Ap242ParameterAccess, QuantityKind,
};
use support::{source_module, TestHost};

// witness: FEA-AUTHORING-GEOMETRY-002/ap242_exchange @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Imported CAD parameters resolve to exact typed source facts; other AP242 content
//   (constraints, associations) is covered by the profile tests.
#[test]
fn imported_cad_parameters_resolve_exact_typed_source_facts() {
    let source = r#"
geometry part {
    file = "./geometry/part.step"
    units = mm
    cad_parameter wall : length {
        select persistent_id "parameter-wall"
        require readable
    }
}
geometry copy {
    construct {
        feature body = box(
            width = geometry.part.parameter.wall,
            depth = 1 m,
            height = 1 m,
        )
        result = feature.body
    }
}
"#;
    let host = TestHost::default().with_geometry_inspection(
        "project/geometry/part.step",
        parameterized_geometry_inspection(),
    );
    let program = check(source, &host);

    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let imported = program
        .geometries
        .values()
        .find(|geometry| geometry.geometry_id == "part")
        .unwrap();
    let parameter = &imported.cad_parameters[0];
    assert_eq!(parameter.source.parameter_id, "parameter-wall");
    assert_eq!(parameter.source.value.canonical_value(), 0.008);
    assert_eq!(parameter.source.value.source_symbol(), "mm");
    let copy = program
        .geometries
        .values()
        .find(|geometry| geometry.geometry_id == "copy")
        .unwrap();
    let runmat_geometry_core::GeometryDefinitionSource::Constructed(graph) = &copy.source else {
        panic!("expected constructed geometry");
    };
    let runmat_geometry_core::GeometryFeatureOperation::Box { width_m, .. } =
        &graph.features[0].operation
    else {
        panic!("expected box feature");
    };
    assert_eq!(
        width_m,
        &runmat_geometry_core::GeometryScalarExpression::SourceMeasure {
            value: parameter.source.value.clone(),
        }
    );
}

// witness: FEA-AUTHORING-GEOMETRY-002/cad_parameter_admission @
//   shared_infrastructure/mesh_independent/source_check;
//   boundary: Fixture host parameter metadata: wrong dimensions/unqualified write rejection only;
//   no CAD kernel regeneration.
#[test]
fn imported_cad_parameters_reject_wrong_dimensions_and_unqualified_writes() {
    for (physical_type, access, expected) in [
        ("mass", "readable", "source has dimension"),
        ("length", "writable", "read-only"),
    ] {
        let source = format!(
            r#"
geometry part {{
    file = "./geometry/part.step"
    units = mm
    cad_parameter wall : {physical_type} {{
        select name "WallThickness"
        require {access}
    }}
}}
"#
        );
        let host = TestHost::default().with_geometry_inspection(
            "project/geometry/part.step",
            parameterized_geometry_inspection(),
        );
        let program = check(&source, &host);

        assert!(!program.is_valid());
        assert!(program
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains(expected)));
    }
}

// witness: FEA-AUTHORING-GEOMETRY-002/qualified_parameter_regeneration @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Source regeneration fails at check without a qualified write capability; the
//   qualified success path is not exercised.
#[test]
fn source_regeneration_fails_during_checking_without_qualified_write_capability() {
    let source = r#"
geometry part {
    file = "./geometry/part.step"
    units = mm
    cad_parameter wall : length {
        select persistent_id "parameter-wall"
        require readable
    }
}
geometry changed from geometry.part {
    input wall : length = 10 mm
    derive {
        set source.parameter.wall = input.wall
        result = regenerate(source)
    }
}
"#;
    let host = TestHost::default().with_geometry_inspection(
        "project/geometry/part.step",
        parameterized_geometry_inspection(),
    );
    let program = check(source, &host);

    assert!(!program.is_valid());
    assert!(
        program.diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message
                .contains("qualified regeneration capability")
        }),
        "{:#?}",
        program.diagnostics
    );
    assert_eq!(program.geometries.len(), 1);
}

#[test]
fn source_regeneration_requires_the_exact_cad_parameter_physical_type() {
    let source = r#"
geometry part {
    file = "./geometry/part.step"
    units = mm
    cad_parameter wall : length {
        select persistent_id "parameter-wall"
        require writable
    }
}
geometry changed from geometry.part {
    derive {
        set source.parameter.wall = 10 kg
        result = regenerate(source)
    }
}
"#;
    let host = TestHost::default().with_geometry_inspection(
        "project/geometry/part.step",
        regenerable_geometry_inspection(),
    );
    let program = check(source, &host);

    assert!(!program.is_valid());
    assert!(
        program.diagnostics.iter().any(|diagnostic| {
            diagnostic.message.contains("requires a length value")
                && diagnostic
                    .message
                    .contains("physically incompatible quantity")
        }),
        "{:#?}",
        program.diagnostics
    );
    assert_eq!(program.geometries.len(), 1);
}

// witness: FEA-AUTHORING-GEOMETRY-002/explicit_material_adoption @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: CAD material adoption is explicit, typed and provenance-bearing; mapping to every
//   constitutive law is not covered.
// witness: FEA-AUTHORING-GEOMETRY-002/adopted_property_provenance @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Adopted properties retain provenance; entity ids, document references and uncertainty
//   are not all asserted.
#[test]
fn cad_material_adoption_is_explicit_typed_and_provenance_bearing() {
    let source = r#"
geometry part {
    file = "./geometry/part.step"
    units = mm
}
material cad_steel from geometry.part.cad_material("material-steel") {
    use mass_density
    use elastic.youngs_modulus
    use elastic.poisson_ratio
    on_conflict = reject
}
"#;
    let host = TestHost::default().with_geometry_inspection(
        "project/geometry/part.step",
        parameterized_geometry_inspection(),
    );
    let program = check(source, &host);

    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let material = program.materials.values().next().unwrap();
    assert_eq!(
        material
            .mass_density_kg_per_m3
            .map(|quantity| quantity.canonical_value()),
        Some(7850.0)
    );
    assert_eq!(
        material
            .require_mechanical()
            .unwrap()
            .youngs_modulus_pa
            .canonical_value(),
        200e9
    );
    let runmat_analysis_core::MaterialProvenance::ImportedCad(provenance) = &material.provenance
    else {
        panic!("expected imported CAD provenance");
    };
    assert_eq!(provenance.source_material_id, "material-steel");
    assert_eq!(provenance.adopted_properties.len(), 3);
    assert_eq!(provenance.associations.len(), 1);
}

#[test]
fn cad_material_environment_requirements_use_shared_typed_quantities() {
    for (requirement, valid, expected) in [
        ("20 degC", true, ""),
        ("20 mm", false, "requires a temperature quantity"),
        (
            "25 degC",
            false,
            "do not carry the required operating temperature",
        ),
    ] {
        let source = format!(
            r#"
geometry part {{
    file = "./geometry/part.step"
    units = mm
}}
material cad_steel from geometry.part.cad_material("material-steel") {{
    use mass_density
    use elastic.youngs_modulus
    use elastic.poisson_ratio
    require operating_temperature = {requirement}
    on_conflict = reject
}}
"#
        );
        let host = TestHost::default().with_geometry_inspection(
            "project/geometry/part.step",
            conditioned_geometry_inspection(),
        );
        let program = check(&source, &host);

        assert_eq!(program.is_valid(), valid, "{:#?}", program.diagnostics);
        if !valid {
            assert!(program
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains(expected)));
        }
    }
}

fn check(source: &str, host: &TestHost) -> CheckedFeaProgram {
    FeaCompiler::new(host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

fn parameterized_geometry_inspection() -> GeometryInspection {
    GeometryInspection {
        preview_extent_m: None,
        unit_evidence: Default::default(),
        shared_faces: Default::default(),
        content_digest: SourceDigest([8; 32]),
        topology_digest: SourceDigest([9; 32]),
        capabilities: vec![
            GeometryInspectionCapability::Topology,
            GeometryInspectionCapability::PersistentIds,
            GeometryInspectionCapability::Parameters,
            GeometryInspectionCapability::EngineeringProperties,
            GeometryInspectionCapability::MaterialCandidates,
        ],
        resolved_units: None,
        authored_names: Default::default(),
        exchange: Some(Ap242ExchangeInventory {
            schemas: vec!["AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF".to_string()],
            capabilities: vec![
                Ap242CapabilityRecord {
                    capability: Ap242Capability::Parameters,
                    disposition: Ap242CapabilityDisposition::ReadOnly,
                    evidence_entity_ids: vec![12],
                    detail: "STEP exchange shape parameters inventory".to_string(),
                },
                Ap242CapabilityRecord {
                    capability: Ap242Capability::Materials,
                    disposition: Ap242CapabilityDisposition::ReadOnly,
                    evidence_entity_ids: vec![20],
                    detail: "STEP exchange physical materials inventory".to_string(),
                },
                Ap242CapabilityRecord {
                    capability: Ap242Capability::EngineeringProperties,
                    disposition: Ap242CapabilityDisposition::ReadOnly,
                    evidence_entity_ids: vec![37, 47, 57],
                    detail: "STEP exchange engineering properties inventory".to_string(),
                },
            ],
            parameters: vec![Ap242Parameter {
                parameter_id: "parameter-wall".to_string(),
                name: "WallThickness".to_string(),
                source_entity_id: 11,
                representation_entity_id: 12,
                owner_entity_ids: vec![14],
                persistent_id: Some("parameter-wall".to_string()),
                value: Ap242Measure::try_new(
                    QuantityKind::Length,
                    8.0,
                    *runmat_geometry_core::LengthUnit::Millimeter.definition(),
                    "mm",
                    runmat_geometry_core::physical::UnitAttribution::FileDeclaration {
                        context: "fixture parameter #11".into(),
                    },
                )
                .unwrap(),
                minimum: None,
                maximum: None,
                access: Ap242ParameterAccess::ReadOnly,
            }],
            engineering_properties: vec![
                material_property(
                    "density",
                    Ap242EngineeringPropertyKind::MassDensity,
                    QuantityKind::MassDensity,
                    7850.0,
                    37,
                ),
                material_property(
                    "youngs",
                    Ap242EngineeringPropertyKind::YoungsModulus,
                    QuantityKind::Pressure,
                    200e9,
                    47,
                ),
                material_property(
                    "poisson",
                    Ap242EngineeringPropertyKind::PoissonRatio,
                    QuantityKind::Dimensionless,
                    0.3,
                    57,
                ),
            ],
            materials: vec![Ap242MaterialCandidate {
                material_id: "material-steel".to_string(),
                name: "Structural steel".to_string(),
                source_entity_id: 20,
                persistent_id: Some("material-steel".to_string()),
                class: Ap242MaterialClass::Physical,
                property_ids: vec![
                    "density".to_string(),
                    "youngs".to_string(),
                    "poisson".to_string(),
                ],
                specification: Some("EN 10025".to_string()),
            }],
            associations: vec![Ap242Association {
                association_id: "steel-on-part".to_string(),
                source_entity_id: 20,
                material_id: "material-steel".to_string(),
                target: Ap242AssociationTarget::ProductDefinitionEntity(2),
            }],
            ..Ap242ExchangeInventory::default()
        }),
    }
}

fn regenerable_geometry_inspection() -> GeometryInspection {
    let mut inspection = parameterized_geometry_inspection();
    inspection
        .capabilities
        .push(GeometryInspectionCapability::Regeneration);
    let exchange = inspection.exchange.as_mut().unwrap();
    exchange.capabilities.push(Ap242CapabilityRecord {
        capability: Ap242Capability::Regeneration,
        disposition: Ap242CapabilityDisposition::Regenerable,
        evidence_entity_ids: vec![12],
        detail: "qualified source regeneration".to_string(),
    });
    exchange.parameters[0].access = Ap242ParameterAccess::Regenerable;
    inspection
}

fn conditioned_geometry_inspection() -> GeometryInspection {
    let mut inspection = parameterized_geometry_inspection();
    for property in &mut inspection.exchange.as_mut().unwrap().engineering_properties {
        property.environment.push(Ap242EnvironmentCondition {
            name: "temperature".to_string(),
            value: Ap242Measure::try_new(
                QuantityKind::Temperature,
                293.15,
                runmat_geometry_core::physical::parse_unit_expression("K").unwrap(),
                "K",
                runmat_geometry_core::physical::UnitAttribution::FileDeclaration {
                    context: "fixture temperature".into(),
                },
            )
            .unwrap(),
        });
    }
    inspection
}

fn material_property(
    id: &str,
    kind: Ap242EngineeringPropertyKind,
    dimension: QuantityKind,
    value: f64,
    entity: u64,
) -> Ap242EngineeringProperty {
    Ap242EngineeringProperty {
        property_id: id.to_string(),
        kind,
        name: id.to_string(),
        source_entity_id: entity,
        representation_entity_id: entity + 1,
        value: Ap242Measure::from_canonical(
            dimension,
            value,
            runmat_geometry_core::physical::UnitAttribution::FileDeclaration {
                context: format!("fixture property #{entity}"),
            },
        )
        .unwrap(),
        specification: None,
        environment: Vec::new(),
    }
}
