mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

#[test]
fn compiler_checks_inputs_defaults_locals_and_outputs_as_one_program() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {
}
study load_study {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    input load : force {
        default = 1 kN
        minimum = 0 N
        maximum = 5 kN
    }
    let doubled = 2 * input.load
    analysis solve {
        structural static {
        }
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    let host = TestHost::default().with_module(
        "project/library.fea",
        "material library_material {\nmass_density = 1 kg/m^3\n}\n",
        2,
    );
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });

    assert!(
        program.is_valid(),
        "graph={:#?}\nbinding={:#?}\nsemantic={:#?}",
        program.bound.graph.diagnostics,
        program.bound.diagnostics,
        program.diagnostics
    );
    let declaration = program
        .declarations
        .iter()
        .find(|((_, key), _)| key.kind == FeaSymbolKind::Study)
        .map(|(_, declaration)| declaration)
        .unwrap();
    let input = &declaration.inputs["load"];
    assert_eq!(
        input.value_type,
        FeaType::Quantity(PhysicalDimension::FORCE)
    );
    assert!(!input.required);
    assert_eq!(
        declaration.outputs["displacement"],
        FeaType::Field(FeaFieldType {
            value: Box::new(FeaType::Vector {
                element: Box::new(FeaType::Quantity(PhysicalDimension::LENGTH)),
                length: Some(3),
            }),
            location: Some(runmat_analysis_core::FieldLocation::Node),
            quantity: Some(runmat_analysis_core::ObservationQuantity::Displacement),
        })
    );
}

#[test]
fn compiler_rejects_wrong_input_defaults_and_duplicate_outputs() {
    let source = r#"
study invalid {
    input load : force = 1 s
    output result = input.load
    output result = 2 s
    analysis solve {
        structural static {
        }
    }
}
"#;
    let host = TestHost::default();
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    let codes = program
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code.as_str())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"fea.semantic.input_value_type"));
    assert!(codes.contains(&"fea.semantic.duplicate_output"));
    assert!(!program.is_valid());
}

#[test]
fn model_header_region_parameters_are_required_and_strongly_typed() {
    let source = "model bracket(region body : region<solid>, region pad : region<face>) {\n}\n";
    let host = TestHost::default();
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let declaration = program.declarations.values().next().unwrap();
    assert_eq!(
        declaration.inputs["body"].value_type,
        FeaType::Region(FeaTopologyRole::Solid)
    );
    assert_eq!(
        declaration.inputs["pad"].value_type,
        FeaType::Region(FeaTopologyRole::Face)
    );
    assert!(declaration.inputs.values().all(|input| input.required));
}

#[test]
fn declaration_calls_require_every_typed_model_input() {
    let valid = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
}

mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
material steel {
    mass_density = 7850 kg/m^3
}
model bracket(region body : region<solid>) {
    input body_material : material
}
study solve {
    geometry = geometry.part
    mesh = mesh.part
    model = model.bracket(
        body = geometry.part.region.body,
        body_material = material.steel,
    )
    analysis static_case {
        structural static {
        }
    }
}
"#;
    let host = TestHost::default();
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", valid, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);

    let missing = valid.replace("        body = geometry.part.region.body,\n", "");
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &missing, 2),
        },
    });
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| { diagnostic.code == "fea.semantic.declaration_arguments" }));
    assert!(!program.is_valid());
}

#[test]
fn material_capabilities_lower_with_canonical_si_values() {
    let source = r#"
material steel {
    name = "Structural steel"
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
"#;
    let host = TestHost::default();
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let material = program.materials.values().next().unwrap();
    assert_eq!(material.mass_density_kg_per_m3, Some(7850.0));
    let mechanical = material.mechanical.as_ref().unwrap();
    assert_eq!(mechanical.youngs_modulus_pa, 200.0e9);
    assert_eq!(mechanical.poisson_ratio, 0.3);
}

#[test]
fn mesh_request_binds_geometry_and_lowers_units_into_canonical_settings() {
    let source = r#"
geometry bracket_geometry {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh bracket_mesh on geometry.bracket_geometry {
    tetrahedra {
        order = quadratic
        target_size = 5 mm
        maximum_deviation = 0.05 mm
        maximum_grading_ratio = 1.4
        deterministic_seed = 17
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let mesh = program.meshes.values().next().unwrap();
    assert_eq!(mesh.mesh_id, "bracket_mesh");
    assert_eq!(mesh.geometry.module_path, "project/main.fea");
    assert_eq!(mesh.geometry.geometry_id, "bracket_geometry");
    assert_eq!(
        mesh.settings.element_order,
        runmat_meshing_core::ElementOrder::Tet10
    );
    assert_eq!(mesh.settings.target_edge_length_m, 0.005);
    assert_eq!(mesh.settings.maximum_chordal_deviation_m, 0.00005);
    assert_eq!(mesh.settings.maximum_grading_ratio, 1.4);
    assert_eq!(mesh.settings.deterministic_seed, 17);
}

#[test]
fn mesh_request_rejects_missing_physical_settings_and_wrong_target_namespace() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
material steel { mass_density = 1 kg/m^3 }
mesh wrong_target on material.steel {
    tetrahedra {
        order = linear
        target_size = 5 mm
        maximum_deviation = 0.05 mm
    }
}
mesh incomplete on geometry.part {
    tetrahedra {
        order = linear
        target_size = 5 s
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program.meshes.is_empty());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.mesh" && diagnostic.message.contains("geometry` namespace")
    }));
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.mesh" && diagnostic.message.contains("maximum_deviation")
    }));
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.mesh" && diagnostic.message.contains("length quantity")
    }));
}

#[test]
fn material_lowering_rejects_empty_and_dimensionally_wrong_capabilities() {
    let source = r#"
material empty {
}
material invalid {
    elastic {
        youngs_modulus = 200 m
        poisson_ratio = 0.3
    }
}
"#;
    let host = TestHost::default();
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "fea.semantic.material"));
    assert!(program.materials.is_empty());
}

#[test]
fn imported_geometry_inspection_is_path_confined_and_retained() {
    let source = r#"
geometry part {
    file = "./geometry/part.step"
    units = mm
    region body : solid {
        select name "Part body"
        expect exactly_one
    }
    region mounting_faces : faces {
        select tag "mount"
        expect one_or_more
    }
}
"#;
    let host = TestHost::default().with_geometry("project/geometry/part.step", 8);
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let inspection = program.geometry_inspections.values().next().unwrap();
    assert_eq!(
        inspection.source_path.as_str(),
        "project/geometry/part.step"
    );
    assert_eq!(inspection.facts.content_digest, SourceDigest([8; 32]));
    assert!(inspection
        .facts
        .capabilities
        .contains(&GeometryInspectionCapability::PersistentIds));
    let geometry = program.geometries.values().next().unwrap();
    assert_eq!(geometry.geometry_id, "part");
    assert_eq!(geometry.regions.len(), 2);
    assert_eq!(
        geometry.regions[1].role,
        runmat_geometry_core::GeometryRegionRole::Face
    );
    assert_eq!(
        geometry.regions[1].cardinality,
        runmat_geometry_core::GeometryRegionCardinality::OneOrMore
    );
}

#[test]
fn constructed_and_derived_geometry_lower_parameterized_feature_graphs() {
    let source = r#"
geometry bracket {
    input width : length = 80 mm
    input thickness : length = 8 mm
    construct {
        feature stock = box(
            width = input.width,
            depth = input.thickness,
            height = 50 mm,
        )
        feature hole = cylinder(
            radius = 5 mm,
            height = input.thickness,
            axis = global([0, 1, 0]),
            origin = [15, 0, 15] mm,
        )
        feature finished = subtract(feature.stock, feature.hole)
        result = feature.finished
    }
    region body : solid {
        select result
        expect exactly_one
    }
}

geometry relieved from geometry.bracket {
    derive {
        feature relief = cylinder(
            radius = 8 mm,
            height = 8 mm,
            axis = global([0, 1, 0]),
            origin = [50, 0, 25] mm,
        )
        result = subtract(source.result, feature.relief)
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let bracket = program
        .geometries
        .iter()
        .find(|((_, key), _)| key.name == "bracket")
        .map(|(_, geometry)| geometry)
        .unwrap();
    assert_eq!(bracket.parameters.len(), 2);
    let runmat_geometry_core::GeometryDefinitionSource::Constructed(graph) = &bracket.source else {
        panic!("expected constructed geometry");
    };
    assert_eq!(graph.features.len(), 3);
    let relieved = program
        .geometries
        .iter()
        .find(|((_, key), _)| key.name == "relieved")
        .map(|(_, geometry)| geometry)
        .unwrap();
    let runmat_geometry_core::GeometryDefinitionSource::Derived(derived) = &relieved.source else {
        panic!("expected derived geometry");
    };
    assert_eq!(derived.source.geometry_id, "bracket");
    assert_eq!(derived.source.module_path, "project/main.fea");
    assert_eq!(derived.graph.result.0, "result");
}

#[test]
fn generated_geometry_freezes_function_provenance_and_typed_arguments() {
    let source = r#"
import function makeBracket from "./makeBracket.m"

geometry bracket {
    input width : length = 80 mm
    input depth : length = 20 mm
    generated by function.makeBracket {
        arguments {
            depth = input.depth
            width = input.width
        }
    }
    region body : solid {
        select output.region.body
        expect exactly_one
    }
}
"#;
    let width = runmat_types::OperationParameter {
        name: "width".to_string(),
        value: runmat_types::OperationTypePattern::Quantity(
            runmat_types::QuantityDimension::Length,
        ),
        arity: runmat_types::OperationParameterArity::Required,
        default: None,
    };
    let depth = runmat_types::OperationParameter {
        name: "depth".to_string(),
        value: runmat_types::OperationTypePattern::Quantity(
            runmat_types::QuantityDimension::Length,
        ),
        arity: runmat_types::OperationParameterArity::Required,
        default: None,
    };
    let host = TestHost::default().with_function(
        "project/makeBracket.m",
        "makeBracket",
        vec![width, depth],
        vec![runmat_types::OperationTypePattern::Analysis(
            runmat_types::AnalysisValueType::Geometry,
        )],
    );
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let geometry = program.geometries.values().next().unwrap();
    let runmat_geometry_core::GeometryDefinitionSource::Generated(generated) = &geometry.source
    else {
        panic!("expected generated geometry");
    };
    assert_eq!(generated.module_path, "project/makeBracket.m");
    assert_eq!(generated.function, "makeBracket");
    assert_eq!(generated.arguments[0].parameter, "width");
    assert_eq!(generated.arguments[1].parameter, "depth");
}

#[test]
fn geometry_feature_operations_reject_wrong_dimensions_and_duplicate_results() {
    let source = r#"
geometry invalid {
    construct {
        feature stock = box(width = 1 s, depth = 1 m, height = 1 m)
        result = feature.stock
        result = feature.stock
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.no_matching_overload"
            || diagnostic.message.contains("more than once")
    }));
    assert!(program.geometries.is_empty());
}

#[test]
fn derived_geometry_rejects_self_lineage() {
    let source = r#"
geometry recursive from geometry.recursive {
    derive {
        feature cut = cylinder(
            radius = 1 mm,
            height = 1 mm,
            axis = global([0, 0, 1]),
            origin = [0, 0, 0] mm,
        )
        result = subtract(source.result, feature.cut)
    }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("cannot derive from itself")));
    assert!(program.geometries.is_empty());
}
