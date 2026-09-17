mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

#[test]
fn compiler_checks_inputs_defaults_locals_and_outputs_as_one_program() {
    let source = r#"
study load_study {
    input load : force {
        default = 1 kN
        minimum = 0 N
        maximum = 5 kN
    }
    let doubled = 2 * input.load
    output applied = doubled
}
"#;
    let host = TestHost::default().with_module(
        "project/library.fea",
        "material library_material {\n}\n",
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
    let declaration = program.declarations.values().next().unwrap();
    let input = &declaration.inputs["load"];
    assert_eq!(
        input.value_type,
        FeaType::Quantity(PhysicalDimension::FORCE)
    );
    assert!(!input.required);
    assert_eq!(
        declaration.outputs["applied"],
        FeaType::Quantity(PhysicalDimension::FORCE)
    );
}

#[test]
fn compiler_rejects_wrong_input_defaults_and_duplicate_outputs() {
    let source = r#"
study invalid {
    input load : force = 1 s
    output result = input.load
    output result = 2 s
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
    assert!(codes.contains(&"fea.semantic.input_default_type"));
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
    region body : solid {
        select result
    }
}
material steel {
    mass_density = 7850 kg/m^3
}
model bracket(region body : region<solid>) {
    input body_material : material
}
study solve {
    model = model.bracket(
        body = geometry.part.region.body,
        body_material = material.steel,
    )
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
fn imported_geometry_inspection_is_path_confined_and_retained() {
    let source = "geometry part {\nfile = \"./geometry/part.step\"\nunits = mm\n}\n";
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
}
