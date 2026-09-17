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
    analysis solve {
        structural static {
        }
    }
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
fn study_lowers_typed_modal_state_flow_into_the_canonical_sequence() {
    let source = r#"
study response {
    analysis modes {
        structural modal {
            modes = 12
        }
        retain basis
    }
    analysis ringdown after analysis.modes {
        structural modal_transient {
            basis = analysis.modes.state.basis
            start_time = 0 s
            end_time = 2 s
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
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let sequence = program.analysis_sequences.values().next().unwrap();
    assert_eq!(sequence.steps.len(), 2);
    assert_eq!(sequence.steps[0].state_outputs[0].state_id.0, "basis");
    assert_eq!(
        sequence.steps[1].solution_space,
        runmat_analysis_core::SolutionSpace::ModalReduced
    );
    assert_eq!(
        sequence.dependencies[0].reason,
        runmat_analysis_core::SequenceDependencyReason::StateTransfer
    );
    assert_eq!(sequence.transitions[0].state_id.0, "basis");
}

#[test]
fn procedure_requiredness_rejects_hidden_physical_defaults() {
    let source = r#"
study response {
    analysis transient {
        structural transient {
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
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.procedure" && diagnostic.message.contains("start_time")
    }));
    assert!(program.analysis_sequences.is_empty());
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
