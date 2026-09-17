mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

const PART_AND_MESH: &str = r#"
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
"#;

fn check(source: &str) -> CheckedFeaProgram {
    let host = TestHost::default()
        .with_geometry("project/unused.step", 3)
        .with_function("project/unused.m", "unused", Vec::new(), Vec::new());
    FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

#[test]
fn study_realization_captures_the_exact_transitive_model_composition() {
    let library_module = r#"
model fastener {
}
model mounting {
    include model.fastener
}
"#;
    let source = r#"
import model mounting from "./library.fea"

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
model bracket {
    include model.mounting
}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.bracket
    analysis solve {
        structural static {
        }
    }
}
"#;
    let host = TestHost::default().with_module("project/library.fea", library_module, 2);
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let study = program.studies.values().next().unwrap();
    let root = program
        .models
        .values()
        .find(|model| model.model_id == "bracket")
        .unwrap();
    assert!(matches!(
        study.realize(root, &[]),
        Err(runmat_analysis_core::AuthoredStudyRealizationError::MissingModel(_))
    ));
    let catalog = program
        .models
        .values()
        .map(|model| (model.reference(), model.clone()))
        .collect();
    let realized = study.realize_with_models(&catalog, &[]).unwrap();
    assert_eq!(realized.model.model_id, "bracket");
    assert_eq!(realized.included_models.len(), 2);
    assert_eq!(realized.included_models[0].model_id, "fastener");
    assert_eq!(realized.included_models[1].model_id, "mounting");
    assert!(realized
        .included_models
        .iter()
        .all(|model| model.module_path == "project/library.fea"));
}

#[test]
fn study_retains_typed_inputs_bounds_and_exact_model_bindings() {
    let source = format!(
        r#"{PART_AND_MESH}
model loaded {{
    input service_load : force {{
        minimum = 0 N
        maximum = 5 kN
    }}
}}
study response {{
    input commanded_load : force {{
        default = 1 kN
        minimum = 0 N
        maximum = 2 kN
    }}
    geometry = geometry.part
    mesh = mesh.part
    model = model.loaded(service_load = input.commanded_load)
    analysis solve {{
        structural static {{
        }}
    }}
}}
"#
    );
    let program = check(&source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(program.models.values().next().unwrap().inputs.len(), 1);
    let study = program.studies.values().next().unwrap();
    assert_eq!(study.geometry.geometry_id, "part");
    assert_eq!(study.mesh.mesh_id, "part");
    assert_eq!(study.inputs.len(), 1);
    assert!(matches!(
        study.inputs[0].default,
        Some(runmat_analysis_core::AuthoredInputValue::Quantity {
            dimension: runmat_types::QuantityDimension::Force,
            value_si: 1000.0,
        })
    ));
    assert!(matches!(
        &study.model.arguments[0].value,
        runmat_analysis_core::AuthoredModelArgumentValue::StudyInput(reference)
            if reference.input_id == "commanded_load"
    ));
}

#[test]
fn study_preserves_the_defining_module_of_imported_declarations() {
    let library = format!("{PART_AND_MESH}\nmodel structure {{\n}}\n");
    let source = r#"
import geometry part from "./library.fea"
import mesh part from "./library.fea"
import model structure from "./library.fea"

study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis solve {
        structural static {
        }
    }
}
"#;
    let host = TestHost::default().with_module("project/library.fea", &library, 2);
    let program = FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let study = program.studies.values().next().unwrap();
    assert_eq!(study.module_path, "project/main.fea");
    assert_eq!(study.geometry.module_path, "project/library.fea");
    assert_eq!(study.mesh.module_path, "project/library.fea");
    assert_eq!(study.model.model.module_path, "project/library.fea");
}

#[test]
fn study_rejects_mesh_lineage_mismatch_and_out_of_bounds_model_constants() {
    let source = r#"
geometry first {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
geometry second {
    construct {
        feature body = box(width = 2 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh second on geometry.second {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model loaded {
    input service_load : force {
        minimum = 0 N
        maximum = 5 kN
    }
}
study response {
    geometry = geometry.first
    mesh = mesh.second
    model = model.loaded(service_load = 4 kN)
    analysis solve {
        structural static {
        }
    }
}
"#;
    let program = check(source);
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("mesh must be authored")));
    assert!(program.studies.is_empty());

    let source = source.replace("geometry = geometry.first", "geometry = geometry.second");
    let source = source.replace("service_load = 4 kN", "service_load = 6 kN");
    let program = check(&source);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("input binding lies outside its declared bounds")));
    assert!(program.studies.is_empty());
}

#[test]
fn study_lowers_typed_modal_state_flow_into_the_canonical_sequence() {
    let source = format!(
        r#"{PART_AND_MESH}
model structure {{
}}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis modes {{
        structural modal {{
            modes = 12
        }}
        retain basis
    }}
    analysis ringdown after analysis.modes {{
        structural modal_transient {{
            basis = analysis.modes.state.basis
            start_time = 0 s
            end_time = 2 s
        }}
    }}
}}
"#
    );
    let program = check(&source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let sequence = &program.studies.values().next().unwrap().sequence;
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
    let source = format!(
        r#"{PART_AND_MESH}
model structure {{
}}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis transient {{
        structural transient {{
        }}
    }}
}}
"#
    );
    let program = check(&source);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.procedure" && diagnostic.message.contains("start_time")
    }));
    assert!(program.studies.is_empty());
}

#[test]
fn study_realization_substitutes_regions_and_evaluates_parameterized_loads() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region pad : face {
        select result
        expect one_or_more
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model loaded(region pad : region<face>) {
    input service_load : force
    load downward {
        force on input.pad = [0, -input.service_load, 0]
    }
    load_case service { include load.downward }
}
study response {
    input commanded_load : force {
        default = 1 kN
        minimum = 0 N
        maximum = 5 kN
    }
    geometry = geometry.part
    mesh = mesh.part
    model = model.loaded(
        pad = geometry.part.region.pad,
        service_load = input.commanded_load,
    )
    analysis solve {
        structural static {
        }
    }
}
"#;
    let program = check(source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let authored_model = program.models.values().next().unwrap();
    let authored_study = program.studies.values().next().unwrap();
    let realized = authored_study.realize(authored_model, &[]).unwrap();
    assert!(realized.model.inputs.is_empty());
    assert!(realized.model.parameterized_signals.is_empty());
    let runmat_analysis_core::AuthoredExcitationTarget::Region(
        runmat_analysis_core::AuthoredRegionSource::Geometry(region),
    ) = &realized.model.excitations[0].target
    else {
        panic!("model region input must resolve to the exact study geometry region");
    };
    assert_eq!(region.region_id, "pad");
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &realized.model.signals[0] else {
        panic!("parameterized force must realize to a concrete force signal");
    };
    let runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Vector3(values),
    ) = &signal.representation
    else {
        panic!("parameterized load must realize as a constant vector");
    };
    assert_eq!(values[1].newtons, -1000.0);

    let overridden = authored_study
        .realize(
            authored_model,
            &[runmat_analysis_core::AuthoredStudyInputBinding {
                input_id: "commanded_load".into(),
                value: runmat_analysis_core::AuthoredInputValue::Quantity {
                    dimension: runmat_types::QuantityDimension::Force,
                    value_si: 2000.0,
                },
            }],
        )
        .unwrap();
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &overridden.model.signals[0] else {
        panic!("parameterized force must realize to a concrete force signal");
    };
    let runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Vector3(values),
    ) = &signal.representation
    else {
        panic!("parameterized load must realize as a constant vector");
    };
    assert_eq!(values[1].newtons, -2000.0);

    let out_of_bounds = authored_study.realize(
        authored_model,
        &[runmat_analysis_core::AuthoredStudyInputBinding {
            input_id: "commanded_load".into(),
            value: runmat_analysis_core::AuthoredInputValue::Quantity {
                dimension: runmat_types::QuantityDimension::Force,
                value_si: 6000.0,
            },
        }],
    );
    assert!(matches!(
        out_of_bounds,
        Err(runmat_analysis_core::AuthoredStudyRealizationError::InvalidStudyInput { .. })
    ));
}
