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
