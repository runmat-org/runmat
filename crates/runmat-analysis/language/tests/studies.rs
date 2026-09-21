mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

const PART_AND_MESH: &str = r#"
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
fn study_formulation_maturity_is_explicit_and_defaults_to_production() {
    let study = |requirement: &str| {
        format!(
            r#"
{PART_AND_MESH}
model structure {{}}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    {requirement}
    analysis solve {{ structural static {{}} }}
}}
"#
        )
    };
    let production = check(&study(""));
    assert!(production.is_valid(), "{:#?}", production.diagnostics);
    assert_eq!(
        production
            .studies
            .values()
            .next()
            .unwrap()
            .formulation_requirement,
        runmat_analysis_core::FormulationRequirement::Production
    );

    let experimental = check(&study("formulation = experimental"));
    assert!(experimental.is_valid(), "{:#?}", experimental.diagnostics);
    let authored = experimental.studies.values().next().unwrap();
    assert_eq!(
        authored.formulation_requirement,
        runmat_analysis_core::FormulationRequirement::Experimental
    );
    let geometry = experimental.geometries.values().next().unwrap();
    let models = experimental
        .models
        .values()
        .map(|model| (model.reference(), model.clone()))
        .collect();
    assert_eq!(
        authored
            .realize_with_models(&models, geometry, &[])
            .unwrap()
            .formulation_requirement,
        runmat_analysis_core::FormulationRequirement::Experimental
    );

    let invalid = check(&study("formulation = preview"));
    assert!(!invalid.is_valid());
    assert!(invalid.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("must be `production` or `experimental`")));

    let retired_spelling = check(&study("require formulation = experimental"));
    assert!(!retired_spelling.is_valid());
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
    assert_eq!(study.geometry.definition.geometry_id, "part");
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
fn study_outputs_lower_fields_projections_and_attributed_spatial_extrema() {
    let source = r#"
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
model structure {
}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis static {
        structural static {
        }
    }
    output displacement = analysis.static.field.node.displacement
    output displacement_magnitude = magnitude(output.displacement)
    output peak_displacement = max(
        output.displacement_magnitude,
        over = geometry.part.region.body,
    )
}
"#;
    let program = check(source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let study = program.studies.values().next().unwrap();
    assert_eq!(study.outputs.len(), 3);
    assert!(matches!(
        study.outputs[0].expression,
        runmat_analysis_core::AuthoredOutputExpression::AnalysisField {
            quantity: runmat_analysis_core::ObservationQuantity::Displacement,
            ..
        }
    ));
    assert!(matches!(
        study.outputs[1].expression,
        runmat_analysis_core::AuthoredOutputExpression::Projection {
            projection: runmat_analysis_core::AuthoredFieldProjection::Magnitude,
            ..
        }
    ));
    let runmat_analysis_core::AuthoredOutputExpression::SpatialReduction {
        reduction,
        target,
        ties,
        ..
    } = &study.outputs[2].expression
    else {
        panic!("peak output must retain its attributed spatial reduction");
    };
    assert_eq!(
        *reduction,
        runmat_analysis_core::AuthoredSpatialReduction::Maximum
    );
    assert_eq!(
        *ties,
        runmat_analysis_core::AuthoredExtremaTiePolicy::FirstCanonical
    );
    let runmat_analysis_core::AuthoredRegionSource::Geometry(region) = target else {
        panic!("spatial reduction must retain an exact geometry region");
    };
    assert_eq!(region.geometry_id, "part");
    assert_eq!(region.region_id, "body");
}

#[test]
fn study_outputs_reject_vector_extrema_and_fields_the_step_cannot_produce() {
    let source = format!(
        r#"{PART_AND_MESH}
model structure {{
}}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis static {{
        structural static {{
        }}
    }}
    output invalid_temperature = analysis.static.field.node.temperature
    output invalid_vector_max = max(
        analysis.static.field.node.displacement,
        over = geometry.part.region.body,
    )
}}
"#
    );
    let program = check(&source);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("explicit scalar field projection")));
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("does not produce `Temperature`")));
    assert!(program.studies.is_empty());
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
    assert_eq!(study.geometry.definition.module_path, "project/library.fea");
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
            time_step = 0.1 s
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
    assert!(matches!(
        sequence.steps[1].independent_domain,
        runmat_analysis_core::IndependentDomain::TimeSeconds {
            interval_count: 20,
            ..
        }
    ));
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
fn transient_time_step_is_dimensioned_and_must_partition_the_authored_domain() {
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
            start_time = 0 s
            end_time = 1 s
            time_step = 0.3 s
        }}
    }}
}}
"#
    );
    let program = check(&source);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.procedure"
            && diagnostic.message.contains("exactly partition")
    }));

    let wrong_dimension = source.replace("0.3 s", "0.25 m");
    let program = check(&wrong_dimension);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "fea.semantic.procedure"
            && diagnostic.message.contains("time quantity `time_step`")
    }));
}
