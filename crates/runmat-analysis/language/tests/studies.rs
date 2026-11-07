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

// witness: FEA-AUTHORING-LANGUAGE-002/formulation_admission @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: A study without any formulation setting checks and realizes; the presence of no
//   setting is asserted, remote and sweep non-implication are not.
#[test]
fn study_has_no_formulation_setting() {
    let study = |setting: &str| {
        format!(
            r#"
{PART_AND_MESH}
model structure {{}}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    {setting}
    analysis solve {{ structural static {{}} }}
}}
"#
        )
    };
    let plain = check(&study(""));
    assert!(plain.is_valid(), "{:#?}", plain.diagnostics);
    let authored = plain.studies.values().next().unwrap();
    let geometry = plain.geometries.values().next().unwrap();
    let models = plain
        .models
        .values()
        .map(|model| (model.reference(), model.clone()))
        .collect();
    authored
        .realize_with_models(&models, geometry, &[])
        .unwrap();

    for stale in [
        "formulation = production",
        "require formulation = production",
    ] {
        let rejected = check(&study(stale));
        assert!(!rejected.is_valid(), "{stale}");
        assert!(
            rejected.diagnostics.iter().any(|diagnostic| diagnostic
                .message
                .contains("`formulation` is not a study setting")),
            "{stale}: {:#?}",
            rejected.diagnostics
        );
    }
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
        Some(runmat_analysis_core::AuthoredInputValue::Quantity(value))
            if value.dimension() == runmat_types::PhysicalDimension::FORCE && value.canonical_value() == 1000.0
    ));
    assert!(matches!(
        &study.model.arguments[0].value,
        runmat_analysis_core::AuthoredModelArgumentValue::StudyInput(reference)
            if reference.input_id == "commanded_load"
    ));
}

// witness: FEA-AUTHORING-COMPILER-002/output_lowering @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Outputs lower to fields, projections and attributed spatial extrema; catalog overload
//   identity is covered in the operations tests.
// witness: FEA-AUTHORING-OUTPUT-002/attributed_extrema @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Spatial extrema lower with their attributed region; tie policy and independent-domain
//   coordinates are not asserted.
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

// witness: FEA-AUTHORING-LANGUAGE-002/imports_and_scopes @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Imported geometry, mesh and model keep their defining module identity through study
//   resolution; function and package import forms and scope shadowing rules are not covered.
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

// witness: FEA-AUTHORING-COMPILER-002/study_and_sweep_lowering @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Mesh-lineage mismatch and out-of-bounds model constants are rejected when a study
//   binds geometry and mesh; sweep lowering is covered elsewhere.
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

// witness: FEA-AUTHORING-LANGUAGE-002/semantic_state_flow @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: A modal analysis retaining a basis feeds a modal transient through a named state
//   reference lowered into the canonical sequence; other state kinds and the study-as-runnable-root
//   rule are not covered.
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

// witness: FEA-AUTHORING-LANGUAGE-002/declarative_semantics @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Modal residual vectors and modal-reduced static correction lower into the typed
//   procedure settings; malformed values are rejected with a procedure diagnostic.
#[test]
fn modal_reduction_enrichment_lowers_into_typed_procedure_settings() {
    let study = |modal: &str, harmonic: &str| {
        format!(
            r#"{PART_AND_MESH}
model structure {{
}}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis modes {{
        structural modal {{
            modes = 8
            {modal}
        }}
        retain basis
    }}
    analysis sweep after analysis.modes {{
        structural modal_harmonic {{
            basis = analysis.modes.state.basis
            frequencies = [20, 40] Hz
            {harmonic}
        }}
    }}
}}
"#
        )
    };
    let program = check(&study(
        "residual_vectors = [[0, 0, 1], [1, 0, 0]]",
        "static_correction = applied_loads",
    ));
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let sequence = &program.studies.values().next().unwrap().sequence;
    assert_eq!(
        sequence.steps[0]
            .modal_residual_vector_directions()
            .iter()
            .map(|direction| direction.vector)
            .collect::<Vec<_>>(),
        vec![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    );
    assert_eq!(
        sequence.steps[1].modal_static_correction(),
        runmat_analysis_core::ModalStaticCorrection::AppliedLoads
    );

    let plain = check(&study("", ""));
    assert!(plain.is_valid(), "{:#?}", plain.diagnostics);
    let sequence = &plain.studies.values().next().unwrap().sequence;
    assert!(sequence.steps[0].modal_residual_vector_directions().is_empty());
    assert_eq!(
        sequence.steps[1].modal_static_correction(),
        runmat_analysis_core::ModalStaticCorrection::None
    );

    for (modal, harmonic) in [
        ("residual_vectors = [[0, 0, 0]]", ""),
        ("residual_vectors = [[0, 0, 1], [0, 0, 2]]", ""),
        ("residual_vectors = [0, 0, 1]", ""),
        ("", "static_correction = inertia"),
    ] {
        let rejected = check(&study(modal, harmonic));
        assert!(!rejected.is_valid(), "{modal} {harmonic} must be rejected");
        assert!(rejected
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.starts_with("fea.semantic")));
    }
}

// witness: FEA-AUTHORING-LANGUAGE-002/declarative_semantics @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: A transient procedure with no start_time is rejected with a located procedure
//   diagnostic and no study is produced, so physical defaults are not invented; covers the
//   transient requiredness case only, not every declaration kind or the three principles as a
//   whole.
// witness: FEA-AUTHORING-LANGUAGE-002/declaration_requiredness @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: One required procedure member (start_time) is enforced for structural transient; the
//   full required/optional member table for all declaration kinds is not exercised.
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
