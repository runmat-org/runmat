pub mod support;

use runmat_analysis_core::{
    AuthoredExcitationKind, AuthoredExcitationTarget, AuthoredNumericExpressionKind,
    AuthoredSignalExpression, CoordinateFrame, Direction, Force, PhysicalQuantityKind, SignalId,
    SignalRepresentation, SignalShape, SignalValue,
};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

#[test]
fn applied_loads_lower_to_typed_signals_actions_and_reusable_case_membership() {
    let source = r#"
geometry bracket {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region pad : face {
        select result
        expect one_or_more
    }
}
model loaded_bracket {
    load downward {
        force on geometry.bracket.region.pad = [0, -1, 0] kN
    }
    load_case service {
        include load.downward
    }
    load_case proof {
        include load.downward
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
    let model = program.models.values().next().unwrap();
    assert_eq!(model.signals.len(), 1);
    assert_eq!(model.excitations.len(), 1);
    assert_eq!(model.excitation_cases.len(), 2);

    let signal = &model.signals[0];
    assert_eq!(signal.quantity(), PhysicalQuantityKind::Force);
    assert_eq!(signal.shape(), &SignalShape::Vector3);
    let runmat_analysis_core::PhysicalSignal::Force(signal) = signal else {
        panic!("force load must produce a force signal");
    };
    assert_eq!(
        signal.representation,
        SignalRepresentation::Constant(SignalValue::Vector3([
            Force::new(0.0),
            Force::new(-1000.0),
            Force::new(0.0),
        ]))
    );

    let excitation = &model.excitations[0];
    assert_eq!(excitation.kind, AuthoredExcitationKind::Force);
    assert!(matches!(
        excitation.target,
        AuthoredExcitationTarget::Region(_)
    ));
    assert!(model
        .excitation_cases
        .iter()
        .all(|case| case.excitation_ids == ["load.downward"]));
}

#[test]
fn model_input_loads_retain_a_typed_parameter_dependency_graph() {
    let source = r#"
model loaded(region pad : region<face>) {
    input service_load : force
    load downward {
        force on input.pad = [0, -input.service_load * 2, 0]
    }
    load_case service { include load.downward }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert!(model.signals.is_empty());
    assert_eq!(model.parameterized_signals.len(), 1);
    let signal = &model.parameterized_signals[0];
    assert_eq!(signal.quantity, PhysicalQuantityKind::Force);
    let AuthoredSignalExpression::Vector3(values) = &signal.value else {
        panic!("force vector must retain three typed expressions");
    };
    let AuthoredNumericExpressionKind::Binary { left, right, .. } = &values[1].kind else {
        panic!("second component must retain the typed scaling");
    };
    let AuthoredNumericExpressionKind::Negate { value } = &left.kind else {
        panic!("left operand must retain the authored negation");
    };
    assert!(matches!(
        &value.kind,
        AuthoredNumericExpressionKind::ModelInput { input }
            if input.input_id == "service_load"
    ));
    assert!(matches!(
        right.kind,
        AuthoredNumericExpressionKind::Literal { value } if value == 2.0
    ));
}

#[test]
fn applied_loads_fail_closed_on_bad_units_topology_and_case_membership() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region volume : solid {
        select result
        expect exactly_one
    }
    region wall : face {
        select result
        expect one_or_more
    }
}

model wrong_topology {
    load pressure {
        pressure on geometry.part.region.volume = 1 MPa
    }
    load_case service { include load.pressure }
}
model wrong_units {
    load force {
        force on geometry.part.region.wall = 1 Pa
    }
    load_case service { include load.force }
}
model unassigned {
    load force {
        force on geometry.part.region.wall = [0 N, 1 N, 0 N]
    }
    load other {
        force on geometry.part.region.wall = [0 N, 2 N, 0 N]
    }
    load_case service { include load.other }
}
model unknown_member {
    load_case service { include load.missing }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program.models.is_empty());
    let messages = program
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages.contains(&"force must be force, found Pa"),
        "{messages:#?}"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("at least one declared case")),
        "a model that declares a case must place every load in a declared case: {messages:#?}"
    );
    assert!(
        messages.iter().any(|message| message.contains("topology")
            || message.contains("entity")
            || message.contains("face")),
        "{messages:#?}"
    );
}

const FIGURE_GEOMETRY: &str = r#"
geometry beam {
    construct {
        feature body = box(width = 1 m, depth = 0.1 m, height = 0.1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
    region tip : face {
        select result
        expect one_or_more
    }
}
mesh beam on geometry.beam {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
"#;

const FIGURE_STUDY: &str = r#"
study cantilever_static {
    geometry = geometry.beam
    mesh = mesh.beam
    model = model.loaded_beam
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
}
"#;

fn figure_source(traction: &str) -> String {
    format!(
        "{FIGURE_GEOMETRY}model loaded_beam {{\n    assign material.steel to geometry.beam.region.body\n    load end_traction {{\n        traction on geometry.beam.region.tip = {traction}\n    }}\n}}\n{FIGURE_STUDY}"
    )
}

fn check_figure(source: &str) -> CheckedFeaProgram {
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

#[test]
fn a_model_without_load_cases_applies_its_loads_in_one_default_case() {
    let program = check_figure(&figure_source("[0, 0, -100000] Pa"));
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(model.excitation_cases.len(), 1);
    assert_eq!(model.excitation_cases[0].case_id, "load_case.default");
    assert_eq!(
        model.excitation_cases[0].excitation_ids,
        ["load.end_traction"]
    );
    assert_eq!(program.studies.len(), 1);
}

#[test]
fn a_unit_mismatch_is_located_at_the_value_and_names_expected_and_found_units() {
    let source = figure_source("[0, 0, -100000] N");
    let program = check_figure(&source);
    assert!(!program.is_valid());
    assert_eq!(
        program.diagnostics.len(),
        1,
        "the dependent study must not repeat the model failure: {:#?}",
        program.diagnostics
    );
    let diagnostic = &program.diagnostics[0];
    assert_eq!(diagnostic.code, "fea.semantic.model");
    assert_eq!(
        diagnostic.message,
        "traction must be force per area, found N"
    );
    assert_eq!(diagnostic.label.as_deref(), Some("expected Pa"));
    let span = diagnostic.location.as_ref().unwrap().span;
    assert_eq!(&source[span.start..span.end], "[0, 0, -100000] N");
}

#[test]
fn explicit_excitations_bind_declared_signals_without_copying_signal_authority() {
    let source = r#"
geometry bracket {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region pad : face {
        select result
        expect one_or_more
    }
}
model loaded_bracket {
    signal thrust : force over time {
        samples = [
            { at = 0 s, value = 0 N },
            { at = 1 s, value = 4 kN },
        ]
        interpolation = linear
        extrapolation = reject
    }
    excitation axial {
        force {
            target = geometry.bracket.region.pad
            signal = signal.thrust
            direction = global([1, 0, 0])
        }
    }
    load_case service { include excitation.axial }
    load_case proof { include excitation.axial }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(model.signals.len(), 1);
    assert_eq!(model.excitations.len(), 1);
    assert_eq!(model.excitation_cases.len(), 2);
    let excitation = &model.excitations[0];
    assert_eq!(excitation.excitation_id, "excitation.axial");
    assert_eq!(excitation.kind, AuthoredExcitationKind::Force);
    assert_eq!(excitation.signal_id, SignalId("thrust".into()));
    assert_eq!(excitation.frame, CoordinateFrame::Global);
    assert_eq!(
        excitation.direction,
        Some(Direction {
            vector: [1.0, 0.0, 0.0]
        })
    );
    assert!(model
        .excitation_cases
        .iter()
        .all(|case| case.excitation_ids == ["excitation.axial"]));
}

#[test]
fn explicit_excitations_fail_closed_on_signal_shape_quantity_and_direction() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region wall : face {
        select result
        expect one_or_more
    }
}
model wrong_quantity {
    signal history : pressure over constant { value = 1 Pa }
    excitation push {
        force {
            target = geometry.part.region.wall
            signal = signal.history
            direction = global([1, 0, 0])
        }
    }
    load_case service { include excitation.push }
}
model non_unit_direction {
    signal history : force over constant { value = 1 N }
    excitation push {
        force {
            target = geometry.part.region.wall
            signal = signal.history
            direction = global([2, 0, 0])
        }
    }
    load_case service { include excitation.push }
}
model missing_direction {
    signal history : force over constant { value = 1 N }
    excitation push {
        force {
            target = geometry.part.region.wall
            signal = signal.history
        }
    }
    load_case service { include excitation.push }
}
model unknown_signal {
    excitation push {
        force {
            target = geometry.part.region.wall
            signal = signal.missing
            direction = global([1, 0, 0])
        }
    }
    load_case service { include excitation.push }
}
"#;
    let program = FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    });
    assert!(!program.is_valid());
    assert!(program.models.is_empty());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("wrong physical quantity")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("finite unit vector")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("signal shape")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("unknown signal")));
}
