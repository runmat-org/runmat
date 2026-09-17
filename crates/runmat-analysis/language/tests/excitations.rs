pub mod support;

use runmat_analysis_core::{
    AuthoredExcitationKind, AuthoredExcitationTarget, CoordinateFrame, Direction, Force,
    PhysicalQuantityKind, SignalId, SignalRepresentation, SignalShape, SignalValue,
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
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.message.contains("topology")
            || diagnostic.message.contains("physical quantity")
            || diagnostic.message.contains("at least one declared case")
            || diagnostic.message.contains("declared excitations")
    }));
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
