pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

#[test]
fn initial_conditions_lower_uniform_signal_sample_and_field_sources() {
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
model prepared {
    signal thermal_history : temperature over time {
        samples = [
            { at = 0 s, value = 293.15 K },
            { at = 1 s, value = 303.15 K },
        ]
        interpolation = linear
        extrapolation = reject
    }
    initial at_rest {
        structural.velocity {
            target = whole_model
            uniform = [0 m/s, 0 m/s, 0 m/s]
            frame = global
        }
    }
    initial warm_start {
        thermal.temperature {
            target = geometry.part.region.body
            sample {
                signal = signal.thermal_history
                at = 0 s
                approach = right_limit
            }
        }
    }
    initial imported_pressure {
        acoustic.pressure {
            target = whole_model
            field_artifact {
                artifact_id = "state-42"
                content_digest = "sha256:content"
                topology_digest = "sha256:topology"
                field_id = "pressure"
            }
        }
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
    assert_eq!(model.initial_conditions.len(), 3);
    assert!(matches!(
        &model.initial_conditions[0].kind,
        runmat_analysis_core::AuthoredInitialConditionKind::Velocity(
            runmat_analysis_core::InitialValue::UniformVector3(values)
        ) if values.iter().all(|value| value.meters_per_second == 0.0)
    ));
    assert!(matches!(
        &model.initial_conditions[1].kind,
        runmat_analysis_core::AuthoredInitialConditionKind::Temperature(
            runmat_analysis_core::InitialValue::SignalSample {
                signal_id,
                abscissa,
                approach: runmat_analysis_core::SignalSampleApproach::RightLimit,
            }
        ) if signal_id.0 == "thermal_history" && *abscissa == 0.0
    ));
    assert!(matches!(
        &model.initial_conditions[2].kind,
        runmat_analysis_core::AuthoredInitialConditionKind::AcousticPressure(
            runmat_analysis_core::InitialValue::FieldArtifact(artifact)
        ) if artifact.artifact_id == "state-42"
    ));
}

#[test]
fn initial_conditions_reject_wrong_shape_domain_dimension_and_temperature() {
    let source = r#"
model invalid_shape {
    signal scalar_velocity : velocity over time {
        samples = [
            { at = 0 s, value = 0 m/s },
            { at = 1 s, value = 1 m/s },
        ]
        interpolation = linear
        extrapolation = reject
    }
    initial velocity {
        structural.velocity {
            target = whole_model
            frame = global
            sample {
                signal = signal.scalar_velocity
                at = 0 s
                approach = at
            }
        }
    }
}
model invalid_frequency {
    signal pressure_spectrum : pressure over frequency {
        samples = [
            { at = 1 Hz, value = 1 Pa },
            { at = 2 Hz, value = 2 Pa },
        ]
        interpolation = linear
        extrapolation = reject
    }
    initial pressure {
        acoustic.pressure {
            target = whole_model
            sample {
                signal = signal.pressure_spectrum
                at = 1 Hz
                approach = at
            }
        }
    }
}
model invalid_units {
    initial velocity {
        structural.velocity {
            target = whole_model
            uniform = [0 m, 0 m, 0 m]
            frame = global
        }
    }
}
model invalid_temperature {
    initial temperature {
        thermal.temperature {
            target = whole_model
            uniform = -1 K
        }
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
    assert!(program.models.is_empty());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("exact quantity and shape")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("frequency-domain signals")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("compatible units")));
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("zero kelvin")));
}
