pub mod support;

use runmat_analysis_core::{AuthoredExcitationKind, ExcitationDistribution, SignalId};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

const SOURCE: &str = r#"
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
model driven {
    signal incoming : pressure over constant { value = 2 Pa }
    excitation wave {
        incident_pressure {
            target = geometry.part.region.wall
            signal = signal.incoming
        }
    }
    load_case service { include excitation.wave }
}
"#;

fn check(source: &str) -> CheckedFeaProgram {
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

#[test]
fn incident_pressure_retains_shared_pressure_signal() {
    let program = check(SOURCE);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    let excitation = &model.excitations[0];
    assert_eq!(
        excitation.kind,
        AuthoredExcitationKind::AcousticIncidentPressure
    );
    assert_eq!(excitation.signal_id, SignalId("incoming".into()));
    assert_eq!(
        excitation.distribution,
        ExcitationDistribution::UniformKinematic
    );
    assert!(excitation.direction.is_none());
    assert_eq!(model.signals.len(), 1);
}

#[test]
fn incident_pressure_retains_complex_frequency_spectrum() {
    let source = SOURCE.replace(
        "signal incoming : pressure over constant { value = 2 Pa }",
        r#"signal incoming : pressure over frequency {
            samples = [
                { at = 10 Hz, value = { real = 2 Pa, imaginary = -3 Pa } },
                { at = 20 Hz, value = { real = 4 Pa, imaginary = -5 Pa } },
            ]
            interpolation = linear
            extrapolation = reject
        }"#,
    );
    let program = check(&source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    let runmat_analysis_core::PhysicalSignal::Pressure(signal) = &model.signals[0] else {
        panic!("incident source must retain pressure authority");
    };
    let runmat_analysis_core::SignalRepresentation::Tabulated(table) = &signal.representation
    else {
        panic!("frequency spectrum must remain tabulated");
    };
    assert_eq!(table.knots.len(), 2);
    assert_eq!(
        table.domain,
        runmat_analysis_core::TabulatedSignalDomain::FrequencyHertz
    );
    assert_eq!(
        table.knots[0].value,
        runmat_analysis_core::SignalKnotValue::Continuous(
            runmat_analysis_core::SignalOrdinate::Complex {
                real: runmat_analysis_core::SignalValue::Scalar(
                    runmat_analysis_core::Pressure::new(2.0)
                ),
                imaginary: runmat_analysis_core::SignalValue::Scalar(
                    runmat_analysis_core::Pressure::new(-3.0)
                ),
            }
        )
    );
    assert_eq!(model.excitations[0].signal_id, signal.signal_id);
}

#[test]
fn incident_pressure_rejects_wrong_quantity_direction_and_follower_policy() {
    for source in [
        SOURCE.replace(
            "pressure over constant { value = 2 Pa }",
            "velocity over constant { value = 2 m/s }",
        ),
        SOURCE.replace(
            "signal = signal.incoming",
            "signal = signal.incoming\n direction = global([1, 0, 0])",
        ),
        SOURCE.replace(
            "signal = signal.incoming",
            "signal = signal.incoming\n kinematics = follower",
        ),
    ] {
        let program = check(&source);
        assert!(!program.is_valid(), "invalid incident policy was accepted");
    }
}
