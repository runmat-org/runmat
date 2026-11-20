use std::fs;

use runmat_analysis_core::{EvaluatedSignalValue, PhysicalQuantityKind};
use tempfile::tempdir;

use super::{prepare_fea_path_in_root, CheckedFeaSource};

const SOURCE: &str = r#"
geometry bracket {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
    region seat : face {
        select result
        expect one_or_more
    }
}
mesh bracket on geometry.bracket {
    tetrahedra {
        order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m
    }
}
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 210 GPa
        poisson_ratio = 0.3
    }
}
model loaded_bracket {
    input service_load : force = 3 kN
    assign material.steel to geometry.bracket.region.body
    constraint mounting {
        structural.fixed { target = geometry.bracket.region.seat }
    }
    load hanging {
        force on geometry.bracket.region.seat = [0 N, -input.service_load, 0 N]
    }
    load_case service { include load.hanging }
}
study bracket_static {
    input service_load : force = 2 kN
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket(service_load = input.service_load)
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
}
study model_default {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
}
study case_bound {
    input service_load : force
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket(service_load = input.service_load)
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
}
sweep load_sweep over study.bracket_static {
    axis service_load {
        light = 1 kN
        heavy = 4 kN
    }
    combine = product
}
sweep case_sweep over study.case_bound {
    axis service_load {
        light = 1 kN
        heavy = 4 kN
    }
    combine = product
}
"#;

fn checked() -> CheckedFeaSource {
    let _filesystem = runmat_filesystem::provider_override_lock();
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("main.fea");
    fs::write(&path, SOURCE).unwrap();
    let checked = prepare_fea_path_in_root(temporary.path(), &path)
        .unwrap()
        .check(Vec::new(), Vec::new())
        .unwrap();
    assert!(
        checked.program.is_valid(),
        "{:#?}",
        checked.program.diagnostics
    );
    checked
}

fn hanging(
    report: &runmat_fea_source::FeaSourceCheckResult,
    study_id: &str,
) -> runmat_analysis_core::EvaluatedParameterizedSignal {
    let study = report
        .program
        .evaluated_signals
        .iter()
        .find(|study| study.study_id == study_id)
        .unwrap_or_else(|| panic!("no evaluated signals for {study_id}"));
    assert_eq!(study.module_path.as_str(), "main.fea");
    assert_eq!(study.signals.len(), 1);
    study.signals[0].clone()
}

#[test]
fn check_reports_each_study_signal_at_the_study_own_inputs() {
    let report = checked().check_result();
    // The sweep binds 1 kN and 4 kN per case; the study on its own runs at its 2 kN default.
    let signal = hanging(&report, "bracket_static");
    assert_eq!(signal.quantity, PhysicalQuantityKind::Force);
    assert_eq!(signal.unit, "N");
    assert_eq!(
        signal.value_si,
        Some(EvaluatedSignalValue::Vector3([0.0, -2000.0, 0.0]))
    );
    // Without a model argument the model-input default applies, as in a run.
    assert_eq!(
        hanging(&report, "model_default").value_si,
        Some(EvaluatedSignalValue::Vector3([0.0, -3000.0, 0.0]))
    );
    // Only sweep cases bind this study's input, so the study alone has no value.
    let unbound = hanging(&report, "case_bound");
    assert_eq!(unbound.unit, "N");
    assert_eq!(unbound.value_si, None);

    let json = serde_json::to_value(&report.program).unwrap();
    assert_eq!(
        json["evaluatedSignals"][0]["signals"][0]["value_si"],
        serde_json::json!([0.0, -2000.0, 0.0])
    );
}

#[test]
fn evaluated_signals_never_change_the_checked_revision() {
    let checked = checked();
    let report = checked.check_result();
    assert!(!report.program.evaluated_signals.is_empty());
    let diagnostics = checked.diagnostics().cloned().collect::<Vec<_>>();
    let mut with_signals = report.program.clone();
    let mut without_signals = report.program.clone();
    without_signals.evaluated_signals.clear();
    let revision = checked.checked_revision(&mut with_signals, &diagnostics);
    assert_eq!(revision, report.revision);
    assert_eq!(
        checked.checked_revision(&mut without_signals, &diagnostics),
        report.revision
    );
    assert_eq!(with_signals, report.program);
}
