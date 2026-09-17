pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

#[test]
fn typed_constant_signals_lower_to_canonical_physical_signals() {
    let source = r#"
model forcing {
    signal pressure_bias : pressure over constant {
        value = 2 kPa
    }
    signal force_vector : force over constant {
        value = [1 N, -2 N, 3 N]
        frame = global
    }
    signal temperature_bias : temperature over constant {
        value = 20 degC
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
    assert_eq!(model.signals.len(), 3);
    assert_eq!(
        model.signals[0].quantity(),
        runmat_analysis_core::PhysicalQuantityKind::Pressure
    );
    assert_eq!(
        model.signals[1].shape(),
        &runmat_analysis_core::SignalShape::Vector3
    );
    assert_eq!(
        model.signals[2].domain(),
        runmat_analysis_core::SignalDomainKind::Constant
    );
}

#[test]
fn signal_headers_values_and_frames_fail_closed() {
    let source = r#"
model invalid {
    signal wrong_dimension : force over constant {
        value = 1 Pa
    }
    signal missing_frame : acceleration over constant {
        value = [1 m/s^2, 0 m/s^2, 0 m/s^2]
    }
    signal scalar_frame : pressure over constant {
        value = 1 Pa
        frame = global
    }
    signal unavailable : force over time {
        value = 1 N
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
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.message.contains("coordinate frame")
            || diagnostic.message.contains("explicit sampled or generated")
    }));
}
