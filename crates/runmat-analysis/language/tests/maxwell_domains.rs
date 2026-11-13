pub mod support;
use runmat_analysis_core::{AuthoredConstraintKind, ElectromagneticDomain, Frequency};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

const SOURCE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region surface : face { select result; expect one_or_more }
}
model field {
    electromagnetic { equation = harmonic_maxwell
        reference_frequency = 2 kHz }
    constraint wall {
        electrical.perfect_conductor { target = geometry.part.region.surface }
    }
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
fn authored_harmonic_maxwell_preserves_frequency_and_distinct_pec() {
    let checked = check(SOURCE);
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let model = checked.models.values().next().unwrap();
    assert_eq!(
        model.electromagnetic,
        Some(ElectromagneticDomain::Harmonic {
            reference_frequency: Frequency::new(2000.0),
        })
    );
    assert!(matches!(
        model.constraints[0].kind,
        AuthoredConstraintKind::ElectricPerfectConductor { .. }
    ));
}

#[test]
fn harmonic_frequency_is_explicit_positive_dimensional_and_unique() {
    for frequency in ["0 Hz", "-1 Hz", "1 m", "60"] {
        assert!(
            !check(&SOURCE.replace("2 kHz", frequency)).is_valid(),
            "accepted {frequency}"
        );
    }
    for source in [
        SOURCE.replace("reference_frequency = 2 kHz", ""),
        SOURCE.replace(
            "reference_frequency = 2 kHz",
            "reference_frequency = 2 kHz\nreference_frequency = 3 kHz",
        ),
        SOURCE.replace("harmonic_maxwell", "unknown_maxwell"),
        SOURCE.replace("harmonic_maxwell", "electrostatics"),
        SOURCE.replace("harmonic_maxwell", "steady_conduction"),
        SOURCE.replace("harmonic_maxwell", "magnetostatics"),
    ] {
        assert!(!check(&source).is_valid(), "accepted {source}");
    }
}

#[test]
fn homogeneous_pec_rejects_values_and_solid_targets() {
    for source in [
        SOURCE.replace(
            "electrical.perfect_conductor {",
            "electrical.perfect_conductor { value = 0 V",
        ),
        SOURCE.replace("region surface : face", "region surface : solid"),
    ] {
        assert!(!check(&source).is_valid(), "accepted {source}");
    }
}

#[test]
fn authored_harmonic_rejects_magnetic_and_scalar_ground_and_static_pec() {
    for source in [
        SOURCE.replace(
            "electrical.perfect_conductor",
            "magnetic.vector_potential_ground",
        ),
        SOURCE.replace(
            "electrical.perfect_conductor",
            "electrical.potential_reference",
        ),
        SOURCE
            .replace("harmonic_maxwell", "electrostatics")
            .replace("reference_frequency = 2 kHz", ""),
        SOURCE
            .replace("harmonic_maxwell", "magnetostatics")
            .replace("reference_frequency = 2 kHz", ""),
    ] {
        assert!(!check(&source).is_valid(), "accepted mixed intent {source}");
    }
}
