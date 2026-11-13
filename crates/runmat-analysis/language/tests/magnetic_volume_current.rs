pub mod support;

use runmat_analysis_core::{
    AuthoredExcitationKind, ExcitationDistribution, PhysicalQuantityKind, SignalShape,
};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(source: &str) -> CheckedFeaProgram {
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

const SOURCE: &str = r#"
geometry conductor {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region bulk : solid { select result; expect one_or_more }
    region surface : face { select result; expect one_or_more }
}
model coil {
    load volume_current {
        magnetic_volume_current_density on geometry.conductor.region.bulk = [1, -2, 3] A/m^2
    }
    signal current : electric_current_density over constant { value = [0, 1, 0] A/m^2; frame = global }
    excitation whole_current {
        magnetic_volume_current_density { target = whole_model; signal = signal.current }
    }
    load_case drive { include load.volume_current; include excitation.whole_current }
}
"#;

#[test]
fn magnetic_volume_current_lowers_typed_vector_volume_source() {
    let program = check(SOURCE);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(model.excitations.len(), 2);
    for action in &model.excitations {
        assert_eq!(
            action.kind,
            AuthoredExcitationKind::MagneticVolumeCurrentDensity
        );
        assert_eq!(
            action.distribution,
            ExcitationDistribution::IntensityPerVolume
        );
        assert!(action.direction.is_none());
        assert!(!action.projection.preserve_resultant);
        assert!(!action.projection.preserve_moment);
    }
    let signal = model
        .signals
        .iter()
        .find(|signal| signal.signal_id().0 == "load.volume_current.value")
        .unwrap();
    assert_eq!(
        signal.quantity(),
        PhysicalQuantityKind::ElectricCurrentDensity
    );
    assert_eq!(signal.shape(), &SignalShape::Vector3);
}

#[test]
fn magnetic_volume_current_rejects_surface_scalar_direction_and_total_current() {
    for source in [
        SOURCE.replace(
            "on geometry.conductor.region.bulk",
            "on geometry.conductor.region.surface",
        ),
        SOURCE.replace("[1, -2, 3] A/m^2", "1 A/m^2"),
        SOURCE.replace("[1, -2, 3] A/m^2", "[1, -2, 3] A"),
        SOURCE.replace(
            "signal = signal.current",
            "signal = signal.current; direction = global([1, 0, 0])",
        ),
        SOURCE.replace(
            "signal = signal.current",
            "signal = signal.current; kinematics = follower",
        ),
        SOURCE.replace("magnetic_volume_current_density on", "current_density on"),
    ] {
        assert!(
            !check(&source).is_valid(),
            "invalid source admitted: {source}"
        );
    }
}
