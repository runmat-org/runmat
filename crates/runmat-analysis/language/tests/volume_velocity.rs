pub mod support;
use runmat_analysis_core::{AuthoredExcitationKind, ExcitationDistribution, PhysicalQuantityKind};
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
    signal flow : volume_flow_rate over constant { value = -0.2 m^3/s }
    excitation speaker {
        volume_velocity {
            target = geometry.part.region.wall
            signal = signal.flow
        }
    }
    load_case service { include excitation.speaker }
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
fn volume_velocity_binds_generic_total_flow_and_resultant_policy() {
    let program = check(SOURCE);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(
        model.excitations[0].kind,
        AuthoredExcitationKind::VolumeVelocity
    );
    assert_eq!(
        model.signals[0].quantity(),
        PhysicalQuantityKind::VolumeFlowRate
    );
    assert_eq!(
        model.excitations[0].distribution,
        ExcitationDistribution::PreserveResultant
    );
    assert!(model.excitations[0].projection.preserve_resultant);
    assert!(!model.excitations[0].projection.preserve_moment);
}

#[test]
fn volume_velocity_rejects_velocity_mass_flow_direction_follower_and_intensity() {
    for source in [
        SOURCE.replace(
            "volume_flow_rate over constant { value = -0.2 m^3/s }",
            "velocity over constant { value = -0.2 m/s }",
        ),
        SOURCE.replace(
            "volume_flow_rate over constant { value = -0.2 m^3/s }",
            "mass_flow_rate over constant { value = -0.2 kg/s }",
        ),
        SOURCE.replace(
            "signal = signal.flow",
            "signal = signal.flow\n direction = global([1, 0, 0])",
        ),
        SOURCE.replace(
            "signal = signal.flow",
            "signal = signal.flow\n kinematics = follower",
        ),
        SOURCE.replace(
            "signal = signal.flow",
            "signal = signal.flow\n distribution = intensity_per_area",
        ),
    ] {
        let program = check(&source);
        assert!(!program.is_valid(), "invalid volume velocity was accepted");
    }
}
