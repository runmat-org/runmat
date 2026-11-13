pub mod support;

use runmat_analysis_core::{
    AuthoredExcitationKind, AuthoredExcitationTarget, ExcitationDistribution, PhysicalQuantityKind,
    PhysicalSignal, SignalRepresentation, SignalShape, SignalValue,
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
geometry dielectric {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region bulk : solid { select result; expect one_or_more }
    region surface : face { select result; expect one_or_more }
}
model charged {
    load bulk_charge {
        electric_charge_density on geometry.dielectric.region.bulk = 2 C/m^3
    }
    signal uniform_charge : electric_charge_density over constant { value = -3 C/m^3 }
    excitation whole_charge {
        electric_charge_density { target = whole_model; signal = signal.uniform_charge }
    }
    load_case charge {
        include load.bulk_charge
        include excitation.whole_charge
    }
}
"#;

#[test]
fn volume_charge_density_lowers_quantity_target_and_volume_distribution() {
    let program = check(SOURCE);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let model = program.models.values().next().unwrap();
    assert_eq!(model.excitations.len(), 2);
    for action in &model.excitations {
        assert_eq!(action.kind, AuthoredExcitationKind::ElectricChargeDensity);
        assert_eq!(
            action.distribution,
            ExcitationDistribution::IntensityPerVolume
        );
        assert!(action.direction.is_none());
        assert!(!action.projection.preserve_moment);
    }
    assert!(model
        .excitations
        .iter()
        .any(|action| matches!(action.target, AuthoredExcitationTarget::WholeModel)));
    let signal = model
        .signals
        .iter()
        .find(|signal| signal.signal_id().0 == "load.bulk_charge.value")
        .unwrap();
    assert_eq!(
        signal.quantity(),
        PhysicalQuantityKind::ElectricChargeDensity
    );
    assert_eq!(signal.shape(), &SignalShape::Scalar);
    let PhysicalSignal::ElectricChargeDensity(signal) = signal else {
        panic!("typed volume charge signal")
    };
    let SignalRepresentation::Constant(SignalValue::Scalar(value)) = &signal.representation else {
        panic!("constant scalar density")
    };
    assert_eq!(value.canonical_value(), 2.0);
}

#[test]
fn volume_charge_density_rejects_surface_total_charge_and_vector_inputs() {
    for source in [
        SOURCE.replace(
            "on geometry.dielectric.region.bulk",
            "on geometry.dielectric.region.surface",
        ),
        SOURCE.replace("= 2 C/m^3", "= 2 C"),
        SOURCE.replace("= 2 C/m^3", "= [1, 2, 3] C/m^3"),
        SOURCE.replace(
            "signal = signal.uniform_charge",
            "signal = signal.uniform_charge; direction = global([1, 0, 0])",
        ),
        SOURCE.replace(
            "signal = signal.uniform_charge",
            "signal = signal.uniform_charge; kinematics = follower",
        ),
    ] {
        let program = check(&source);
        assert!(
            !program.is_valid(),
            "invalid charge-density input admitted: {source}"
        );
    }
}
