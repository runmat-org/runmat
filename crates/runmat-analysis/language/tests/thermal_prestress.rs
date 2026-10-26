#[path = "thermal_prestress/planning.rs"]
mod planning;
#[path = "thermal_prestress/reduced.rs"]
mod reduced;
pub mod support;
use runmat_analysis_core::{
    CoupledPhysicsFamily, PhysicsFamily, ReferenceStatePolicy, StateKind, StateSource,
    StateTransitionMethod, StateUse, TopologyTransitionPolicy,
};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

const SOURCE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m
    }
}
model solid {
    thermal { reference_temperature = 293.15 K }
    thermomechanical { reference_temperature = 300 K }
}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.solid
    analysis heat {
        thermal steady {}
        retain heated
    }
    analysis preload after analysis.heat {
        thermomechanical preload { temperature = analysis.heat.state.heated }
    }
    analysis modes after analysis.preload {
        structural prestressed_modal {
            prestress = analysis.preload.state.geometric_stiffness
            modes = 2
        }
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
fn explicit_domains_and_named_thermal_state_survive_checked_realization() {
    let checked = check(SOURCE);
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked
        .resolve_study("project/main.fea", "response")
        .unwrap();
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    assert_eq!(
        realized
            .model
            .thermal
            .as_ref()
            .unwrap()
            .reference_temperature
            .canonical_value(),
        293.15
    );
    assert_eq!(
        realized
            .model
            .thermo_mechanical
            .as_ref()
            .unwrap()
            .reference_temperature
            .canonical_value(),
        300.0
    );
    let preload = &realized.sequence.steps[1];
    assert_eq!(
        preload.family,
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
    );
    assert_eq!(preload.state_inputs.len(), 1);
    let input = &preload.state_inputs[0];
    assert_eq!(input.required_kind, StateKind::Thermal);
    assert_eq!(input.use_as, StateUse::Prestress);
    assert!(
        matches!(&input.source, StateSource::Upstream { producer_step, state_id } if producer_step.as_str() == "heat" && state_id.0 == "heated")
    );
    assert_eq!(realized.sequence.transitions.len(), 2);
    let thermal_transfer = &realized.sequence.transitions[0];
    assert_eq!(thermal_transfer.producer_step.as_str(), "heat");
    assert_eq!(thermal_transfer.consumer_step.as_str(), "preload");
    assert_eq!(thermal_transfer.method, StateTransitionMethod::Reuse);
    assert_eq!(thermal_transfer.reference, ReferenceStatePolicy::Preserve);
    assert_eq!(
        thermal_transfer.topology,
        TopologyTransitionPolicy::RequireIdentical
    );
    let prestress = &realized.sequence.transitions[1];
    assert_eq!(prestress.producer_step.as_str(), "preload");
    assert_eq!(prestress.consumer_step.as_str(), "modes");
    assert_eq!(prestress.method, StateTransitionMethod::Reuse);
    assert_eq!(prestress.reference, ReferenceStatePolicy::Preserve);
}

#[test]
fn authored_prestress_consumers_reuse_the_producers_geometric_stiffness_state() {
    for (procedure, domain) in [
        ("prestressed_modal", "modes = 2"),
        ("buckling", "modes = 2"),
        ("prestressed_harmonic", "frequencies = [10 Hz, 20 Hz]"),
    ] {
        let source = SOURCE
            .replace(
                "structural prestressed_modal",
                &format!("structural {procedure}"),
            )
            .replace("modes = 2", domain);
        let checked = check(&source);
        assert!(
            checked.is_valid(),
            "{procedure}: {:#?}",
            checked.diagnostics
        );
        let resolved = checked
            .resolve_study("project/main.fea", "response")
            .unwrap();
        let realized = resolved
            .study
            .realize_with_models(&resolved.models, &resolved.geometry, &[])
            .unwrap();
        let consumer = &realized.sequence.steps[2];
        assert_eq!(
            consumer.state_inputs[0].required_kind,
            StateKind::GeometricStiffness
        );
        let transition = &realized.sequence.transitions[1];
        assert_eq!(transition.state_id.0, "geometric_stiffness");
        assert_eq!(transition.method, StateTransitionMethod::Reuse);
        assert_eq!(transition.reference, ReferenceStatePolicy::Preserve);
        assert_eq!(
            transition.topology,
            TopologyTransitionPolicy::RequireIdentical
        );
    }
}

#[test]
fn thermal_domains_reject_missing_duplicate_unknown_nonconstant_and_wrong_units() {
    let baseline = check(SOURCE);
    assert!(baseline.is_valid(), "{:#?}", baseline.diagnostics);
    for replacement in [
        "thermal {}",
        "thermal { reference_temperature = 1 m }",
        "thermal { reference_temperature = 293.15 }",
        "thermal { reference_temperature = -1 K }",
        "thermal { unknown = 1 K\nreference_temperature = 293.15 K }",
        "thermal { reference_temperature = 293.15 K\nreference_temperature = 300 K }",
        "thermal { reference_temperature = 293.15 K } thermal { reference_temperature = 300 K }",
        "thermal { reference_temperature = 200 K + 93.15 K }",
    ] {
        let checked =
            check(&SOURCE.replace("thermal { reference_temperature = 293.15 K }", replacement));
        assert!(
            !checked.is_valid(),
            "accepted invalid domain: {replacement}"
        );
    }
    for replacement in [
        "thermomechanical {}",
        "thermomechanical { reference_temperature = 1 Pa }",
        "thermomechanical { reference_temperature = 300 K\nunknown = 0 }",
    ] {
        assert!(
            !check(&SOURCE.replace(
                "thermomechanical { reference_temperature = 300 K }",
                replacement
            ))
            .is_valid(),
            "accepted invalid domain: {replacement}"
        );
    }
}

#[test]
fn thermal_preload_rejects_missing_wrong_kind_state_and_ambiguous_members() {
    let baseline = check(SOURCE);
    assert!(baseline.is_valid(), "{:#?}", baseline.diagnostics);
    for replacement in [
        "",
        "temperature = analysis.heat.state.absent",
        "temperature = analysis.absent.state.heated",
        "temperature = analysis.heat.state.heated\ntemperature = analysis.heat.state.heated",
        "temperature = analysis.heat.state.heated\nunknown = 1",
        "temperature = 293.15 K",
    ] {
        assert!(
            !check(&SOURCE.replace("temperature = analysis.heat.state.heated", replacement))
                .is_valid(),
            "accepted invalid preload: {replacement}"
        );
    }
    assert!(!check(&SOURCE.replace("thermal steady {}", "structural static {}")).is_valid());
    assert!(!check(&SOURCE.replace("after analysis.heat", "after analysis.modes")).is_valid());
}
