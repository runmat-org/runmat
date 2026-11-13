pub mod support;
use runmat_analysis_core::{ExcitationCaseId, ExcitationCaseSelection, StepObjectRef};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

const SOURCE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region pad : face {
        select result
        expect one_or_more
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m
    }
}
model loaded {
    constraint support { structural.fixed { target = geometry.part.region.pad } }
    load probe { force on geometry.part.region.pad = [0, 1, 0] N }
    load_case service { include load.probe }
    load_case proof { include load.probe }
}
study selected {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.loaded
    analysis implicit { structural static {} }
    analysis unloaded { structural static { load_cases = [] } }
    analysis loaded { structural static { load_cases = [load_case.service] } }
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
fn load_case_selection_survives_realization_without_disabling_supports() {
    let checked = check(SOURCE);
    assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
    let resolved = checked
        .resolve_study("project/main.fea", "selected")
        .unwrap();
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .unwrap();
    let steps = &realized.sequence.steps;
    assert_eq!(
        steps[0].excitation_case_selection,
        ExcitationCaseSelection::All
    );
    assert_eq!(
        steps[1].excitation_case_selection,
        ExcitationCaseSelection::Only(vec![])
    );
    assert_eq!(
        steps[2].excitation_case_selection,
        ExcitationCaseSelection::Only(vec![ExcitationCaseId("load_case.service".into())])
    );
    assert_eq!(realized.model.constraints.len(), 1);
    for step in steps {
        assert!(step.object_is_active(&StepObjectRef::Constraint(
            realized.model.constraints[0].constraint_id.clone()
        )));
    }
}

#[test]
fn load_case_selection_rejects_invalid_or_ambiguous_lists() {
    for replacement in [
        "load_cases = [load_case.absent]",
        "load_cases = [load_case.service, load_case.service]",
        "load_cases = [load.probe]",
        "load_cases = [1]",
        "load_cases = 1",
        "load_cases <= []",
        "load_cases.nested = []",
        "load_cases = []\nload_cases = [load_case.service]",
    ] {
        let checked = check(&SOURCE.replace("load_cases = [load_case.service]", replacement));
        assert!(!checked.is_valid(), "accepted {replacement}");
    }
}
