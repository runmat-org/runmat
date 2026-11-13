pub mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

// witness: FEA-AUTHORING-COMPILER-002/binding_pass @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Wrong region roles, wrong values and composition conflicts between included models
//   are rejected at binding; namespace and shadowing rules are not enumerated.
#[test]
fn model_physics_rejects_wrong_roles_values_and_composition_conflicts() {
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region volume : solid {
        select result
        expect exactly_one
    }
    region wall : face {
        select result
        expect one_or_more
    }
}
model base {
    load outlet {
        fluid.outlet_pressure {
            target = geometry.part.region.wall
            value = 1 Pa
        }
    }
}
model conflict {
    include model.base
    load outlet {
        acoustic.impedance {
            target = geometry.part.region.wall
            impedance = 400 Pa*s/m
        }
    }
}
model invalid {
    load pressure {
        structural.pressure {
            target = geometry.part.region.volume
            value = 1 Pa
        }
    }
    interface contact {
        contact.coulomb {
            primary = geometry.part.region.wall
            secondary = geometry.part.region.wall
            coefficient = 0
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
    assert!(program
        .models
        .values()
        .any(|model| model.model_id == "base"));
    assert!(!program
        .models
        .values()
        .any(|model| model.model_id == "conflict"));
    assert!(!program
        .models
        .values()
        .any(|model| model.model_id == "invalid"));
    assert!(program.diagnostics.iter().any(|diagnostic| {
        diagnostic.message.contains("conflicting boundary")
            || diagnostic.message.contains("incompatible topology")
            || diagnostic.message.contains("distinct")
    }));
}
