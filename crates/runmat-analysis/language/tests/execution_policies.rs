mod support;

use runmat_analysis_core::{
    AcceleratorPreference, BackendPreference, CheckpointPreference, ParallelismLimit,
    PlacementPreference,
};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

const BASE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {
}
"#;

#[test]
fn execution_policy_is_typed_and_inherited_by_a_sweep() {
    let source = format!(
        "{BASE}\n{}",
        r#"
execution scalable {
    placement = profile.engineering
    backend = accelerator
    resources {
        workers = 2..64
        memory_per_worker <= 32 GiB
        accelerators = preferred
    }
    fanout {
        sweep_cases = 16
        frequency_samples = auto
    }
    cooperative {
        operator_partitions = 8
        partitioner = graph
        balance = nonzeros
    }
    resilience {
        retries = 2
        checkpoint = required
    }
}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    execution = execution.scalable
    input load : force {
        default = 1 kN
    }
    analysis static {
        structural static {
        }
    }
}
sweep loads : study.response {
    axis load {
        low = 0.5 kN
        high = 1.5 kN
    }
}
"#
    );
    let program = check(&source);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let policy = program.execution_policies.values().next().unwrap();
    assert_eq!(
        policy.placement,
        PlacementPreference::Cluster {
            profile: Some("engineering".into())
        }
    );
    assert_eq!(policy.backend, BackendPreference::Accelerator);
    assert_eq!(policy.resources.workers.unwrap().minimum, 2);
    assert_eq!(policy.resources.workers.unwrap().maximum, 64);
    assert_eq!(
        policy.resources.maximum_memory_bytes_per_worker,
        Some(32 << 30)
    );
    assert_eq!(
        policy.resources.accelerators,
        AcceleratorPreference::Preferred
    );
    assert_eq!(policy.fanout.sweep_cases, ParallelismLimit::Fixed(16));
    assert_eq!(
        policy.cooperative.operator_partitions,
        ParallelismLimit::Fixed(8)
    );
    assert_eq!(policy.resilience.checkpoint, CheckpointPreference::Required);
    let study_key = program.studies.keys().next().unwrap();
    let sweep_key = program.sweeps.keys().next().unwrap();
    assert_eq!(
        program.execution_bindings.get(study_key),
        program.execution_bindings.get(sweep_key)
    );
    assert_eq!(
        program.execution_bindings.get(study_key).unwrap().policy_id,
        "scalable"
    );

    let resolved_study = program
        .resolve_study("project/main.fea", "response")
        .unwrap();
    assert_eq!(
        resolved_study.execution_policy.as_ref().unwrap().policy_id,
        "scalable"
    );
    let resolved_sweep = program.resolve_sweep("project/main.fea", "loads").unwrap();
    assert_eq!(resolved_sweep.study.study.study_id, "response");
    assert_eq!(
        resolved_sweep
            .study
            .execution_policy
            .as_ref()
            .unwrap()
            .policy_id,
        "scalable"
    );
}

#[test]
fn execution_policy_rejects_contradictions_and_invalid_limits() {
    for policy in [
        "backend = cpu\nresources { accelerators = required }",
        "backend = cpu\nresources { accelerators = preferred }",
        "backend = accelerator",
        "resources { workers = 0..4 }",
        "resources { memory_per_worker = 32 GiB }",
        "fanout { sweep_cases = 0 }",
        "resilience { retries = -1 }",
        "resources { mystery = 1 }",
        "placement = auto\nplacement = local",
        "unknown = auto",
    ] {
        let source = format!("{BASE}\nexecution invalid {{\n{policy}\n}}");
        let program = check(&source);
        assert!(!program.is_valid(), "invalid policy was accepted: {policy}");
        assert!(program.execution_policies.is_empty());
    }
}

fn check(source: &str) -> CheckedFeaProgram {
    let host = TestHost::default()
        .with_module("project/unused.fea", "", 2)
        .with_geometry("project/unused.step", 3)
        .with_function("project/unused.m", "unused", Vec::new(), Vec::new());
    FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}
