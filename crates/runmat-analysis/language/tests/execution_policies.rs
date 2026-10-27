mod support;

use runmat_analysis_core::{
    AcceleratorPreference, BackendPreference, CheckpointPreference, ParallelismLimit,
    PlacementPreference,
};
use runmat_analysis_language::*;
use runmat_types::{CapabilityRequirement, CapabilitySet};
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

// witness: FEA-AUTHORING-LANGUAGE-002/execution_precedence @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: A typed execution policy is inherited by a sweep; CLI and project default precedence
//   is not covered.
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
        cpu_threads_per_worker = 4
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
sweep loads over study.response {
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
    assert_eq!(policy.resources.cpu_threads_per_worker, Some(4));
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

// witness: FEA-AUTHORING-LANGUAGE-002/execution_precedence @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Contradictory or out-of-range policy members are rejected.
#[test]
fn execution_policy_rejects_contradictions_and_invalid_limits() {
    for policy in [
        "backend = cpu\nresources { accelerators = required }",
        "backend = cpu\nresources { accelerators = preferred }",
        "backend = accelerator",
        "resources { workers = 0..4 }",
        "resources { cpu_threads_per_worker = 0 }",
        "resources { cpu_threads_per_worker = 1025 }",
        "resources { cpu_threads_per_worker = 2..4 }",
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

// witness: FEA-AUTHORING-LANGUAGE-002/execution_precedence @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: memory_per_worker is a positive whole byte count no larger than the shared FEA
//   wire-integer bound 2^53 - 1; 2^53 and above, zero and fractional bytes are rejected.
#[test]
fn memory_per_worker_shares_the_wire_integer_bound() {
    let policy_bytes = |bound: &str| {
        let source = format!(
            "{BASE}\nexecution bounded {{\nresources {{ memory_per_worker <= {bound} }}\n}}"
        );
        let program = check(&source);
        program
            .is_valid()
            .then(|| {
                program
                    .execution_policies
                    .values()
                    .next()
                    .unwrap()
                    .resources
                    .maximum_memory_bytes_per_worker
            })
            .flatten()
    };
    let max = runmat_types::wire_integer::MAX_WIRE_INTEGER;
    assert_eq!(policy_bytes("9007199254740991 B"), Some(max));
    assert_eq!(policy_bytes("1 B"), Some(1));
    for rejected in ["9007199254740992 B", "8388608 GiB", "0 B", "1.5 B", "-1 B"] {
        assert_eq!(policy_bytes(rejected), None, "accepted {rejected}");
    }
}

#[test]
fn named_profiles_are_resolved_and_must_admit_the_authored_placement() {
    let source = format!(
        "{BASE}\nexecution selected {{\nplacement = profile.engineering\nbackend = accelerator\nresources {{ accelerators = required }}\n}}"
    );
    let unavailable = TestHost::default();
    let program = compile_with(&unavailable, &source);
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "fea.test.execution_profile_missing"));

    let local = host_with_profile(
        "engineering",
        ExecutionPlacement::Inline,
        CapabilitySet::default(),
    );
    let program = compile_with(&local, &source);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("must admit remote distributed execution")));

    let cpu_only = host_with_profile(
        "engineering",
        ExecutionPlacement::Remote,
        CapabilitySet([CapabilityRequirement::DistributedRuntime].into()),
    );
    let program = compile_with(&cpu_only, &source);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("does not admit accelerator execution")));
}

fn check(source: &str) -> CheckedFeaProgram {
    let host = TestHost::default()
        .with_module("project/unused.fea", "", 2)
        .with_geometry("project/unused.step", 3)
        .with_function("project/unused.m", "unused", Vec::new(), Vec::new());
    let mut host = host;
    host.profiles = host_with_profile(
        "engineering",
        ExecutionPlacement::Remote,
        CapabilitySet(
            [
                CapabilityRequirement::DistributedRuntime,
                CapabilityRequirement::Accelerator,
            ]
            .into(),
        ),
    )
    .profiles;
    compile_with(&host, source)
}

fn host_with_profile(
    name: &str,
    placement: ExecutionPlacement,
    capabilities: CapabilitySet,
) -> TestHost {
    let mut host = TestHost::default();
    host.profiles.insert(
        name.to_string(),
        ExecutionProfileFacts {
            profile: ExecutionProfileIdentity::new(name).unwrap(),
            placement,
            capabilities,
        },
    );
    host
}

fn compile_with(host: &TestHost, source: &str) -> CheckedFeaProgram {
    FeaCompiler::new(host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}
