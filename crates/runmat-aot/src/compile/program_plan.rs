use std::collections::BTreeSet;

use runmat_mir::analysis::{ReachabilityNodeKind, ReachabilityReport};

use crate::{
    archive::{RuntimeArchive, RuntimeArchiveCapabilities},
    AotError, AotResult,
};

use super::CompilationPolicy;

pub const PROGRAM_LINK_PLAN_SCHEMA_VERSION: u16 = 4;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeFamilyRetention {
    pub module: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramLinkPlan {
    pub schema_version: u16,
    pub policy: CompilationPolicy,
    pub program_graph_digest: String,
    pub program_source_digest: String,
    pub target_triple: String,
    pub runtime_archive_digest: String,
    pub catalog_fingerprint: String,
    pub runtime_capabilities: RuntimeArchiveCapabilities,
    pub reachability: ReachabilityReport,
    /// Canonical MIR identities; report node strings are presentation only.
    pub retained_function_ids: BTreeSet<runmat_types::ProgramFunctionId>,
    pub retained_builtin_bindings: Vec<runmat_native_codegen::aot::AotBuiltinBinding>,
    pub interop: runmat_types::InteropManifest,
    pub retained_runtime_families: Vec<RuntimeFamilyRetention>,
    pub omitted_runtime_families: Vec<String>,
}

impl ProgramLinkPlan {
    pub fn retained_functions(&self) -> BTreeSet<runmat_types::ProgramFunctionId> {
        self.retained_function_ids.clone()
    }
}

pub fn build_program_link_plan(
    unit: &runmat_core::ExecutableUnit,
    runtime: &RuntimeArchive,
    policy: CompilationPolicy,
) -> AotResult<ProgramLinkPlan> {
    build_program_link_plan_with_interop(
        unit,
        runtime,
        policy,
        runmat_types::InteropManifest::empty(),
    )
}

pub fn build_program_link_plan_with_interop(
    unit: &runmat_core::ExecutableUnit,
    runtime: &RuntimeArchive,
    policy: CompilationPolicy,
    interop: runmat_types::InteropManifest,
) -> AotResult<ProgramLinkPlan> {
    interop.validate().map_err(|error| {
        AotError::contract(
            "aot.compile.interop",
            format!("{}: {}", error.path, error.message),
        )
    })?;
    runtime.manifest.validate()?;
    policy.validate(&runtime.manifest.capabilities)?;
    runtime
        .manifest
        .validate_environment(&unit.revision().program_revision.environment())?;
    let (reachability, retained_functions) = unit.reachability_analysis();
    let retained_function_ids = retained_functions
        .into_iter()
        .map(|function| {
            u32::try_from(function.0)
                .map(runmat_types::ProgramFunctionId)
                .map_err(|_| {
                    AotError::contract(
                        "aot.compile.function_identity",
                        "reachable function identity exceeds portable bounds",
                    )
                })
        })
        .collect::<AotResult<BTreeSet<_>>>()?;
    if policy == CompilationPolicy::ClosedWorld && reachability.has_unknown_edges {
        return Err(AotError::contract(
            "aot.compile.reachability_unknown",
            "closed-world compilation cannot retain an unbounded runtime call target",
        ));
    }
    let retained_builtin_bindings = retained_builtin_bindings(policy, &reachability)?;
    let (retained_runtime_families, omitted_runtime_families) = runtime_families(
        policy,
        &runtime.manifest.capabilities,
        &reachability,
        &retained_builtin_bindings,
    );
    Ok(ProgramLinkPlan {
        schema_version: PROGRAM_LINK_PLAN_SCHEMA_VERSION,
        policy,
        program_graph_digest: unit.revision().program_revision.graph_digest().to_string(),
        program_source_digest: unit.revision().program_revision.source_digest().to_string(),
        target_triple: runtime.manifest.target_triple.clone(),
        runtime_archive_digest: runtime.manifest.archive_digest.to_string(),
        catalog_fingerprint: runtime.manifest.catalog_fingerprint.to_string(),
        runtime_capabilities: runtime.manifest.capabilities.clone(),
        reachability,
        retained_function_ids,
        retained_builtin_bindings,
        interop,
        retained_runtime_families,
        omitted_runtime_families,
    })
}

fn runtime_families(
    policy: CompilationPolicy,
    capabilities: &RuntimeArchiveCapabilities,
    reachability: &ReachabilityReport,
    builtin_bindings: &[runmat_native_codegen::aot::AotBuiltinBinding],
) -> (Vec<RuntimeFamilyRetention>, Vec<String>) {
    let mut retained = vec![
        family("runmat-value", "program value ABI"),
        family("runmat-gc", "program allocation and roots"),
        family("runmat-runtime", "builtin and native-call runtime"),
        family(
            "runmat-hir::runtime-schema",
            "runtime operation identities embedded in verified Native IR",
        ),
        family(
            "runmat-mir::runtime-schema",
            "runtime operation payloads embedded in verified Native IR",
        ),
        family(
            "runmat-native-codegen::runtime-schema",
            "Native IR decoding and verification",
        ),
        family("runmat-native-executor", "native execution host"),
    ];
    let mut omitted = vec![
        "runmat-core".into(),
        "runmat-hir::frontend".into(),
        "runmat-mir::frontend".into(),
        "runmat-native-codegen::compiler".into(),
        "runmat-parser".into(),
    ];
    if reachability
        .nodes
        .iter()
        .any(|node| node.builtin_id().is_some())
    {
        let reason = if policy == CompilationPolicy::ClosedWorld {
            format!(
                "{} exact reachable runtime binding(s)",
                builtin_bindings.len()
            )
        } else {
            "runtime-discovered builtin symbols".into()
        };
        retained.push(RuntimeFamilyRetention {
            module: "runmat-builtins".into(),
            reason,
        });
    }
    if reachability.nodes.iter().any(|node| {
        matches!(
            node.kind,
            ReachabilityNodeKind::Provider | ReachabilityNodeKind::Extension
        )
    }) {
        retained.push(family(
            "runmat-accelerate-api",
            "reachable provider or extension boundary",
        ));
    }
    match policy {
        CompilationPolicy::NativeSpecialized | CompilationPolicy::DynamicRuntime => {}
        CompilationPolicy::ClosedWorld => omitted.extend(["runmat-vm".into(), "runmat-jit".into()]),
        CompilationPolicy::Portable => {}
    }
    let plot_is_reachable = reachability.nodes.iter().any(|node| {
        node.kind == ReachabilityNodeKind::ArtifactDependency && node.symbol == "runmat-plot-core"
    });
    if capabilities.plot_core && (policy != CompilationPolicy::ClosedWorld || plot_is_reachable) {
        retained.push(family(
            "runmat-plot-core",
            "runtime archive plot capability",
        ));
    } else {
        omitted.push("runmat-plot-core".into());
    }
    retained.sort_by(|left, right| left.module.cmp(&right.module));
    omitted.sort();
    omitted.dedup();
    (retained, omitted)
}

fn retained_builtin_bindings(
    policy: CompilationPolicy,
    reachability: &ReachabilityReport,
) -> AotResult<Vec<runmat_native_codegen::aot::AotBuiltinBinding>> {
    if policy != CompilationPolicy::ClosedWorld {
        return Ok(Vec::new());
    }
    let mut bindings = Vec::new();
    for (node, builtin) in reachability
        .nodes
        .iter()
        .filter_map(|node| node.builtin_id().map(|builtin| (node, builtin)))
    {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(&builtin.0).ok_or_else(|| {
            AotError::contract(
                "aot.compile.closed_world_builtin",
                format!(
                    "closed-world compilation requires builtin `{}` to have a canonical catalog runtime binding",
                    node.symbol
                ),
            )
        })?;
        if !matches!(
            entry.link.reachability,
            runmat_builtins::BuiltinReachability::Always
        ) {
            return Err(AotError::contract(
                "aot.compile.closed_world_builtin_dynamic",
                format!(
                    "closed-world compilation cannot prove the complete runtime target set for builtin `{}`",
                    node.symbol
                ),
            ));
        }
        if entry.bindings.is_empty() {
            return Err(AotError::contract(
                "aot.compile.closed_world_builtin",
                format!(
                    "closed-world builtin `{}` has no runtime binding declaration",
                    node.symbol
                ),
            ));
        }
        if entry.bindings.iter().any(|binding| {
            binding.availability == runmat_builtins::BuiltinBindingAvailability::TargetConditional
        }) {
            return Err(AotError::contract(
                "aot.compile.closed_world_builtin_target",
                format!(
                    "closed-world compilation cannot select target-conditional runtime bindings for builtin `{}`",
                    node.symbol
                ),
            ));
        }
        bindings.extend(entry.bindings.iter().map(|binding| {
            let identity = entry.binding_identity(binding);
            runmat_native_codegen::aot::AotBuiltinBinding {
                name: identity.builtin.name.to_string(),
                variant: identity.variant.to_string(),
                native_symbol: runmat_builtins::native_binding_symbol(
                    identity.builtin.name,
                    identity.variant,
                ),
            }
        }));
    }
    bindings.sort_by(|left, right| (&left.name, &left.variant).cmp(&(&right.name, &right.variant)));
    bindings.dedup_by(|left, right| left.name == right.name && left.variant == right.variant);
    Ok(bindings)
}

fn family(module: &str, reason: &str) -> RuntimeFamilyRetention {
    RuntimeFamilyRetention {
        module: module.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_mir::analysis::{
        ReachabilityCertainty, ReachabilityNode, ReachabilityNodeKind, ReachabilityReport,
        REACHABILITY_SCHEMA_VERSION,
    };

    use super::{
        retained_builtin_bindings, CompilationPolicy, ProgramLinkPlan,
        PROGRAM_LINK_PLAN_SCHEMA_VERSION,
    };

    fn builtin_report(name: &str) -> ReachabilityReport {
        ReachabilityReport {
            schema_version: REACHABILITY_SCHEMA_VERSION,
            nodes: vec![ReachabilityNode {
                id: format!("builtin:{}", name.to_ascii_lowercase()),
                kind: ReachabilityNodeKind::Builtin(runmat_types::BuiltinId(name.into())),
                module: "runmat-builtins".into(),
                symbol: name.into(),
                certainty: ReachabilityCertainty::Definite,
            }],
            edges: Vec::new(),
            has_unknown_edges: false,
        }
    }

    #[test]
    fn closed_world_maps_catalog_bindings_to_stable_native_symbols() {
        for name in [
            "abs", "angle", "conj", "double", "exp", "expm1", "imag", "int8", "int16", "int32",
            "flintmax", "int64", "intmax", "intmin", "log", "log10", "log1p", "log2", "plus",
            "real", "realmax", "realmin", "realsqrt", "sign", "single", "sqrt", "uint8", "uint16",
            "uint32", "uint64",
        ] {
            let bindings =
                retained_builtin_bindings(CompilationPolicy::ClosedWorld, &builtin_report(name))
                    .expect("catalog-backed closed-world binding");
            assert_eq!(bindings.len(), 1);
            assert_eq!(bindings[0].name, name);
            assert_eq!(bindings[0].variant, "default");
            assert_eq!(
                bindings[0].native_symbol,
                runmat_builtins::native_binding_symbol(name, "default")
            );
        }
    }

    #[test]
    fn closed_world_rejects_unregistered_and_unbounded_builtin_authorities() {
        let unregistered = retained_builtin_bindings(
            CompilationPolicy::ClosedWorld,
            &builtin_report("unregistered_aot_fixture"),
        )
        .expect_err("an undeclared builtin must not imply exact retention");
        assert!(matches!(
            unregistered,
            crate::AotError::Contract {
                code: "aot.compile.closed_world_builtin",
                ..
            }
        ));

        let dynamic =
            retained_builtin_bindings(CompilationPolicy::ClosedWorld, &builtin_report("feval"))
                .expect_err("dynamic builtin target set must be proven");
        assert!(matches!(
            dynamic,
            crate::AotError::Contract {
                code: "aot.compile.closed_world_builtin_dynamic",
                ..
            }
        ));

        let legacy_untyped = serde_json::json!({
            "id": "builtin:abs",
            "kind": "builtin",
            "module": "runmat-builtins",
            "symbol": "abs",
            "certainty": "definite"
        });
        assert!(serde_json::from_value::<ReachabilityNode>(legacy_untyped).is_err());
    }

    #[test]
    fn link_plan_retention_uses_typed_wire_field_not_report_spelling() {
        let retained = BTreeSet::from([runmat_types::ProgramFunctionId(7)]);
        let plan = ProgramLinkPlan {
            schema_version: PROGRAM_LINK_PLAN_SCHEMA_VERSION,
            policy: CompilationPolicy::NativeSpecialized,
            program_graph_digest: "graph".into(),
            program_source_digest: "sources".into(),
            target_triple: "test-target".into(),
            runtime_archive_digest: "archive".into(),
            catalog_fingerprint: "catalog".into(),
            runtime_capabilities: super::RuntimeArchiveCapabilities::standalone_host(),
            reachability: ReachabilityReport {
                schema_version: REACHABILITY_SCHEMA_VERSION,
                nodes: vec![ReachabilityNode {
                    id: "function:not-an-ordinal".into(),
                    kind: ReachabilityNodeKind::Function,
                    module: "test".into(),
                    symbol: "presentation".into(),
                    certainty: ReachabilityCertainty::Definite,
                }],
                edges: Vec::new(),
                has_unknown_edges: false,
            },
            retained_function_ids: retained.clone(),
            retained_builtin_bindings: Vec::new(),
            interop: runmat_types::InteropManifest::empty(),
            retained_runtime_families: Vec::new(),
            omitted_runtime_families: Vec::new(),
        };
        assert_eq!(plan.retained_functions(), retained);
        let mut wire = serde_json::to_value(&plan).expect("serialize typed plan");
        let decoded: ProgramLinkPlan =
            serde_json::from_value(wire.clone()).expect("decode current plan");
        assert_eq!(decoded.retained_functions(), retained);
        wire.as_object_mut()
            .expect("plan object")
            .remove("retained_function_ids");
        assert!(serde_json::from_value::<ProgramLinkPlan>(wire).is_err());
    }
}
