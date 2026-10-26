use std::collections::BTreeSet;

use runmat_mir::analysis::{ReachabilityNodeKind, ReachabilityReport};

use crate::{
    archive::{RuntimeArchive, RuntimeArchiveCapabilities},
    AotError, AotResult,
};

use super::{CompilationPolicy, RuntimeComposition};

pub const PROGRAM_LINK_PLAN_SCHEMA_VERSION: u16 = 7;

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
    pub product: runmat_execution::host::NativeProductMode,
    pub policy: CompilationPolicy,
    pub program_graph_digest: String,
    pub program_source_digest: String,
    pub target_triple: String,
    pub runtime_archive_digest: String,
    pub catalog_fingerprint: String,
    pub runtime_capabilities: RuntimeArchiveCapabilities,
    pub reachability: ReachabilityReport,
    /// Canonical MIR identities; report node strings are presentation only.
    pub retained_function_ids: BTreeSet<runmat_types::FunctionId>,
    pub runtime_composition: RuntimeComposition,
    pub interop: runmat_types::InteropManifest,
    pub retained_runtime_families: Vec<RuntimeFamilyRetention>,
    pub omitted_runtime_families: Vec<String>,
}

impl ProgramLinkPlan {
    pub fn retained_functions(&self) -> BTreeSet<runmat_types::FunctionId> {
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
    let retained_function_ids = retained_functions;
    let runtime_composition =
        RuntimeComposition::select(policy, &runtime.manifest.capabilities, &reachability)?;
    let (retained_runtime_families, omitted_runtime_families) = runtime_families(
        &runtime_composition,
        &runtime.manifest.capabilities,
        &reachability,
    );
    Ok(ProgramLinkPlan {
        schema_version: PROGRAM_LINK_PLAN_SCHEMA_VERSION,
        product: runtime.manifest.product.clone(),
        policy,
        program_graph_digest: unit.revision().program_revision.graph_digest().to_string(),
        program_source_digest: unit.revision().program_revision.source_digest().to_string(),
        target_triple: runtime.manifest.target_triple.clone(),
        runtime_archive_digest: runtime.manifest.archive_digest.to_string(),
        catalog_fingerprint: runtime.manifest.catalog_fingerprint.to_string(),
        runtime_capabilities: runtime.manifest.capabilities.clone(),
        reachability,
        retained_function_ids,
        runtime_composition,
        interop,
        retained_runtime_families,
        omitted_runtime_families,
    })
}

fn runtime_families(
    composition: &RuntimeComposition,
    capabilities: &RuntimeArchiveCapabilities,
    reachability: &ReachabilityReport,
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
        let reason = if composition.is_selected() {
            format!(
                "{} exact reachable runtime binding(s)",
                composition.bindings().len()
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
    if composition.is_selected() {
        omitted.extend(["runmat-vm".into(), "runmat-jit".into()]);
    }
    let plot_is_reachable = reachability.nodes.iter().any(|node| {
        node.kind == ReachabilityNodeKind::ArtifactDependency && node.symbol == "runmat-plot-core"
    });
    if capabilities.plot_core && (!composition.is_selected() || plot_is_reachable) {
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

    use super::{CompilationPolicy, ProgramLinkPlan, PROGRAM_LINK_PLAN_SCHEMA_VERSION};

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
            let bindings = super::super::composition::closed_world_bindings(&builtin_report(name))
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
        let unregistered = super::super::composition::closed_world_bindings(&builtin_report(
            "unregistered_aot_fixture",
        ))
        .expect_err("an undeclared builtin must not imply exact retention");
        assert!(matches!(
            unregistered,
            crate::AotError::Contract {
                code: "aot.compile.closed_world_builtin",
                ..
            }
        ));

        let dynamic = super::super::composition::closed_world_bindings(&builtin_report("feval"))
            .expect_err("dynamic builtin target set must be proven");
        assert!(matches!(
            dynamic,
            crate::AotError::Contract {
                code: "aot.compile.closed_world_builtin_dynamic",
                ..
            }
        ));

        use super::super::{DynamicCompositionReason, RuntimeComposition};
        let mut capabilities = super::RuntimeArchiveCapabilities::standalone_host();
        let selected = RuntimeComposition::select(
            CompilationPolicy::NativeSpecialized,
            &capabilities,
            &builtin_report("abs"),
        )
        .unwrap();
        assert!(selected.is_selected());
        assert_eq!(selected.bindings()[0].name, "abs");
        let dynamic = RuntimeComposition::select(
            CompilationPolicy::NativeSpecialized,
            &capabilities,
            &builtin_report("feval"),
        )
        .unwrap();
        assert!(matches!(
            dynamic,
            RuntimeComposition::Dynamic {
                reason: DynamicCompositionReason::DynamicBuiltin { .. }
            }
        ));
        let mut unknown = builtin_report("abs");
        unknown.has_unknown_edges = true;
        assert!(matches!(
            RuntimeComposition::select(
                CompilationPolicy::NativeSpecialized,
                &capabilities,
                &unknown
            )
            .unwrap(),
            RuntimeComposition::Dynamic {
                reason: DynamicCompositionReason::UnknownTargets
            }
        ));
        assert!(RuntimeComposition::select(
            CompilationPolicy::ClosedWorld,
            &capabilities,
            &unknown,
        )
        .is_err());
        capabilities.closed_world_linking = false;
        assert!(matches!(
            RuntimeComposition::select(
                CompilationPolicy::NativeSpecialized,
                &capabilities,
                &builtin_report("abs")
            )
            .unwrap(),
            RuntimeComposition::Dynamic {
                reason: DynamicCompositionReason::ArchiveCapability
            }
        ));
        capabilities.dynamic_source_loading = true;
        assert!(matches!(
            RuntimeComposition::select(
                CompilationPolicy::DynamicRuntime,
                &capabilities,
                &builtin_report("abs")
            )
            .unwrap(),
            RuntimeComposition::Dynamic {
                reason: DynamicCompositionReason::Requested
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
        let retained = BTreeSet::from([runmat_types::FunctionId(7)]);
        let plan = ProgramLinkPlan {
            schema_version: PROGRAM_LINK_PLAN_SCHEMA_VERSION,
            product: runmat_execution::host::NativeProductMode::HostUnselected,
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
            runtime_composition: super::RuntimeComposition::Selected {
                bindings: Vec::new(),
            },
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
