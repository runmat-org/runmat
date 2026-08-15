use std::collections::BTreeSet;

use runmat_mir::analysis::{ReachabilityNodeKind, ReachabilityReport};

use crate::{
    archive::{RuntimeArchive, RuntimeArchiveCapabilities},
    AotError, AotResult,
};

use super::CompilationPolicy;

pub const PROGRAM_LINK_PLAN_SCHEMA_VERSION: u16 = 1;

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
    pub retained_runtime_families: Vec<RuntimeFamilyRetention>,
    pub omitted_runtime_families: Vec<String>,
}

impl ProgramLinkPlan {
    pub fn retained_functions(&self) -> BTreeSet<runmat_types::ProgramFunctionId> {
        self.reachability
            .retained_function_ids()
            .filter_map(|function| u32::try_from(function).ok())
            .map(runmat_types::ProgramFunctionId)
            .collect()
    }
}

pub fn build_program_link_plan(
    unit: &runmat_core::ExecutableUnit,
    runtime: &RuntimeArchive,
    policy: CompilationPolicy,
) -> AotResult<ProgramLinkPlan> {
    runtime.manifest.validate()?;
    policy.validate(&runtime.manifest.capabilities)?;
    runtime
        .manifest
        .validate_environment(&unit.revision().program_revision.environment())?;
    let reachability = unit.reachability_report();
    if policy == CompilationPolicy::ClosedWorld && reachability.has_unknown_edges {
        return Err(AotError::contract(
            "aot.compile.reachability_unknown",
            "closed-world compilation cannot retain an unbounded runtime call target",
        ));
    }
    let (retained_runtime_families, omitted_runtime_families) =
        runtime_families(policy, &runtime.manifest.capabilities, &reachability);
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
        retained_runtime_families,
        omitted_runtime_families,
    })
}

fn runtime_families(
    policy: CompilationPolicy,
    capabilities: &RuntimeArchiveCapabilities,
    reachability: &ReachabilityReport,
) -> (Vec<RuntimeFamilyRetention>, Vec<String>) {
    let mut retained = vec![
        family("runmat-value", "program value ABI"),
        family("runmat-gc", "program allocation and roots"),
        family("runmat-runtime", "builtin and native-call runtime"),
    ];
    let mut omitted = vec![
        "runmat-core".into(),
        "runmat-hir".into(),
        "runmat-parser".into(),
    ];
    if reachability
        .nodes
        .iter()
        .any(|node| node.kind == ReachabilityNodeKind::Builtin)
    {
        retained.push(family("runmat-builtins", "reachable builtin symbols"));
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
        CompilationPolicy::NativeSpecialized | CompilationPolicy::DynamicRuntime => {
            retained.extend([
                family("runmat-vm", "runtime function registry and fallback calls"),
                family("runmat-jit", "generic native executor"),
                family("runmat-native-codegen", "verified Native IR schema"),
            ]);
        }
        CompilationPolicy::ClosedWorld => {
            omitted.extend([
                "runmat-vm".into(),
                "runmat-jit".into(),
                "runmat-native-codegen".into(),
            ]);
        }
        CompilationPolicy::Portable => {}
    }
    if capabilities.plot_core {
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
