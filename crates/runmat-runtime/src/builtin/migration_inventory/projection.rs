use std::collections::{BTreeMap, BTreeSet};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprBuilder,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use runmat_builtins::{
    builtin_catalog_entries, builtin_functions, builtin_implementation_provenance,
    BuiltinCatalogIdentity,
};

use super::schema::{
    FusionShapeRecord, FusionSpecRecord, FusionTemplateRecord, GpuSpecRecord, ProviderHookRecord,
    SpecOwnerRecord,
};

pub(super) struct SpecOwnershipClassifier {
    exact_builtin_names: BTreeSet<&'static str>,
    identities_by_builtin_path: BTreeMap<&'static str, BTreeSet<&'static str>>,
}

impl SpecOwnershipClassifier {
    pub(super) fn compiled() -> Self {
        let mut identities_by_builtin_path = BTreeMap::<_, BTreeSet<_>>::new();
        for provenance in builtin_implementation_provenance() {
            identities_by_builtin_path
                .entry(canonical_builtin_path(provenance.builtin_path))
                .or_default()
                .insert(provenance.name);
        }
        Self {
            exact_builtin_names: builtin_catalog_entries()
                .iter()
                .map(|entry| entry.identity.name)
                .chain(
                    builtin_functions()
                        .into_iter()
                        .map(|function| function.name),
                )
                .collect(),
            identities_by_builtin_path,
        }
    }

    pub(super) fn gpu_spec(
        &self,
        spec: &'static BuiltinGpuSpec,
        declaration: &'static str,
        builtin_path: &'static str,
        source_file: String,
        module_path: &'static str,
    ) -> GpuSpecRecord {
        gpu_spec(
            spec,
            declaration,
            builtin_path,
            source_file,
            module_path,
            self,
        )
    }

    pub(super) fn fusion_spec(
        &self,
        spec: &'static BuiltinFusionSpec,
        declaration: &'static str,
        builtin_path: &'static str,
        source_file: String,
        module_path: &'static str,
    ) -> FusionSpecRecord {
        fusion_spec(
            spec,
            declaration,
            builtin_path,
            source_file,
            module_path,
            self,
        )
    }

    #[cfg(test)]
    pub(super) fn from_exact_builtin_names(names: impl IntoIterator<Item = &'static str>) -> Self {
        Self {
            exact_builtin_names: names.into_iter().collect(),
            identities_by_builtin_path: BTreeMap::new(),
        }
    }

    #[cfg(test)]
    pub(super) fn with_builtin_ownership(
        mut self,
        builtin_path: &'static str,
        names: impl IntoIterator<Item = &'static str>,
    ) -> Self {
        self.identities_by_builtin_path
            .insert(builtin_path, names.into_iter().collect());
        self
    }

    #[cfg(test)]
    pub(super) fn owner(&self, key: &'static str, builtin_path: &'static str) -> SpecOwnerRecord {
        spec_owner(key, builtin_path, self)
    }
}

fn gpu_spec(
    spec: &'static BuiltinGpuSpec,
    declaration: &'static str,
    builtin_path: &'static str,
    source_file: String,
    module_path: &'static str,
    classifier: &SpecOwnershipClassifier,
) -> GpuSpecRecord {
    GpuSpecRecord {
        key: spec.name,
        declaration,
        builtin_path: canonical_builtin_path(builtin_path),
        source_file,
        module_path,
        owner: spec_owner(spec.name, builtin_path, classifier),
        operation: match spec.op_kind {
            GpuOpKind::Elementwise => "elementwise".into(),
            GpuOpKind::Reduction => "reduction".into(),
            GpuOpKind::MatMul => "matmul".into(),
            GpuOpKind::Transpose => "transpose".into(),
            GpuOpKind::PlotRender => "plot_render".into(),
            GpuOpKind::Custom(name) => format!("custom:{name}"),
        },
        supported_precisions: spec
            .supported_precisions
            .iter()
            .copied()
            .map(scalar_type)
            .collect(),
        broadcast: match spec.broadcast {
            BroadcastSemantics::Matlab => "matlab",
            BroadcastSemantics::ScalarOnly => "scalar_only",
            BroadcastSemantics::None => "none",
        },
        provider_hooks: spec
            .provider_hooks
            .iter()
            .copied()
            .map(provider_hook)
            .collect(),
        constant_strategy: constant_strategy(spec.constant_strategy),
        residency: match spec.residency {
            ResidencyPolicy::InheritInputs => "inherit_inputs",
            ResidencyPolicy::NewHandle => "new_handle",
            ResidencyPolicy::GatherImmediately => "gather_immediately",
        },
        nan_mode: match spec.nan_mode {
            ReductionNaN::Include => "include",
            ReductionNaN::Omit => "omit",
        },
        two_pass_threshold: spec.two_pass_threshold,
        workgroup_size: spec.workgroup_size,
        accepts_nan_mode: spec.accepts_nan_mode,
        notes: spec.notes,
    }
}

fn fusion_spec(
    spec: &'static BuiltinFusionSpec,
    declaration: &'static str,
    builtin_path: &'static str,
    source_file: String,
    module_path: &'static str,
    classifier: &SpecOwnershipClassifier,
) -> FusionSpecRecord {
    FusionSpecRecord {
        key: spec.name,
        declaration,
        builtin_path: canonical_builtin_path(builtin_path),
        source_file,
        module_path,
        owner: spec_owner(spec.name, builtin_path, classifier),
        shape: match spec.shape {
            ShapeRequirements::BroadcastCompatible => FusionShapeRecord::BroadcastCompatible,
            ShapeRequirements::Exact(dimensions) => FusionShapeRecord::Exact(dimensions.to_vec()),
            ShapeRequirements::Any => FusionShapeRecord::Any,
        },
        constant_strategy: constant_strategy(spec.constant_strategy),
        elementwise: spec.elementwise.as_ref().map(fusion_template),
        reduction: spec.reduction.as_ref().map(fusion_template),
        emits_nan: spec.emits_nan,
        notes: spec.notes,
    }
}

fn spec_owner(
    key: &'static str,
    builtin_path: &'static str,
    classifier: &SpecOwnershipClassifier,
) -> SpecOwnerRecord {
    if classifier.exact_builtin_names.contains(key) {
        return SpecOwnerRecord::ExactBuiltin {
            identity: BuiltinCatalogIdentity { name: key },
        };
    }
    let canonical_path = canonical_builtin_path(builtin_path);
    let affected_identities = classifier
        .identities_by_builtin_path
        .get(canonical_path)
        .into_iter()
        .flatten()
        .copied()
        .map(|name| BuiltinCatalogIdentity { name })
        .collect();
    SpecOwnerRecord::LegacyGroup {
        raw: key,
        affected_identities,
    }
}

fn canonical_builtin_path(path: &'static str) -> &'static str {
    path.strip_prefix("crate::").unwrap_or(path)
}

fn fusion_template(template: &FusionKernelTemplate) -> FusionTemplateRecord {
    FusionTemplateRecord {
        scalar_precisions: template
            .scalar_precisions
            .iter()
            .copied()
            .map(scalar_type)
            .collect(),
        expression_builder_registered: function_registered(template.wgsl_body),
    }
}

fn function_registered(_: FusionExprBuilder) -> bool {
    true
}

fn scalar_type(scalar: ScalarType) -> &'static str {
    match scalar {
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::I32 => "i32",
        ScalarType::Bool => "bool",
    }
}

fn constant_strategy(strategy: ConstantStrategy) -> &'static str {
    match strategy {
        ConstantStrategy::InlineLiteral => "inline_literal",
        ConstantStrategy::UniformBuffer => "uniform_buffer",
        ConstantStrategy::WorkgroupMemory => "workgroup_memory",
    }
}

fn provider_hook(hook: ProviderHook) -> ProviderHookRecord {
    match hook {
        ProviderHook::Unary { name } => ProviderHookRecord {
            kind: "unary",
            name,
            commutative: None,
        },
        ProviderHook::Binary { name, commutative } => ProviderHookRecord {
            kind: "binary",
            name,
            commutative: Some(commutative),
        },
        ProviderHook::Reduction { name } => ProviderHookRecord {
            kind: "reduction",
            name,
            commutative: None,
        },
        ProviderHook::Custom(name) => ProviderHookRecord {
            kind: "custom",
            name,
            commutative: None,
        },
    }
}
