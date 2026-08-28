use runmat_builtins::{
    builtin_catalog_entry_by_name, builtin_name_is_known, BuiltinAcceleratorPolicy,
    BuiltinReachability,
};
use runmat_hir::CallableIdentity;

use crate::MirOperand;

use super::{kind_token, Certainty, Kind, Reason, Walker};

impl Walker<'_> {
    pub(super) fn identity(&mut self, from: &str, identity: &CallableIdentity, reason: Reason) {
        match identity {
            CallableIdentity::BoundFunction(function)
            | CallableIdentity::AnonymousFunction(function)
            | CallableIdentity::ExternalFunction { function, .. } => self.retain_function(
                *function,
                Certainty::Definite,
                Some(from.into()),
                reason,
                identity.display_name(),
            ),
            CallableIdentity::Builtin(id) => self.builtin_reference(
                from,
                &id.0,
                if reason == Reason::FunctionHandle {
                    Certainty::FiniteDynamic
                } else {
                    Certainty::Definite
                },
                reason,
                None,
            ),
            CallableIdentity::Imported(path) => {
                let symbol = path
                    .module
                    .display_name()
                    .unwrap_or_else(|| "<import>".into());
                self.named_dynamic(from, Kind::ExternalCallable, symbol, "imported callable");
            }
            CallableIdentity::Method(method) => self.named_dynamic(
                from,
                Kind::Method,
                method.0.clone(),
                "runtime object method dispatch",
            ),
            CallableIdentity::DynamicName(name) => {
                self.resolve_named_dynamic(from, &name.0, "runtime name resolution")
            }
            CallableIdentity::ExternalName(name) => self.named_dynamic(
                from,
                Kind::ExternalCallable,
                name.display_name().unwrap_or_else(|| "<external>".into()),
                "external callable boundary",
            ),
        }
    }

    pub(super) fn named_dynamic(&mut self, from: &str, kind: Kind, symbol: String, detail: &str) {
        let id = format!("{}:{}", kind_token(kind), symbol.to_ascii_lowercase());
        self.node(
            id.clone(),
            kind,
            "runtime".into(),
            symbol,
            Certainty::FiniteDynamic,
        );
        self.edge(
            Some(from.into()),
            id,
            Certainty::FiniteDynamic,
            if kind == Kind::Method {
                Reason::MethodDispatch
            } else {
                Reason::DynamicNamedCall
            },
            Some(detail.into()),
        );
    }

    pub(super) fn resolve_named_dynamic(&mut self, from: &str, symbol: &str, detail: &str) {
        let functions = self
            .assembly
            .functions
            .iter()
            .filter_map(|(function, metadata)| {
                metadata
                    .name
                    .0
                    .eq_ignore_ascii_case(symbol)
                    .then_some(*function)
            })
            .collect::<Vec<_>>();
        let builtin = builtin_name_is_known(symbol);
        if functions.is_empty() && !builtin {
            self.named_dynamic(from, Kind::RuntimeCatalog, symbol.into(), detail);
            return;
        }
        for function in functions {
            self.retain_function(
                function,
                Certainty::FiniteDynamic,
                Some(from.into()),
                Reason::DynamicNamedCall,
                Some(detail.into()),
            );
        }
        if builtin {
            self.builtin_reference(
                from,
                symbol,
                Certainty::FiniteDynamic,
                Reason::DynamicNamedCall,
                Some(detail.into()),
            );
        }
    }

    pub(super) fn unknown_dynamic(&mut self, from: &str, detail: &str) {
        let id = "runtime_catalog:unknown".to_string();
        self.node(
            id.clone(),
            Kind::RuntimeCatalog,
            "runtime".into(),
            "unknown callable".into(),
            Certainty::Unknown,
        );
        self.edge(
            Some(from.into()),
            id,
            Certainty::Unknown,
            Reason::DynamicCall,
            Some(detail.into()),
        );
    }

    fn builtin_reference(
        &mut self,
        from: &str,
        name: &str,
        requested_certainty: Certainty,
        reason: Reason,
        detail: Option<String>,
    ) {
        let id = format!("builtin:{}", name.to_ascii_lowercase());
        let contract_certainty = builtin_catalog_entry_by_name(name)
            .map(|entry| match entry.link.reachability {
                BuiltinReachability::Always => Certainty::Definite,
                BuiltinReachability::Dynamic | BuiltinReachability::Feature(_) => {
                    Certainty::FiniteDynamic
                }
            })
            .unwrap_or(Certainty::Definite);
        let certainty = requested_certainty.max(contract_certainty);
        self.node(
            id.clone(),
            Kind::Builtin,
            "runmat-builtins".into(),
            name.into(),
            certainty,
        );
        self.edge(Some(from.into()), id.clone(), certainty, reason, detail);
        let Some(entry) = builtin_catalog_entry_by_name(name) else {
            return;
        };
        for dependency in entry.link.artifact_dependencies {
            let target = format!("artifact_dependency:{dependency}");
            self.node(
                target.clone(),
                Kind::ArtifactDependency,
                "runtime-archive".into(),
                (*dependency).into(),
                certainty,
            );
            self.edge(
                Some(id.clone()),
                target,
                certainty,
                Reason::BuiltinLinkContract,
                Some(format!("{:?}", entry.link.policy)),
            );
        }
        if !matches!(
            entry.placement.accelerator,
            BuiltinAcceleratorPolicy::Forbidden
        ) {
            let target = "provider:accelerator".to_string();
            self.node(
                target.clone(),
                Kind::Provider,
                "runmat-accelerate-api".into(),
                "accelerator".into(),
                Certainty::FiniteDynamic,
            );
            self.edge(
                Some(id.clone()),
                target,
                Certainty::FiniteDynamic,
                Reason::AcceleratorPlacement,
                Some(format!("{:?}", entry.placement.accelerator)),
            );
        }
        for extension in entry.extensions {
            let target = format!("extension:{}", extension.id);
            self.node(
                target.clone(),
                Kind::Extension,
                "runmat-builtins".into(),
                extension.id.into(),
                Certainty::FiniteDynamic,
            );
            self.edge(
                Some(id.clone()),
                target,
                Certainty::FiniteDynamic,
                Reason::ExtensionCapability,
                Some(extension.description.into()),
            );
        }
    }

    pub(super) fn class(&mut self, from: &str, class_name: &str, reason: Reason) {
        let id = format!("class:{}", class_name.to_ascii_lowercase());
        self.node(
            id.clone(),
            Kind::Class,
            "class".into(),
            class_name.into(),
            Certainty::FiniteDynamic,
        );
        self.edge(
            Some(from.into()),
            id.clone(),
            Certainty::FiniteDynamic,
            reason,
            None,
        );
        let methods = self
            .assembly
            .classes
            .iter()
            .filter(|class| {
                class
                    .name
                    .display_name()
                    .is_some_and(|name| name.eq_ignore_ascii_case(class_name))
            })
            .flat_map(|class| class.methods.iter().map(|method| method.function))
            .collect::<Vec<_>>();
        for method in methods {
            self.retain_function(
                method,
                Certainty::FiniteDynamic,
                Some(id.clone()),
                Reason::MethodDispatch,
                Some("local class method candidate".into()),
            );
        }
    }

    pub(super) fn dynamic_target(&mut self, from: &str, target: &MirOperand, detail: &str) {
        match target {
            MirOperand::FunctionHandle(identity) => {
                self.identity(from, identity, Reason::FunctionHandle)
            }
            MirOperand::Constant(crate::MirConstant::String(value)) => {
                self.resolve_named_dynamic(from, &value.runtime_text(), detail)
            }
            MirOperand::Constant(crate::MirConstant::Symbol(value)) => {
                self.resolve_named_dynamic(from, &value.0, detail)
            }
            MirOperand::Local(_) | MirOperand::Constant(_) => self.unknown_dynamic(from, detail),
        }
    }

    pub(super) fn operator(&mut self, from: &str, operator: runmat_types::OperatorKind) {
        let Some(name) = operator.overload_name() else {
            return;
        };
        self.builtin_reference(
            from,
            name,
            Certainty::FiniteDynamic,
            Reason::OperatorDispatch,
            Some(format!("{operator:?} overload fallback")),
        );
    }
}
