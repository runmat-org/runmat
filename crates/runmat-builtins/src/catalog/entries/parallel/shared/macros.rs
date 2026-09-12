macro_rules! parallel_entry {
    (
        $constant:ident,
        $name:literal,
        $rule:expr,
        $documentation:expr,
        $descriptor:ident,
        $maturity:expr,
        $async_behavior:expr,
        $purity:expr,
        $effects:expr
    ) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: $maturity,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: $async_behavior,
                purity: $purity,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: $effects,
                capabilities: &PARALLEL_RUNTIME,
            },
            placement: CURRENT_PLACEMENT,
            link: CURRENT_LINK,
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

macro_rules! signature {
    ($name:ident, $label:literal, $inputs:expr, $outputs:expr) => {
        const $name: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: $inputs,
            outputs: $outputs,
        }];
    };
}

macro_rules! descriptor {
    ($name:ident, $signatures:ident) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &LOWERING_ERRORS,
        };
    };
}

macro_rules! documented_parallel_data_entry {
    ($constant:ident, $name:literal, $rule:expr, $documentation:expr, $descriptor:ident) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &PARALLEL_EFFECTS,
                capabilities: &PARALLEL_RUNTIME,
            },
            placement: PARALLEL_PLACEMENT,
            link: PARALLEL_LINK,
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

macro_rules! codistributor_entry {
    ($constant:ident, $name:literal, $rule:expr, $documentation:expr, $descriptor:ident) => {
        codistributor_entry!(
            $constant,
            $name,
            $rule,
            $documentation,
            $descriptor,
            PARALLEL_PLACEMENT
        );
    };
    ($constant:ident, $name:literal, $rule:expr, $documentation:expr, $descriptor:ident, $placement:expr) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Complete,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::NeverSuspends,
                purity: BuiltinPurity::Pure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[],
                capabilities: &[],
            },
            placement: $placement,
            link: PARALLEL_LINK,
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}
