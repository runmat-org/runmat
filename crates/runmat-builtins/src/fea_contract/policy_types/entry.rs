//! Declaration macro for one typed FEA policy catalog identity.
//!
//! The macro only assembles the identity-owned facts it is given: signature,
//! static argument rules, result class, documentation, and integer facts. The
//! contract is a pure host constructor (`fea_contract::catalog_entry::entry`).

/// One input descriptor: required, or optional with its documented default when
/// a `default:` is given (its static rule must then be `PolicyArg::Optional`).
macro_rules! policy_param {
    ($param:literal, $ty:ident, $description:expr) => {
        $crate::fea_contract::policy_types::descriptor::required(
            $param,
            $crate::BuiltinParamType::$ty,
            $description,
        )
    };
    ($param:literal, $ty:ident, $description:expr, $default:literal) => {
        $crate::fea_contract::policy_types::descriptor::optional(
            $param,
            $crate::BuiltinParamType::$ty,
            $default,
            $description,
        )
    };
}

/// Declares `$entry` plus the identity's `DESCRIPTOR` and `POLICY_CALL` in the
/// invoking module. Each identity lives in its own module.
macro_rules! policy_entry {
    (
        $entry:ident,
        name: $name:literal,
        result: $result:expr,
        signature: $label:expr,
        inputs: [$((
            $param:literal,
            $ty:ident,
            $description:expr,
            $rule:expr
            $(, default: $default:literal)?
            $(,)?
        )),* $(,)?],
        output: ($output:literal, $output_description:expr),
        integer_capabilities: $capabilities:expr,
        integer_audit: $audit:expr,
        documentation: $documentation:expr $(,)?
    ) => {
        const INPUTS: &[$crate::BuiltinParamDescriptor] = &[$(
            $crate::fea_contract::policy_types::entry::policy_param!(
                $param, $ty, $description $(, $default)?
            )
        ),*];
        const OUTPUTS: &[$crate::BuiltinParamDescriptor] =
            &[$crate::fea_contract::policy_types::descriptor::output($output, $output_description)];
        const SIGNATURES: &[$crate::BuiltinSignatureDescriptor] =
            &[$crate::BuiltinSignatureDescriptor { label: $label, inputs: INPUTS, outputs: OUTPUTS }];
        pub(crate) const DESCRIPTOR: $crate::BuiltinDescriptor =
            $crate::fea_contract::policy_types::descriptor::fixed_descriptor(SIGNATURES);
        pub(crate) const POLICY_CALL: $crate::fea_contract::policy_types::inference::PolicyCall =
            $crate::fea_contract::policy_types::inference::PolicyCall {
                result: $result,
                arguments: &[$($rule),*],
            };
        fn infer(
            request: &runmat_types::CallRequest,
            entry: &$crate::BuiltinCatalogEntry,
        ) -> runmat_types::CallInference {
            $crate::fea_contract::policy_types::inference::infer_policy(request, entry, &POLICY_CALL)
        }
        pub(crate) const $entry: $crate::BuiltinCatalogEntry =
            $crate::fea_contract::catalog_entry::entry(
                $crate::fea_contract::catalog_entry::ConstructorEntryDefinition {
                    provenance: $crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
                    name: $name,
                    documentation: $documentation,
                    descriptor: &DESCRIPTOR,
                    inference: infer,
                    integer_capabilities: $capabilities,
                    integer_audit: $audit,
                    filesystem_read: false,
                    distributed: $crate::BuiltinDistributedPolicy::Unsupported,
                },
            );
    };
}

pub(crate) use policy_entry;
pub(crate) use policy_param;
