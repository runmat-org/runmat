//! Declaration macros for time-sampled and callable-backed signal identities.
//!
//! Like `policy_types::entry::policy_entry`, each macro declares `$entry`, the
//! identity's `DESCRIPTOR`, and its inference function in the invoking module,
//! so each identity lives in its own module. The contract is a pure host
//! constructor (`fea_contract::catalog_entry::entry`).

/// `fea.<Quantity>Signal.fromTimeSamples(id, times, values, interpolation, extrapolation)`.
macro_rules! time_samples_entry {
    (
        $entry:ident,
        name: $name:literal,
        class: $class:expr,
        shape: $shape:ident,
        signature: $label:literal,
        documentation: $documentation:expr $(,)?
    ) => {
        const SIGNATURES: &[$crate::BuiltinSignatureDescriptor] =
            &[$crate::BuiltinSignatureDescriptor {
                label: $label,
                inputs: &$crate::fea_contract::policy_choices::descriptor::TIME_SAMPLE_INPUTS,
                outputs: &$crate::fea_contract::policy_choices::descriptor::SIGNAL_OUTPUT,
            }];
        pub(crate) const DESCRIPTOR: $crate::BuiltinDescriptor =
            $crate::fea_contract::policy_types::descriptor::fixed_descriptor(SIGNATURES);
        fn infer(
            request: &runmat_types::CallRequest,
            entry: &$crate::BuiltinCatalogEntry,
        ) -> runmat_types::CallInference {
            $crate::fea_contract::policy_choices::inference::infer_time_samples(
                request,
                entry,
                $class,
                $crate::fea_contract::policy_choices::inference::SignalShape::$shape,
            )
        }
        pub(crate) const $entry: $crate::BuiltinCatalogEntry =
            $crate::fea_contract::catalog_entry::entry(
                $crate::fea_contract::catalog_entry::ConstructorEntryDefinition {
                    provenance: $crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
                    name: $name,
                    documentation: $documentation,
                    descriptor: &DESCRIPTOR,
                    inference: infer,
                    integer_capabilities:
                        &$crate::fea_contract::policy_choices::integer::TIME_SAMPLE_CAPABILITIES,
                    integer_audit: None,
                    filesystem_read: false,
                    distributed: $crate::BuiltinDistributedPolicy::Unsupported,
                },
            );
    };
}

/// `fea.<Quantity>Signal.fromFunction(id, callable, domain, sampling, Name, Value, ...)`.
macro_rules! function_signal_entry {
    (
        $entry:ident,
        name: $name:literal,
        class: $class:expr,
        signature: $label:literal,
        documentation: $documentation:expr $(,)?
    ) => {
        const SIGNATURES: &[$crate::BuiltinSignatureDescriptor] =
            &[$crate::BuiltinSignatureDescriptor {
                label: $label,
                inputs: &$crate::fea_contract::policy_choices::descriptor::FUNCTION_INPUTS,
                outputs: &$crate::fea_contract::policy_choices::descriptor::SIGNAL_OUTPUT,
            }];
        pub(crate) const DESCRIPTOR: $crate::BuiltinDescriptor =
            $crate::fea_contract::policy_types::descriptor::fixed_descriptor(SIGNATURES);
        fn infer(
            request: &runmat_types::CallRequest,
            entry: &$crate::BuiltinCatalogEntry,
        ) -> runmat_types::CallInference {
            $crate::fea_contract::policy_choices::inference::infer_function_signal(
                request, entry, $class,
            )
        }
        pub(crate) const $entry: $crate::BuiltinCatalogEntry =
            $crate::fea_contract::catalog_entry::entry(
                $crate::fea_contract::catalog_entry::ConstructorEntryDefinition {
                    provenance: $crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
                    name: $name,
                    documentation: $documentation,
                    descriptor: &DESCRIPTOR,
                    inference: infer,
                    integer_capabilities: &[],
                    integer_audit: Some(
                        &$crate::fea_contract::policy_choices::integer::FUNCTION_SIGNAL_AUDIT,
                    ),
                    filesystem_read: false,
                    distributed: $crate::BuiltinDistributedPolicy::Unsupported,
                },
            );
    };
}

pub(crate) use function_signal_entry;
pub(crate) use time_samples_entry;
