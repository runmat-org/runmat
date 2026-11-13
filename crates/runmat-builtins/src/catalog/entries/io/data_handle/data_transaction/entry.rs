//! Typed declaration construction shared by transaction identity packages.

macro_rules! transaction_entry {
    ($entry:ident, $descriptor:ident, $infer:ident, $name:literal, $label:literal, $inputs:expr, $output:expr, $form:expr, $doc:expr, $effects:expr, $purity:expr, $integer:expr, $audit:expr) => {
        transaction_entry!(
            $entry,
            $descriptor,
            $infer,
            $name,
            $label,
            $inputs,
            $output,
            $form,
            $doc,
            $effects,
            $purity,
            $integer,
            $audit,
            crate::BuiltinAsyncBehavior::NeverSuspends
        );
    };
    ($entry:ident, $descriptor:ident, $infer:ident, $name:literal, $label:literal, $inputs:expr, $output:expr, $form:expr, $doc:expr, $effects:expr, $purity:expr, $integer:expr, $audit:expr, $async_behavior:expr) => {
        pub const $descriptor: crate::BuiltinDescriptor = crate::BuiltinDescriptor {
            signatures: &[crate::BuiltinSignatureDescriptor {
                label: $label,
                inputs: $inputs,
                outputs: $output,
            }],
            output_mode: crate::BuiltinOutputMode::Fixed,
            completion_policy: crate::BuiltinCompletionPolicy::Public,
            errors: $crate::catalog::entries::io::data_handle::descriptor::ERRORS,
        };
        fn $infer(
            request: &runmat_types::CallRequest,
            entry: &crate::BuiltinCatalogEntry,
        ) -> runmat_types::CallInference {
            $crate::catalog::entries::io::data_handle::transaction_inference::infer(
                request, entry, $form,
            )
        }
        pub const $entry: crate::BuiltinCatalogEntry =
            $crate::catalog::entries::io::data_handle::contract::entry(
                crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
                $name,
                $doc,
                &$descriptor,
                $infer,
                $crate::catalog::entries::io::data_handle::contract::Behavior {
                    effects: $effects,
                    capabilities: $crate::catalog::entries::io::data_handle::contract::FILESYSTEM,
                    purity: $purity,
                    async_behavior: $async_behavior,
                    integer_capabilities: $integer,
                    integer_audit: $audit,
                },
            );
    };
}
