use sha2::{Digest, Sha256};

use crate::{
    builtin_catalog_aliases, builtin_catalog_entries, builtin_constant_catalog_entries,
    builtin_functions, canonical_catalog_fingerprint, AccelTag, BuiltinCompletionPolicy,
    BuiltinConstantLiteral, BuiltinOutputMode, BuiltinParamArity, BuiltinParamType, Type,
    TypeResolverKind, BUILTIN_CATALOG_SCHEMA_VERSION,
};

/// Bump when execution-relevant builtin behavior changes in a way the
/// declarative catalog cannot observe, such as type-resolver semantics.
pub const BUILTIN_CATALOG_SCHEMA: u32 = BUILTIN_CATALOG_SCHEMA_VERSION;

/// Return a target-independent fingerprint of the builtin execution contract.
///
/// Documentation and function addresses are deliberately excluded: the former
/// is not executable and the latter is host-specific. Runtime release
/// compatibility is represented by the separate runtime fingerprint.
pub fn builtin_catalog_fingerprint() -> [u8; 32] {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static FINGERPRINT: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
        *FINGERPRINT.get_or_init(compute_builtin_catalog_fingerprint)
    }

    #[cfg(target_arch = "wasm32")]
    {
        WASM_FINGERPRINT.with(|cache| cache.get_or_init(compute_builtin_catalog_fingerprint))
    }
}

#[cfg(any(target_arch = "wasm32", test))]
mod cache;
#[cfg(any(target_arch = "wasm32", test))]
use cache::RegistryFingerprintCache;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WASM_FINGERPRINT: RegistryFingerprintCache = const { RegistryFingerprintCache::new() };
}

/// Browser registries accept late submissions. Every mutation invalidates the
/// calling worker's cache before changing its registry; session resets do not.
#[cfg(target_arch = "wasm32")]
pub(crate) fn invalidate_wasm_registry_fingerprint() {
    WASM_FINGERPRINT.with(RegistryFingerprintCache::invalidate);
}

fn compute_builtin_catalog_fingerprint() -> [u8; 32] {
    let mut functions = builtin_functions();
    functions.sort_unstable_by_key(|function| function.name);

    let mut hash = Sha256::new();
    field(&mut hash, b"runmat-builtin-catalog");
    number(&mut hash, BUILTIN_CATALOG_SCHEMA as u64);
    field(
        &mut hash,
        &canonical_catalog_fingerprint(builtin_catalog_entries(), builtin_catalog_aliases())
            .expect("static builtin catalog must serialize"),
    );
    let constants = builtin_constant_catalog_entries();
    number(&mut hash, constants.len() as u64);
    for constant in constants {
        field(&mut hash, constant.name.as_bytes());
        field(
            &mut hash,
            &serde_json::to_vec(&constant.kind).expect("static constant kind must serialize"),
        );
        match constant.literal {
            None => field(&mut hash, b"no-scalar-literal"),
            Some(BuiltinConstantLiteral::Logical(value)) => {
                field(&mut hash, b"logical-scalar-literal");
                boolean(&mut hash, value);
            }
        }
    }
    number(&mut hash, functions.len() as u64);
    for function in functions {
        field(&mut hash, function.name.as_bytes());
        number(&mut hash, function.param_types.len() as u64);
        for ty in &function.param_types {
            encode_type(&mut hash, ty);
        }
        encode_type(&mut hash, &function.return_type);
        field(
            &mut hash,
            match function.execution_stack {
                crate::ExecutionStackRequirement::Any => b"stack:any",
                crate::ExecutionStackRequirement::Process => b"stack:process",
            },
        );
        field(
            &mut hash,
            match function.type_resolver {
                None => b"none",
                Some(TypeResolverKind::Simple(_)) => b"simple",
                Some(TypeResolverKind::WithContext(_)) => b"context",
            },
        );
        number(&mut hash, function.accel_tags.len() as u64);
        for tag in function.accel_tags {
            field(
                &mut hash,
                match tag {
                    AccelTag::Unary => b"unary",
                    AccelTag::Elementwise => b"elementwise",
                    AccelTag::Reduction => b"reduction",
                    AccelTag::MatMul => b"matmul",
                    AccelTag::Transpose => b"transpose",
                    AccelTag::ArrayConstruct => b"array-construct",
                },
            );
        }
        boolean(&mut hash, function.is_sink);
        boolean(&mut hash, function.suppress_auto_output);
        match function.descriptor {
            None => field(&mut hash, b"no-descriptor"),
            Some(descriptor) => {
                field(&mut hash, b"descriptor");
                field(
                    &mut hash,
                    match descriptor.output_mode {
                        BuiltinOutputMode::Fixed => b"fixed",
                        BuiltinOutputMode::ByRequestedOutputCount => b"requested-output-count",
                    },
                );
                field(
                    &mut hash,
                    match descriptor.completion_policy {
                        BuiltinCompletionPolicy::Public => b"public",
                        BuiltinCompletionPolicy::MethodOnly => b"method-only",
                        BuiltinCompletionPolicy::HiddenInternal => b"hidden-internal",
                    },
                );
                number(&mut hash, descriptor.signatures.len() as u64);
                for signature in descriptor.signatures {
                    field(&mut hash, signature.label.as_bytes());
                    encode_parameters(&mut hash, signature.inputs);
                    encode_parameters(&mut hash, signature.outputs);
                }
                number(&mut hash, descriptor.errors.len() as u64);
                for error in descriptor.errors {
                    field(&mut hash, error.code.as_bytes());
                    optional(&mut hash, error.identifier);
                }
            }
        }
    }
    hash.finalize().into()
}

fn encode_parameters(hash: &mut Sha256, parameters: &[crate::BuiltinParamDescriptor]) {
    number(hash, parameters.len() as u64);
    for parameter in parameters {
        field(hash, parameter.name.as_bytes());
        field(
            hash,
            match parameter.ty {
                BuiltinParamType::Any => b"any",
                BuiltinParamType::NumericScalar => b"numeric-scalar",
                BuiltinParamType::IntegerScalar => b"integer-scalar",
                BuiltinParamType::StringScalar => b"string-scalar",
                BuiltinParamType::NumericArray => b"numeric-array",
                BuiltinParamType::LogicalArray => b"logical-array",
                BuiltinParamType::SizeArg => b"size-arg",
                BuiltinParamType::LikePrototype => b"like-prototype",
                BuiltinParamType::AxesHandle => b"axes-handle",
                BuiltinParamType::StyleSpec => b"style-spec",
                BuiltinParamType::PropertyName => b"property-name",
                BuiltinParamType::PropertyValue => b"property-value",
                BuiltinParamType::Callable => b"callable",
            },
        );
        field(
            hash,
            match parameter.arity {
                BuiltinParamArity::Required => b"required",
                BuiltinParamArity::Optional => b"optional",
                BuiltinParamArity::Variadic => b"variadic",
            },
        );
        optional(hash, parameter.default);
    }
}

fn encode_type(hash: &mut Sha256, ty: &Type) {
    match ty {
        Type::Int => field(hash, b"int"),
        Type::Num => field(hash, b"num"),
        Type::Bool => field(hash, b"bool"),
        Type::Logical { shape } => {
            field(hash, b"logical");
            encode_shape(hash, shape);
        }
        Type::String => field(hash, b"string"),
        Type::Tensor { shape } => {
            field(hash, b"tensor");
            encode_shape(hash, shape);
        }
        Type::Symbolic => field(hash, b"symbolic"),
        Type::SymbolicArray { shape } => {
            field(hash, b"symbolic-array");
            encode_shape(hash, shape);
        }
        Type::Cell {
            element_type,
            length,
        } => {
            field(hash, b"cell");
            match element_type {
                Some(ty) => {
                    boolean(hash, true);
                    encode_type(hash, ty);
                }
                None => boolean(hash, false),
            }
            optional_number(hash, *length);
        }
        Type::Function { params, returns } => {
            field(hash, b"function");
            number(hash, params.len() as u64);
            for parameter in params {
                encode_type(hash, parameter);
            }
            encode_type(hash, returns);
        }
        Type::Void => field(hash, b"void"),
        Type::Unknown => field(hash, b"unknown"),
        Type::Union(types) => {
            field(hash, b"union");
            number(hash, types.len() as u64);
            for ty in types {
                encode_type(hash, ty);
            }
        }
        Type::Struct { known_fields } => {
            field(hash, b"struct");
            match known_fields {
                None => boolean(hash, false),
                Some(fields) => {
                    boolean(hash, true);
                    number(hash, fields.len() as u64);
                    for name in fields {
                        field(hash, name.as_bytes());
                    }
                }
            }
        }
        Type::Object { class_name, shape } => {
            field(hash, b"object");
            optional(
                hash,
                class_name.as_ref().map(|identity| identity.display_name()),
            );
            encode_shape(hash, shape);
        }
        Type::OutputList(types) => {
            field(hash, b"output-list");
            number(hash, types.len() as u64);
            for ty in types {
                encode_type(hash, ty);
            }
        }
    }
}

fn encode_shape(hash: &mut Sha256, shape: &Option<Vec<Option<usize>>>) {
    match shape {
        None => boolean(hash, false),
        Some(dimensions) => {
            boolean(hash, true);
            number(hash, dimensions.len() as u64);
            for dimension in dimensions {
                optional_number(hash, *dimension);
            }
        }
    }
}

fn optional(hash: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            boolean(hash, true);
            field(hash, value.as_bytes());
        }
        None => boolean(hash, false),
    }
}

fn optional_number(hash: &mut Sha256, value: Option<usize>) {
    match value {
        Some(value) => {
            boolean(hash, true);
            number(hash, value as u64);
        }
        None => boolean(hash, false),
    }
}

fn boolean(hash: &mut Sha256, value: bool) {
    field(hash, if value { b"true" } else { b"false" });
}

fn number(hash: &mut Sha256, value: u64) {
    field(hash, &value.to_be_bytes());
}

fn field(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_be_bytes());
    hash.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::legacy_index::INVENTORY_TRAVERSALS;

    #[test]
    fn catalog_fingerprint_is_stable_for_one_process() {
        let first = builtin_catalog_fingerprint();
        let second = builtin_catalog_fingerprint();
        assert_eq!(first, second);
        assert_ne!(first, [0; 32]);

        // Model late registration without mutating the process's static native
        // inventory. Repeated reads stay O(1), but a mutation cannot retain an
        // earlier execution fingerprint.
        let cache = RegistryFingerprintCache::new();
        let calls = std::cell::Cell::new(0);
        for _ in 0..4 {
            assert_eq!(
                cache.get_or_init(|| {
                    calls.set(calls.get() + 1);
                    first
                }),
                first
            );
        }
        assert_eq!(calls.get(), 1);
        cache.invalidate();
        let changed = [7; 32];
        assert_eq!(
            cache.get_or_init(|| {
                calls.set(calls.get() + 1);
                changed
            }),
            changed
        );
        assert_eq!(
            cache.get_or_init(|| panic!("unchanged registry was recomputed")),
            changed
        );
        assert_eq!(calls.get(), 2);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_cached_fingerprint_does_not_repeat_inventory_traversal() {
        let cached = builtin_catalog_fingerprint();
        let after_initialization = INVENTORY_TRAVERSALS.with(std::cell::Cell::get);
        for _ in 0..4 {
            assert_eq!(builtin_catalog_fingerprint(), cached);
        }
        assert_eq!(
            INVENTORY_TRAVERSALS.with(std::cell::Cell::get),
            after_initialization
        );

        // The independent uncached computation still traverses the same
        // inventory and must produce exactly the public native bytes.
        assert_eq!(compute_builtin_catalog_fingerprint(), cached);
        assert_eq!(
            INVENTORY_TRAVERSALS.with(std::cell::Cell::get),
            after_initialization + 1
        );
        assert_eq!(builtin_catalog_fingerprint(), cached);
        assert_eq!(
            INVENTORY_TRAVERSALS.with(std::cell::Cell::get),
            after_initialization + 1
        );
    }
}
