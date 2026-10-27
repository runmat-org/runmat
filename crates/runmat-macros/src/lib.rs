use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use syn::parse::{Parse, ParseStream};
use syn::{
    parse_macro_input, AttributeArgs, Expr, FnArg, ItemConst, ItemFn, Lit, LitStr, Meta,
    MetaNameValue, NestedMeta, Pat,
};

mod invocation_parameters;
use invocation_parameters::{
    is_optional_call_admission_sink, is_optional_runtime_context, is_optional_usize,
};

static WASM_REGISTRY_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
static WASM_REGISTRY_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static WASM_REGISTRY_INIT: OnceLock<()> = OnceLock::new();

/// Attribute used to mark functions as implementing a runtime builtin.
///
/// Example:
/// ```rust,ignore
/// use runmat_macros::runtime_builtin;
///
/// #[runtime_builtin(name = "plot")]
/// pub fn plot_line(xs: &[f64]) {
///     /* implementation */
/// }
/// ```
///
/// This registers the function with the `runmat-builtins` inventory
/// so the runtime can discover it at start-up.
/// Canonical `binding_variant` functions may opt into `context = true`
/// and `requested_outputs = true`. Those leading, structurally typed
/// invocation parameters are not user arguments and do not affect arity.
/// `output_sequence = true` selects a handler returning a checked
/// `ValueSequence` directly, without a language-value output carrier.
#[proc_macro_attribute]
pub fn runtime_builtin(args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse attribute arguments as `name = "..."`
    let args = parse_macro_input!(args as AttributeArgs);
    let mut name_lit: Option<Lit> = None;
    let mut category_lit: Option<Lit> = None;
    let mut summary_lit: Option<Lit> = None;
    let mut keywords_lit: Option<Lit> = None;
    let mut errors_lit: Option<Lit> = None;
    let mut related_lit: Option<Lit> = None;
    let mut introduced_lit: Option<Lit> = None;
    let mut status_lit: Option<Lit> = None;
    let mut examples_lit: Option<Lit> = None;
    let mut accel_values: Vec<String> = Vec::new();
    let mut builtin_path_lit: Option<LitStr> = None;
    let mut type_resolver_path: Option<syn::Path> = None;
    let mut type_resolver_ctx_path: Option<syn::Path> = None;
    let mut descriptor_path: Option<syn::Path> = None;
    let mut extensions_path: Option<syn::Path> = None;
    let mut integer_capabilities_path: Option<syn::Path> = None;
    let mut integer_audit_path: Option<syn::Path> = None;
    let mut binding_variant_lit: Option<LitStr> = None;
    let mut explicit_context = false;
    let mut explicit_requested_outputs = false;
    let mut explicit_call_observation = false;
    let mut output_sequence = false;
    let mut sink_flag = false;
    let mut suppress_auto_output_flag = false;
    let mut process_execution_stack = false;
    let mut required_capabilities: Vec<String> = Vec::new();
    for arg in args {
        match arg {
            NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) => {
                if path.is_ident("name") {
                    name_lit = Some(lit);
                } else if path.is_ident("category") {
                    category_lit = Some(lit);
                } else if path.is_ident("summary") {
                    summary_lit = Some(lit);
                } else if path.is_ident("keywords") {
                    keywords_lit = Some(lit);
                } else if path.is_ident("errors") {
                    errors_lit = Some(lit);
                } else if path.is_ident("related") {
                    related_lit = Some(lit);
                } else if path.is_ident("introduced") {
                    introduced_lit = Some(lit);
                } else if path.is_ident("status") {
                    status_lit = Some(lit);
                } else if path.is_ident("examples") {
                    examples_lit = Some(lit);
                } else if path.is_ident("accel") {
                    if let Lit::Str(ls) = lit {
                        accel_values.extend(
                            ls.value()
                                .split(|c: char| c == ',' || c == '|' || c.is_ascii_whitespace())
                                .filter(|s| !s.is_empty())
                                .map(|s| s.to_ascii_lowercase()),
                        );
                    }
                } else if path.is_ident("sink") {
                    if let Lit::Bool(lb) = lit {
                        sink_flag = lb.value;
                    }
                } else if path.is_ident("suppress_auto_output") {
                    if let Lit::Bool(lb) = lit {
                        suppress_auto_output_flag = lb.value;
                    }
                } else if path.is_ident("execution_stack") {
                    if let Lit::Str(value) = lit {
                        match value.value().as_str() {
                            "process" => process_execution_stack = true,
                            "any" => process_execution_stack = false,
                            other => panic!(
                                "execution_stack must be \"any\" or \"process\", got {other:?}"
                            ),
                        }
                    } else {
                        panic!("execution_stack must be a string literal");
                    }
                } else if path.is_ident("capabilities") {
                    if let Lit::Str(value) = lit {
                        required_capabilities.extend(
                            value
                                .value()
                                .split(|character: char| {
                                    character == ','
                                        || character == '|'
                                        || character.is_ascii_whitespace()
                                })
                                .filter(|value| !value.is_empty())
                                .map(str::to_ascii_lowercase),
                        );
                    } else {
                        panic!("capabilities must be a string literal");
                    }
                } else if path.is_ident("builtin_path") {
                    if let Lit::Str(ls) = lit {
                        builtin_path_lit = Some(ls);
                    } else {
                        panic!("builtin_path must be a string literal");
                    }
                } else if path.is_ident("binding_variant") {
                    if let Lit::Str(ls) = lit {
                        binding_variant_lit = Some(ls);
                    } else {
                        panic!("binding_variant must be a string literal");
                    }
                } else if path.is_ident("context") {
                    if let Lit::Bool(value) = lit {
                        explicit_context = value.value;
                    } else {
                        panic!("context must be a boolean literal");
                    }
                } else if path.is_ident("requested_outputs") {
                    if let Lit::Bool(value) = lit {
                        explicit_requested_outputs = value.value;
                    } else {
                        panic!("requested_outputs must be a boolean literal");
                    }
                } else if path.is_ident("call_observation") {
                    if let Lit::Bool(value) = lit {
                        explicit_call_observation = value.value;
                    } else {
                        panic!("call_observation must be a boolean literal");
                    }
                } else if path.is_ident("output_sequence") {
                    if let Lit::Bool(value) = lit {
                        output_sequence = value.value;
                    } else {
                        panic!("output_sequence must be a boolean literal");
                    }
                } else if path.is_ident("type_resolver") {
                    if let Lit::Str(ls) = lit {
                        let parsed: syn::Path = ls.parse().expect("type_resolver must be a path");
                        type_resolver_path = Some(parsed);
                    } else {
                        panic!("type_resolver must be a string literal path");
                    }
                } else if path.is_ident("type_resolver_ctx") {
                    if let Lit::Str(ls) = lit {
                        let parsed: syn::Path =
                            ls.parse().expect("type_resolver_ctx must be a path");
                        type_resolver_ctx_path = Some(parsed);
                    } else {
                        panic!("type_resolver_ctx must be a string literal path");
                    }
                } else {
                    // Gracefully ignore unknown parameters for better IDE experience
                }
            }
            NestedMeta::Meta(Meta::List(list)) if list.path.is_ident("type_resolver") => {
                if list.nested.len() != 1 {
                    panic!("type_resolver expects exactly one path argument");
                }
                let nested = list.nested.first().unwrap();
                if let NestedMeta::Meta(Meta::Path(path)) = nested {
                    type_resolver_path = Some(path.clone());
                } else {
                    panic!("type_resolver expects a path argument");
                }
            }
            NestedMeta::Meta(Meta::List(list)) if list.path.is_ident("type_resolver_ctx") => {
                if list.nested.len() != 1 {
                    panic!("type_resolver_ctx expects exactly one path argument");
                }
                let nested = list.nested.first().unwrap();
                if let NestedMeta::Meta(Meta::Path(path)) = nested {
                    type_resolver_ctx_path = Some(path.clone());
                } else {
                    panic!("type_resolver_ctx expects a path argument");
                }
            }
            NestedMeta::Meta(Meta::List(list)) if list.path.is_ident("descriptor") => {
                if list.nested.len() != 1 {
                    panic!("descriptor expects exactly one path argument");
                }
                let nested = list.nested.first().unwrap();
                if let NestedMeta::Meta(Meta::Path(path)) = nested {
                    descriptor_path = Some(path.clone());
                } else {
                    panic!("descriptor expects a path argument");
                }
            }
            NestedMeta::Meta(Meta::List(list)) if list.path.is_ident("extensions") => {
                if list.nested.len() != 1 {
                    panic!("extensions expects exactly one path argument");
                }
                let nested = list.nested.first().unwrap();
                if let NestedMeta::Meta(Meta::Path(path)) = nested {
                    extensions_path = Some(path.clone());
                } else {
                    panic!("extensions expects a path argument");
                }
            }
            NestedMeta::Meta(Meta::List(list)) if list.path.is_ident("integer_capabilities") => {
                if list.nested.len() != 1 {
                    panic!("integer_capabilities expects exactly one path argument");
                }
                let nested = list.nested.first().unwrap();
                if let NestedMeta::Meta(Meta::Path(path)) = nested {
                    integer_capabilities_path = Some(path.clone());
                } else {
                    panic!("integer_capabilities expects a path argument");
                }
            }
            NestedMeta::Meta(Meta::List(list)) if list.path.is_ident("integer_audit") => {
                if list.nested.len() != 1 {
                    panic!("integer_audit expects exactly one path argument");
                }
                let nested = list.nested.first().unwrap();
                if let NestedMeta::Meta(Meta::Path(path)) = nested {
                    integer_audit_path = Some(path.clone());
                } else {
                    panic!("integer_audit expects a path argument");
                }
            }
            _ => {}
        }
    }
    let name_lit = name_lit.expect("expected `name = \"...\"` argument");
    let name_str = if let Lit::Str(ref s) = name_lit {
        s.value()
    } else {
        panic!("name must be a string literal");
    };

    let func: ItemFn = parse_macro_input!(input as ItemFn);
    let ident = &func.sig.ident;
    let is_async = func.sig.asyncness.is_some();

    // Extract param idents and types
    let mut param_idents = Vec::new();
    let mut param_types = Vec::new();
    for arg in &func.sig.inputs {
        match arg {
            FnArg::Typed(pt) => {
                // pattern must be ident
                if let Pat::Ident(pi) = pt.pat.as_ref() {
                    param_idents.push(pi.ident.clone());
                } else {
                    panic!("parameters must be simple identifiers");
                }
                param_types.push((*pt.ty).clone());
            }
            _ => panic!("self parameter not allowed"),
        }
    }
    if (explicit_context || explicit_requested_outputs || explicit_call_observation)
        && binding_variant_lit.is_none()
    {
        panic!("invocation metadata requires a canonical binding_variant");
    }
    if output_sequence && binding_variant_lit.is_none() {
        panic!("output_sequence requires a canonical binding_variant");
    }
    let context_param = if explicit_context {
        let context_type = param_types
            .first()
            .expect("context injection requires a first Option<RuntimeContext> parameter");
        if !is_optional_runtime_context(context_type) {
            panic!("context injection requires Option<RuntimeContext> as the first parameter");
        }
        Some((param_idents.remove(0), param_types.remove(0)))
    } else {
        None
    };
    let requested_outputs_param = if explicit_requested_outputs {
        let output_type = param_types.first().expect(
            "requested_outputs injection requires an Option<usize> parameter after context",
        );
        if !is_optional_usize(output_type) {
            panic!("requested_outputs injection requires Option<usize> after context");
        }
        Some((param_idents.remove(0), param_types.remove(0)))
    } else {
        None
    };
    let call_observation_param = if explicit_call_observation {
        let observation_type = param_types.first().expect(
            "call_observation injection requires an Option<CallAdmissionSink> parameter after invocation metadata",
        );
        if !is_optional_call_admission_sink(observation_type) {
            panic!("call_observation injection requires Option<CallAdmissionSink> after invocation metadata");
        }
        Some((param_idents.remove(0), param_types.remove(0)))
    } else {
        None
    };
    let param_len = param_idents.len();

    // Infer parameter types for BuiltinFunction
    let inferred_param_types: Vec<proc_macro2::TokenStream> =
        param_types.iter().map(infer_builtin_type).collect();

    // Infer return type for BuiltinFunction
    let inferred_return_type = match &func.sig.output {
        syn::ReturnType::Default => quote! { runmat_builtins::Type::Void },
        syn::ReturnType::Type(_, ty) => infer_builtin_type(ty),
    };

    // Canonical variadics retain their source owner; historical bindings may
    // still expose an explicitly ordinary Vec<Value> ABI.
    let is_last_variadic = param_types
        .last()
        .map(|ty| {
            // crude detection: type path starts with Vec and inner type is runmat_value::Value or Value
            if let syn::Type::Path(tp) = ty {
                if tp
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "BuiltinArguments")
                {
                    return true;
                }
                if tp
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident == "Vec")
                    .unwrap_or(false)
                {
                    if let syn::PathArguments::AngleBracketed(ab) =
                        &tp.path.segments.last().unwrap().arguments
                    {
                        if let Some(syn::GenericArgument::Type(syn::Type::Path(inner))) =
                            ab.args.first()
                        {
                            return inner
                                .path
                                .segments
                                .last()
                                .map(|s| s.ident == "Value")
                                .unwrap_or(false);
                        }
                    }
                }
            }
            false
        })
        .unwrap_or(false);

    // Generate wrapper ident
    let wrapper_ident = format_ident!("__rt_wrap_{}", ident);
    let body_ident = format_ident!("__rt_owned_body_{}", ident);
    let layout_ident = format_ident!("__rt_owned_layout_{}", ident);

    let conv_stmts: Vec<proc_macro2::TokenStream> = if is_last_variadic && param_len > 0 {
        let mut stmts = Vec::new();
        // Convert fixed params (all but last)
        for (i, (ident, ty)) in param_idents
            .iter()
            .zip(param_types.iter())
            .enumerate()
            .take(param_len - 1)
        {
            let conversion =
                builtin_parameter_conversion(binding_variant_lit.is_some(), i, ident, ty);
            stmts.push(conversion);
        }
        // Canonical calls transfer a logical range of the original owner.
        let last_ident = &param_idents[param_len - 1];
        if matches!(&param_types[param_len - 1], syn::Type::Path(path)
            if path.path.segments.last().is_some_and(|segment| segment.ident == "BuiltinArguments"))
        {
            stmts.push(quote! {
                let #last_ident: crate::call::BuiltinArguments = args.into_range((#param_len - 1)..)?;
            });
        } else {
            if binding_variant_lit.is_some() {
                return syn::Error::new_spanned(
                    &param_types[param_len - 1],
                    "canonical variadic bindings require BuiltinArguments to retain their argument owner",
                )
                .to_compile_error()
                .into();
            }
            stmts.push(quote! {
            let #last_ident : Vec<runmat_value::Value> = {
                let mut v = Vec::new();
                for j in (#param_len-1)..args.len() {
                    let item : runmat_value::Value = std::convert::TryInto::try_into(&args[j])?;
                    v.push(item);
                }
                v
            };
            });
        }
        stmts
    } else {
        param_idents
            .iter()
            .zip(param_types.iter())
            .enumerate()
            .map(|(i, (ident, ty))| {
                builtin_parameter_conversion(binding_variant_lit.is_some(), i, ident, ty)
            })
            .collect()
    };

    let invocation_parameters = context_param
        .iter()
        .chain(requested_outputs_param.iter())
        .chain(call_observation_param.iter())
        .map(|(ident, _)| ident)
        .collect::<Vec<_>>();
    let call_expr = if is_async {
        quote! { #ident(#(#invocation_parameters,)* #(#param_idents),*).await? }
    } else {
        quote! { #ident(#(#invocation_parameters,)* #(#param_idents),*)? }
    };

    let context_binding = context_param.as_ref().map(|(ident, ty)| {
        quote! { let #ident: #ty = invocation.context; }
    });
    let owned_context_binding = context_param.as_ref().map(|(ident, ty)| {
        quote! { let #ident: #ty = input.context; }
    });
    let requested_outputs_binding = requested_outputs_param.as_ref().map(|(ident, ty)| {
        quote! { let #ident: #ty = invocation.requested_outputs; }
    });
    let owned_requested_outputs_binding = requested_outputs_param.as_ref().map(|(ident, ty)| {
        quote! { let #ident: #ty = input.requested_outputs; }
    });
    let call_observation_binding = call_observation_param.as_ref().map(|(ident, ty)| {
        quote! { let #ident: #ty = invocation.call_observation; }
    });
    let owned_call_observation_binding = call_observation_param.as_ref().map(|(ident, ty)| {
        quote! { let #ident: #ty = input.call_observation; }
    });

    let wrapper_input = if binding_variant_lit.is_some() {
        quote! { invocation: crate::builtin::RuntimeBuiltinInvocation<'_> }
    } else {
        quote! { args: &[runmat_value::Value] }
    };
    let wrapper_return = if binding_variant_lit.is_some() {
        quote! { crate::builtin::RuntimeBuiltinFuture }
    } else {
        quote! { runmat_builtins::BuiltinFuture }
    };
    let wrapper_args = if binding_variant_lit.is_some() {
        quote! {
            let input = invocation.into_continuation_input();
        }
    } else {
        quote! { let args = args.to_vec(); }
    };

    let result_adaptation = if binding_variant_lit.is_some() {
        quote! { crate::builtin::CanonicalBuiltinOutput::into_rooted_sequence(value) }
    } else if output_sequence {
        quote! { Ok(value) }
    } else {
        quote! { Ok(runmat_value::Value::from(value)) }
    };
    let arity_check = quote! {
        let arity_error = if #is_last_variadic {
            (args.len() < #param_len - 1).then(|| format!(
                "expected at least {} args, got {}",
                #param_len - 1,
                args.len()
            ))
        } else {
            (args.len() != #param_len).then(|| format!(
                "expected {} args, got {}",
                #param_len,
                args.len()
            ))
        };
    };
    let body_after_arity = quote! {
        #(#conv_stmts)*
        let value = #call_expr;
        #result_adaptation
    };
    let invocation_body = quote! {
        #arity_check
        if let Some(message) = arity_error {
            return Err(std::convert::From::from(message));
        }
        #body_after_arity
    };
    let owned_args_binding = if param_len == 0 {
        quote! { let _args = input.args; }
    } else {
        quote! { let args = input.args; }
    };
    let owned_helpers = if binding_variant_lit.is_some() {
        quote! {
            async fn #body_ident(
                input: crate::builtin::BuiltinContinuationInput,
            ) -> ::std::result::Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
                let _argument_context = input.context.clone();
                // A variadic body consumes its argument range. Keep the same
                // original roots until its returned values have acquired the
                // canonical output owner, including direct embedding calls.
                let _argument_owner = input.args.fork();
                #owned_args_binding
                #owned_context_binding
                #owned_requested_outputs_binding
                #owned_call_observation_binding
                #body_after_arity
            }

            fn #layout_ident() -> ::std::result::Result<u64, crate::context::host_allocation::HostAllocationError> {
                crate::builtin::continuation_layout_for(#body_ident)
            }
        }
    } else {
        quote! {}
    };
    let wrapper = if binding_variant_lit.is_some() {
        quote! {
            fn #wrapper_ident(#wrapper_input) -> #wrapper_return {
                #wrapper_args
                let input = match input {
                    Ok(input) => input,
                    Err(error) => {
                        return crate::builtin::rejected_builtin_future(error);
                    }
                };
                let args = &input.args;
                #arity_check
                if let Some(message) = arity_error {
                    return crate::builtin::rejected_builtin_future(std::convert::From::from(message));
                }
                // Keep the target-sized async state out of this wrapper's
                // caller stack. The factory boundary admits the exact named
                // body type before constructing and erasing it once.
                let allocation_context = input.context.clone();
                crate::builtin::admitted_builtin_future_with(
                    allocation_context,
                    #name_lit,
                    move |_| #body_ident(input),
                )
            }
        }
    } else {
        quote! {
            fn #wrapper_ident(#wrapper_input) -> #wrapper_return {
                #wrapper_args
                #context_binding
                #requested_outputs_binding
                #call_observation_binding
                Box::pin(async move {
                    #invocation_body
                })
            }
        }
    };

    // Prepare tokens for defaults and options
    let default_category = syn::LitStr::new("general", proc_macro2::Span::call_site());
    let default_summary =
        syn::LitStr::new("Runtime builtin function", proc_macro2::Span::call_site());

    let category_tok: proc_macro2::TokenStream = match &category_lit {
        Some(syn::Lit::Str(ls)) => quote! { #ls },
        _ => quote! { #default_category },
    };
    let summary_tok: proc_macro2::TokenStream = match &summary_lit {
        Some(syn::Lit::Str(ls)) => quote! { #ls },
        _ => quote! { #default_summary },
    };

    fn opt_tok(lit: &Option<syn::Lit>) -> proc_macro2::TokenStream {
        if let Some(syn::Lit::Str(ls)) = lit {
            quote! { Some(#ls) }
        } else {
            quote! { None }
        }
    }
    let category_opt_tok = opt_tok(&category_lit);
    let summary_opt_tok = opt_tok(&summary_lit);
    let keywords_opt_tok = opt_tok(&keywords_lit);
    let errors_opt_tok = opt_tok(&errors_lit);
    let related_opt_tok = opt_tok(&related_lit);
    let introduced_opt_tok = opt_tok(&introduced_lit);
    let status_opt_tok = opt_tok(&status_lit);
    let examples_opt_tok = opt_tok(&examples_lit);

    let accel_tokens: Vec<proc_macro2::TokenStream> = accel_values
        .iter()
        .map(|mode| match mode.as_str() {
            "unary" => quote! { runmat_builtins::AccelTag::Unary },
            "binary" => quote! { runmat_builtins::AccelTag::Elementwise },
            "elementwise" => quote! { runmat_builtins::AccelTag::Elementwise },
            "reduction" => quote! { runmat_builtins::AccelTag::Reduction },
            "matmul" => quote! { runmat_builtins::AccelTag::MatMul },
            "transpose" => quote! { runmat_builtins::AccelTag::Transpose },
            "array_construct" => quote! { runmat_builtins::AccelTag::ArrayConstruct },
            _ => quote! {},
        })
        .filter(|ts| !ts.is_empty())
        .collect();
    let accel_slice = if accel_tokens.is_empty() {
        quote! { &[] as &[runmat_builtins::AccelTag] }
    } else {
        quote! { &[#(#accel_tokens),*] }
    };
    let type_resolver_expr = if let Some(path) = type_resolver_ctx_path.as_ref() {
        quote! { Some(runmat_builtins::type_resolver_kind_ctx(#path)) }
    } else if let Some(path) = type_resolver_path.as_ref() {
        quote! { Some(runmat_builtins::type_resolver_kind_ctx(#path)) }
    } else {
        quote! { None }
    };
    let sink_bool = sink_flag;
    let suppress_auto_output_bool = suppress_auto_output_flag;
    let execution_stack_expr = if process_execution_stack {
        quote! { runmat_builtins::ExecutionStackRequirement::Process }
    } else {
        quote! { runmat_builtins::ExecutionStackRequirement::Any }
    };
    let required_capability_tokens =
        required_capabilities
            .iter()
            .map(|capability| match capability.as_str() {
                "host_runtime" => quote! { runmat_types::CapabilityRequirement::HostRuntime },
                "filesystem" => quote! { runmat_types::CapabilityRequirement::Filesystem },
                "network" => quote! { runmat_types::CapabilityRequirement::Network },
                "user_interface" => quote! { runmat_types::CapabilityRequirement::UserInterface },
                "accelerator" => quote! { runmat_types::CapabilityRequirement::Accelerator },
                "native_code" => quote! { runmat_types::CapabilityRequirement::NativeCode },
                "foreign_runtime" => quote! { runmat_types::CapabilityRequirement::ForeignRuntime },
                "parallel_runtime" => {
                    quote! { runmat_types::CapabilityRequirement::ParallelRuntime }
                }
                "distributed_runtime" => {
                    quote! { runmat_types::CapabilityRequirement::DistributedRuntime }
                }
                other => panic!("unknown builtin capability {other:?}"),
            });
    let descriptor_expr = if let Some(path) = descriptor_path.as_ref() {
        quote! { Some(&#path) }
    } else {
        quote! { None }
    };
    let extensions_expr = if let Some(path) = extensions_path.as_ref() {
        quote! { &#path }
    } else {
        quote! { &[] }
    };
    let integer_capabilities_expr = if let Some(path) = integer_capabilities_path.as_ref() {
        quote! { &#path }
    } else {
        quote! { &[] }
    };
    let integer_audit_expr = if let Some(path) = integer_audit_path.as_ref() {
        quote! { Some(&#path) }
    } else {
        quote! { None }
    };

    let builtin_expr = quote! {
        runmat_builtins::BuiltinFunction::new(
            #name_str,
            #summary_tok,
            #category_tok,
            "",
            "",
            vec![#(#inferred_param_types),*],
            #inferred_return_type,
            #type_resolver_expr,
            #wrapper_ident,
            #accel_slice,
            #sink_bool,
            #suppress_auto_output_bool,
        )
        .with_execution_stack(#execution_stack_expr)
        .with_required_capabilities(&[#(#required_capability_tokens),*])
        .with_descriptor_option(#descriptor_expr)
        .with_extensions(#extensions_expr)
        .with_integer_capabilities(#integer_capabilities_expr)
        .with_integer_audit(#integer_audit_expr)
    };

    if binding_variant_lit.is_some()
        && (category_lit.is_some()
            || summary_lit.is_some()
            || keywords_lit.is_some()
            || errors_lit.is_some()
            || related_lit.is_some()
            || introduced_lit.is_some()
            || status_lit.is_some()
            || examples_lit.is_some()
            || !accel_values.is_empty()
            || type_resolver_path.is_some()
            || type_resolver_ctx_path.is_some()
            || descriptor_path.is_some()
            || extensions_path.is_some()
            || integer_capabilities_path.is_some()
            || integer_audit_path.is_some()
            || sink_flag
            || suppress_auto_output_flag
            || process_execution_stack)
    {
        panic!(
            "catalog-backed runtime bindings may declare only name, binding_variant, and builtin_path"
        );
    }

    let doc_expr = quote! {
        runmat_builtins::BuiltinDoc {
            name: #name_str,
            category: #category_opt_tok,
            summary: #summary_opt_tok,
            keywords: #keywords_opt_tok,
            errors: #errors_opt_tok,
            related: #related_opt_tok,
            introduced: #introduced_opt_tok,
            status: #status_opt_tok,
            examples: #examples_opt_tok,
        }
    };

    let builtin_path_lit =
        builtin_path_lit.expect("runtime_builtin requires `builtin_path = \"...\"`");
    let builtin_path_text = builtin_path_lit.clone();
    let builtin_path: syn::Path = syn::parse_str(&builtin_path_lit.value())
        .expect("runtime_builtin `builtin_path` must be a valid path");
    let helper_ident = format_ident!("__runmat_wasm_register_builtin_{}", ident);
    let builtin_expr_helper = builtin_expr.clone();
    let doc_expr_helper = doc_expr.clone();
    let provenance_authority = if binding_variant_lit.is_some() {
        quote! { runmat_builtins::BuiltinImplementationAuthority::CanonicalBinding }
    } else {
        quote! { runmat_builtins::BuiltinImplementationAuthority::LegacyFunction }
    };
    let provenance_variant = if let Some(variant) = binding_variant_lit.as_ref() {
        quote! { Some(#variant) }
    } else {
        quote! { None }
    };
    let provenance_expr = quote! {
        runmat_builtins::BuiltinImplementationProvenance {
            name: #name_str,
            binding_variant: #provenance_variant,
            source_file: file!(),
            module_path: module_path!(),
            function: stringify!(#ident),
            builtin_path: #builtin_path_text,
            authority: #provenance_authority,
        }
    };
    let provenance_expr_helper = provenance_expr.clone();
    let manifest_expr = quote! {
        runmat_builtins::RegistrationManifestEntry {
            kind: runmat_builtins::RegistrationKind::Builtin,
            declaration: #name_str,
            variant: #provenance_variant,
            builtin_path: #builtin_path_text,
        }
    };
    let manifest_expr_helper = manifest_expr.clone();
    let manifest_variant = binding_variant_lit.as_ref().map(|value| value.value());
    let (wasm_helper, register_native) = if let Some(variant) = binding_variant_lit {
        let native_symbol = runmat_builtins::native_binding_symbol(&name_str, &variant.value());
        let native_symbol = syn::LitStr::new(&native_symbol, proc_macro2::Span::call_site());
        let native_binding_ident = format_ident!(
            "__RUNMAT_NATIVE_BINDING_{}",
            ident.to_string().to_ascii_uppercase()
        );
        let binding_expr = quote! {
            crate::builtin::RuntimeBuiltinBinding::new(
                runmat_builtins::BuiltinBindingIdentity {
                    builtin: runmat_builtins::BuiltinCatalogIdentity { name: #name_str },
                    variant: #variant,
                },
                #wrapper_ident,
            ).with_continuation_layout(#layout_ident).with_provenance(&#provenance_expr)
        };
        (
            quote! {
                #[cfg(target_arch = "wasm32")]
                #[allow(non_snake_case)]
                pub(crate) fn #helper_ident() {
                    crate::builtin::wasm_registry::submit(#binding_expr);
                    runmat_builtins::wasm_registry::submit_builtin_implementation_provenance(#provenance_expr_helper);
                    runmat_builtins::wasm_registry::submit_registration_manifest_entry(#manifest_expr_helper);
                }
            },
            quote! {
                #[cfg(not(target_arch = "wasm32"))]
                runmat_builtins::inventory::submit! { #binding_expr }
                #[cfg(not(target_arch = "wasm32"))]
                runmat_builtins::inventory::submit! { #provenance_expr }
                #[cfg(not(target_arch = "wasm32"))]
                runmat_builtins::inventory::submit! { #manifest_expr }
                // A package's unit-test harness can link both its cfg(test)
                // copy and a normal dependency copy of the runtime. Only the
                // latter should publish the process-wide AOT ABI symbols.
                #[cfg(all(not(target_arch = "wasm32"), not(test)))]
                #[export_name = #native_symbol]
                static #native_binding_ident: crate::builtin::RuntimeBuiltinBinding = #binding_expr;
            },
        )
    } else {
        (
            quote! {
                #[cfg(target_arch = "wasm32")]
                #[allow(non_snake_case)]
                pub(crate) fn #helper_ident() {
                    runmat_builtins::wasm_registry::submit_builtin_function(#builtin_expr_helper);
                    runmat_builtins::wasm_registry::submit_builtin_doc(#doc_expr_helper);
                    runmat_builtins::wasm_registry::submit_builtin_implementation_provenance(#provenance_expr_helper);
                    runmat_builtins::wasm_registry::submit_registration_manifest_entry(#manifest_expr_helper);
                }
            },
            quote! {
                #[cfg(not(target_arch = "wasm32"))]
                runmat_builtins::inventory::submit! { #builtin_expr }
                #[cfg(not(target_arch = "wasm32"))]
                runmat_builtins::inventory::submit! { #doc_expr }
                #[cfg(not(target_arch = "wasm32"))]
                runmat_builtins::inventory::submit! { #provenance_expr }
                #[cfg(not(target_arch = "wasm32"))]
                runmat_builtins::inventory::submit! { #manifest_expr }
            },
        )
    };
    append_wasm_block(
        quote! { #builtin_path::#helper_ident(); },
        "builtin",
        &name_str,
        manifest_variant.as_deref(),
        &builtin_path_lit.value(),
    );

    TokenStream::from(quote! {
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        #func
        #owned_helpers
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        #wrapper
        #wasm_helper
        #register_native
    })
}

/// Attribute used to declare a runtime constant.
///
/// Example:
/// ```rust,ignore
/// use runmat_macros::runtime_constant;
/// use runmat_value::Value;
///
/// #[runtime_constant(name = "pi", value = std::f64::consts::PI)]
/// const PI_CONSTANT: ();
/// ```
///
/// This registers the constant with the `runmat-builtins` inventory
/// so the runtime can discover it at start-up.
#[proc_macro_attribute]
pub fn runtime_constant(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as AttributeArgs);
    let mut name_lit: Option<Lit> = None;
    let mut value_expr: Option<Expr> = None;
    let mut builtin_path_lit: Option<LitStr> = None;

    for arg in args {
        match arg {
            NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) => {
                if path.is_ident("name") {
                    name_lit = Some(lit);
                } else if path.is_ident("builtin_path") {
                    if let Lit::Str(ls) = lit {
                        builtin_path_lit = Some(ls);
                    } else {
                        panic!("builtin_path must be a string literal");
                    }
                } else {
                    panic!("Unknown attribute parameter: {}", quote!(#path));
                }
            }
            NestedMeta::Meta(Meta::Path(path)) if path.is_ident("value") => {
                panic!("value parameter requires assignment: value = expression");
            }
            NestedMeta::Lit(lit) => {
                // This handles the case where value is provided as a literal
                value_expr = Some(syn::parse_quote!(#lit));
            }
            _ => panic!("Invalid attribute syntax"),
        }
    }

    let name = match name_lit {
        Some(Lit::Str(s)) => s.value(),
        _ => panic!("name parameter must be a string literal"),
    };

    let value = value_expr.unwrap_or_else(|| {
        panic!("value parameter is required");
    });

    let builtin_path_lit =
        builtin_path_lit.expect("runtime_constant requires `builtin_path = \"...\"` argument");
    let builtin_path: syn::Path = syn::parse_str(&builtin_path_lit.value())
        .expect("runtime_constant `builtin_path` must be a valid path");
    let item = parse_macro_input!(input as syn::Item);

    let constant_expr = quote! {
        runmat_builtins::ConstantRegistration {
            constant: runmat_builtins::Constant {
                name: #name,
                value: #value,
            },
            source_file: file!(),
            module_path: module_path!(),
            builtin_path: #builtin_path_lit,
        }
    };

    let helper_ident = helper_ident_from_name("__runmat_wasm_register_const_", &name);
    let constant_expr_helper = constant_expr.clone();
    let manifest_expr = quote! {
        runmat_builtins::RegistrationManifestEntry {
            kind: runmat_builtins::RegistrationKind::Constant, declaration: #name, variant: None, builtin_path: #builtin_path_lit,
        }
    };
    let manifest_expr_helper = manifest_expr.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            runmat_builtins::wasm_registry::submit_constant(#constant_expr_helper);
            runmat_builtins::wasm_registry::submit_registration_manifest_entry(#manifest_expr_helper);
        }
    };
    let register_native = quote! {
        #[cfg(not(target_arch = "wasm32"))]
        #[allow(non_upper_case_globals)]
        runmat_builtins::inventory::submit! { #constant_expr }
        #[cfg(not(target_arch = "wasm32"))]
        runmat_builtins::inventory::submit! { #manifest_expr }
    };
    append_wasm_block(
        quote! { #builtin_path::#helper_ident(); },
        "constant",
        &name,
        None,
        &builtin_path_lit.value(),
    );

    TokenStream::from(quote! {
        #item
        #wasm_helper
        #register_native
    })
}

struct RegisterConstantArgs {
    name: LitStr,
    value: Expr,
    builtin_path: LitStr,
}

impl syn::parse::Parse for RegisterConstantArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name: LitStr = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let value: Expr = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let builtin_path: LitStr = input.parse()?;
        if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
        }
        Ok(RegisterConstantArgs {
            name,
            value,
            builtin_path,
        })
    }
}

#[proc_macro]
pub fn register_constant(input: TokenStream) -> TokenStream {
    let RegisterConstantArgs {
        name,
        value,
        builtin_path,
    } = parse_macro_input!(input as RegisterConstantArgs);
    let builtin_path_lit = builtin_path.clone();
    let constant_expr = quote! {
        runmat_builtins::ConstantRegistration {
            constant: runmat_builtins::Constant {
                name: #name,
                value: #value,
            },
            source_file: file!(),
            module_path: module_path!(),
            builtin_path: #builtin_path_lit,
        }
    };
    let helper_ident = helper_ident_from_name("__runmat_wasm_register_const_", &name.value());
    let builtin_path: syn::Path = syn::parse_str(&builtin_path.value())
        .expect("register_constant `builtin_path` must be a valid path");
    let constant_expr_helper = constant_expr.clone();
    let manifest_expr = quote! {
        runmat_builtins::RegistrationManifestEntry {
            kind: runmat_builtins::RegistrationKind::Constant, declaration: #name, variant: None, builtin_path: #builtin_path_lit,
        }
    };
    let manifest_expr_helper = manifest_expr.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            runmat_builtins::wasm_registry::submit_constant(#constant_expr_helper);
            runmat_builtins::wasm_registry::submit_registration_manifest_entry(#manifest_expr_helper);
        }
    };
    append_wasm_block(
        quote! { #builtin_path::#helper_ident(); },
        "constant",
        &name.value(),
        None,
        &builtin_path_lit.value(),
    );
    TokenStream::from(quote! {
        #wasm_helper
        #[cfg(not(target_arch = "wasm32"))]
        runmat_builtins::inventory::submit! { #constant_expr }
        #[cfg(not(target_arch = "wasm32"))]
        runmat_builtins::inventory::submit! { #manifest_expr }
    })
}

struct RegisterSpecAttrArgs {
    spec_expr: Option<Expr>,
    builtin_path: Option<LitStr>,
    semantics: Option<Expr>,
    structural: Option<Expr>,
    epilogue: Option<Expr>,
    reduction: Option<Expr>,
    host_resources: Option<Expr>,
}

#[cfg(test)]
#[path = "spec_registration_tests.rs"]
mod spec_registration_tests;

impl Parse for RegisterSpecAttrArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut spec_expr = None;
        let mut builtin_path = None;
        let mut semantics = None;
        let mut structural = None;
        let mut epilogue = None;
        let mut reduction = None;
        let mut host_resources = None;
        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            if ident == "spec" {
                spec_expr = Some(input.parse()?);
            } else if ident == "semantics" {
                semantics = Some(input.parse()?);
            } else if ident == "structural" {
                if structural.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        "duplicate structural declaration",
                    ));
                }
                structural = Some(input.parse()?);
            } else if ident == "epilogue" {
                if epilogue.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        "duplicate epilogue declaration",
                    ));
                }
                epilogue = Some(input.parse()?);
            } else if ident == "reduction" {
                if reduction.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        "duplicate reduction declaration",
                    ));
                }
                reduction = Some(input.parse()?);
            } else if ident == "host_resources" {
                if host_resources.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        "duplicate host resource declaration",
                    ));
                }
                host_resources = Some(input.parse()?);
            } else if ident == "builtin_path" {
                let lit: LitStr = input.parse()?;
                builtin_path = Some(lit);
            } else {
                return Err(syn::Error::new(ident.span(), "unknown attribute argument"));
            }
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            }
        }
        Ok(Self {
            spec_expr,
            builtin_path,
            semantics,
            structural,
            epilogue,
            reduction,
            host_resources,
        })
    }
}

#[proc_macro_attribute]
pub fn register_gpu_spec(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterSpecAttrArgs);
    let RegisterSpecAttrArgs {
        spec_expr,
        builtin_path,
        semantics,
        structural,
        epilogue,
        reduction,
        host_resources,
    } = args;
    if semantics.is_some()
        || structural.is_some()
        || epilogue.is_some()
        || reduction.is_some()
        || host_resources.is_some()
    {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "semantics and structural operations belong to a fusion registration",
        )
        .to_compile_error()
        .into();
    }
    let item_const = parse_macro_input!(item as ItemConst);
    let spec_tokens = spec_expr.map(|expr| quote! { #expr }).unwrap_or_else(|| {
        let ident = &item_const.ident;
        quote! { #ident }
    });
    let spec_for_native = spec_tokens.clone();
    let builtin_path_lit =
        builtin_path.expect("register_gpu_spec requires `builtin_path = \"...\"` argument");
    let builtin_path: syn::Path = syn::parse_str(&builtin_path_lit.value())
        .expect("register_gpu_spec `builtin_path` must be a valid path");
    let helper_ident = format_ident!(
        "__runmat_wasm_register_gpu_spec_{}",
        item_const.ident.to_string()
    );
    let declaration = item_const.ident.to_string();
    let spec_tokens_helper = spec_tokens.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            crate::builtins::common::spec::wasm_registry::submit_gpu_spec(
                crate::builtins::common::spec::GpuSpecRegistration {
                    spec: &#spec_tokens_helper, builtin_path: #builtin_path_lit, declaration: #declaration,
                    source_file: file!(), module_path: module_path!(),
                }
            );
            runmat_builtins::wasm_registry::submit_registration_manifest_entry(
                runmat_builtins::RegistrationManifestEntry {
                    kind: runmat_builtins::RegistrationKind::GpuSpec, declaration: #declaration, variant: None, builtin_path: #builtin_path_lit,
                }
            );
        }
    };
    append_wasm_block(
        quote! { #builtin_path::#helper_ident(); },
        "gpu_spec",
        &declaration,
        None,
        &builtin_path_lit.value(),
    );
    let expanded = quote! {
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        #item_const
        #wasm_helper
        #[cfg(not(target_arch = "wasm32"))]
        inventory::submit! {
            crate::builtins::common::spec::GpuSpecInventory {
                spec: &#spec_for_native,
                builtin_path: #builtin_path_lit,
                declaration: #declaration,
                source_file: file!(),
                module_path: module_path!(),
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        runmat_builtins::inventory::submit! {
            runmat_builtins::RegistrationManifestEntry {
                kind: runmat_builtins::RegistrationKind::GpuSpec, declaration: #declaration, variant: None, builtin_path: #builtin_path_lit,
            }
        }
    };
    expanded.into()
}

#[proc_macro_attribute]
pub fn register_fusion_spec(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterSpecAttrArgs);
    let RegisterSpecAttrArgs {
        spec_expr,
        builtin_path,
        semantics,
        structural,
        epilogue,
        reduction,
        host_resources,
    } = args;
    let host_resources = host_resources
        .map(|value| quote! { Some(#value) })
        .unwrap_or_else(|| quote! { None });
    let reduction = reduction
        .map(|value| quote! { Some(&#value) })
        .unwrap_or_else(|| quote! { None });
    let epilogue = epilogue
        .map(|value| quote! { Some(#value) })
        .unwrap_or_else(|| quote! { None });
    let structural = structural
        .map(|value| quote! { Some(#value) })
        .unwrap_or_else(|| quote! { None });
    let semantics = semantics
        .map(|value| quote! { Some(&#value) })
        .unwrap_or_else(|| quote! { None });
    let item_const = parse_macro_input!(item as ItemConst);
    let spec_tokens = spec_expr.map(|expr| quote! { #expr }).unwrap_or_else(|| {
        let ident = &item_const.ident;
        quote! { #ident }
    });
    let spec_for_native = spec_tokens.clone();
    let builtin_path_lit =
        builtin_path.expect("register_fusion_spec requires `builtin_path = \"...\"` argument");
    let builtin_path: syn::Path = syn::parse_str(&builtin_path_lit.value())
        .expect("register_fusion_spec `builtin_path` must be a valid path");
    let helper_ident = format_ident!(
        "__runmat_wasm_register_fusion_spec_{}",
        item_const.ident.to_string()
    );
    let declaration = item_const.ident.to_string();
    let spec_tokens_helper = spec_tokens.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            crate::builtins::common::spec::wasm_registry::submit_fusion_spec(
                crate::builtins::common::spec::FusionSpecRegistration {
                    spec: &#spec_tokens_helper, builtin_path: #builtin_path_lit, declaration: #declaration,
                    semantics: #semantics,
                    structural: #structural,
                    epilogue: #epilogue,
                    reduction: #reduction,
                    host_resources: #host_resources,
                    source_file: file!(), module_path: module_path!(),
                }
            );
            runmat_builtins::wasm_registry::submit_registration_manifest_entry(
                runmat_builtins::RegistrationManifestEntry {
                    kind: runmat_builtins::RegistrationKind::FusionSpec, declaration: #declaration, variant: None, builtin_path: #builtin_path_lit,
                }
            );
        }
    };
    append_wasm_block(
        quote! { #builtin_path::#helper_ident(); },
        "fusion_spec",
        &declaration,
        None,
        &builtin_path_lit.value(),
    );
    let expanded = quote! {
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        #item_const
        #wasm_helper
        #[cfg(not(target_arch = "wasm32"))]
        inventory::submit! {
            crate::builtins::common::spec::FusionSpecInventory {
                spec: &#spec_for_native,
                semantics: #semantics,
                structural: #structural,
                epilogue: #epilogue,
                reduction: #reduction,
                host_resources: #host_resources,
                builtin_path: #builtin_path_lit,
                declaration: #declaration,
                source_file: file!(),
                module_path: module_path!(),
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        runmat_builtins::inventory::submit! {
            runmat_builtins::RegistrationManifestEntry {
                kind: runmat_builtins::RegistrationKind::FusionSpec, declaration: #declaration, variant: None, builtin_path: #builtin_path_lit,
            }
        }
    };
    expanded.into()
}

fn append_wasm_block(
    block: proc_macro2::TokenStream,
    kind: &str,
    declaration: &str,
    variant: Option<&str>,
    builtin_path: &str,
) {
    if !should_generate_wasm_registry() {
        return;
    }
    let path = match wasm_registry_path() {
        Some(p) => p,
        None => return,
    };
    let _guard = wasm_registry_lock().lock().unwrap();
    initialize_registry_file(path);
    let mut contents = fs::read_to_string(path).expect("failed to read wasm registry file");
    for value in [kind, declaration, variant.unwrap_or("-"), builtin_path] {
        assert!(
            !value.contains(['\t', '\n', '\r']),
            "WASM registry manifest fields cannot contain control separators"
        );
    }
    let (variant_presence, variant_value) = variant.map_or(("none", ""), |value| ("some", value));
    let insertion = format!(
        "    // @runmat-wasm-registration-v2\t{kind}\t{declaration}\t{variant_presence}\t{variant_value}\t{builtin_path}\n    {block}\n"
    );
    if let Some(pos) = contents.rfind('}') {
        contents.insert_str(pos, &insertion);
    } else {
        contents.push_str(&insertion);
        contents.push_str("}\n");
    }
    fs::write(path, contents).expect("failed to update wasm registry file");
}

fn wasm_registry_path() -> Option<&'static PathBuf> {
    WASM_REGISTRY_PATH
        .get_or_init(workspace_registry_path)
        .as_ref()
}

fn wasm_registry_lock() -> &'static Mutex<()> {
    WASM_REGISTRY_LOCK.get_or_init(|| Mutex::new(()))
}

fn initialize_registry_file(path: &Path) {
    WASM_REGISTRY_INIT.get_or_init(|| {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if fs::metadata(path)
            .map(|metadata| metadata.len() > 0)
            .unwrap_or(false)
        {
            return;
        }
        const HEADER: &str = "// @generated by `scripts/regenerate-wasm-registry.sh`\n\
pub const REGISTRY_COMPLETE: bool = false;\n\
pub const REGISTRY_SOURCE_FINGERPRINT: &str = \"missing-build-script\";\n\
pub const REGISTRY_BUILD_CONFIGURATION: &str = \"missing-build-script\";\n\
pub const REGISTRY_ENTRY_COUNT: usize = 0;\n\n\
pub const REGISTRY_MANIFEST_DIGEST: &str = \"0000000000000000000000000000000000000000000000000000000000000000\";\n\
pub const REGISTRY_BUILTIN_COUNT: usize = 0;\n\
pub const REGISTRY_CONSTANT_COUNT: usize = 0;\n\
pub const REGISTRY_GPU_SPEC_COUNT: usize = 0;\n\
pub const REGISTRY_FUSION_SPEC_COUNT: usize = 0;\n\n\
pub fn register_all() {\n}\n";
        fs::write(path, HEADER).expect("failed to initialize wasm registry file");
    });
}

fn should_generate_wasm_registry() -> bool {
    // Only write to the registry file when explicitly requested. Writing on every
    // build would keep the included generated source changing and force repeated
    // runmat-runtime recompiles. The regeneration script sets this together with
    // RUNMAT_WASM_REGISTRY_OUT so helper calls are written to a temporary file and
    // copied into place only after cargo succeeds.
    matches!(
        std::env::var("RUNMAT_GENERATE_WASM_REGISTRY"),
        Ok(ref value) if value == "1"
    )
}

fn workspace_registry_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("RUNMAT_WASM_REGISTRY_OUT") {
        return Some(PathBuf::from(path));
    }
    let mut dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    loop {
        if dir.join("Cargo.lock").exists() {
            return Some(
                dir.join("crates")
                    .join("runmat-runtime")
                    .join("src")
                    .join("builtins")
                    .join("generated_wasm_registry.rs"),
            );
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn helper_ident_from_name(prefix: &str, name: &str) -> proc_macro2::Ident {
    let mut sanitized = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    format_ident!("{}{}", prefix, sanitized)
}

/// Smart type inference from Rust types to our enhanced Type enum
fn builtin_parameter_conversion(
    canonical: bool,
    index: usize,
    ident: &syn::Ident,
    ty: &syn::Type,
) -> proc_macro2::TokenStream {
    if canonical
        && matches!(ty, syn::Type::Path(path)
        if path.path.segments.last().is_some_and(|segment| segment.ident == "Utf8Text"))
    {
        quote! {
            let #ident: #ty = crate::call::argument_text::convert(
                &args[#index], _argument_context.as_ref(),
            )?;
        }
    } else {
        quote! { let #ident: #ty = std::convert::TryInto::try_into(&args[#index])?; }
    }
}

fn infer_builtin_type(ty: &syn::Type) -> proc_macro2::TokenStream {
    use syn::Type;

    match ty {
        // Basic primitive types
        Type::Path(type_path) => {
            if let Some(ident) = type_path.path.get_ident() {
                match ident.to_string().as_str() {
                    "i32" | "i64" | "isize" => quote! { runmat_builtins::Type::Int },
                    "f32" | "f64" => quote! { runmat_builtins::Type::Num },
                    "bool" => quote! { runmat_builtins::Type::Bool },
                    "String" => quote! { runmat_builtins::Type::String },
                    _ => infer_complex_type(type_path),
                }
            } else {
                infer_complex_type(type_path)
            }
        }

        // Reference types like &str, &Value, &Matrix
        Type::Reference(type_ref) => match type_ref.elem.as_ref() {
            Type::Path(type_path) => {
                if let Some(ident) = type_path.path.get_ident() {
                    match ident.to_string().as_str() {
                        "str" => quote! { runmat_builtins::Type::String },
                        _ => infer_builtin_type(&type_ref.elem),
                    }
                } else {
                    infer_builtin_type(&type_ref.elem)
                }
            }
            _ => infer_builtin_type(&type_ref.elem),
        },

        // Slice types like &[Value], &[f64]
        Type::Slice(type_slice) => {
            let element_type = infer_builtin_type(&type_slice.elem);
            quote! { runmat_builtins::Type::Cell {
                element_type: Some(Box::new(#element_type)),
                length: None
            } }
        }

        // Array types like [f64; N]
        Type::Array(type_array) => {
            let element_type = infer_builtin_type(&type_array.elem);
            // Try to extract length if it's a literal
            if let syn::Expr::Lit(expr_lit) = &type_array.len {
                if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                    if let Ok(length) = lit_int.base10_parse::<usize>() {
                        return quote! { runmat_builtins::Type::Cell {
                            element_type: Some(Box::new(#element_type)),
                            length: Some(#length)
                        } };
                    }
                }
            }
            // Fallback to unknown length
            quote! { runmat_builtins::Type::Cell {
                element_type: Some(Box::new(#element_type)),
                length: None
            } }
        }

        // Generic or complex types
        _ => quote! { runmat_builtins::Type::Unknown },
    }
}

/// Infer types for complex path types like Result<T, E>, Option<T>, Matrix, Value
fn infer_complex_type(type_path: &syn::TypePath) -> proc_macro2::TokenStream {
    let path_str = quote! { #type_path }.to_string();

    // Handle common patterns
    if path_str.contains("Matrix") || path_str.contains("Tensor") {
        quote! { runmat_builtins::Type::tensor() }
    } else if path_str.contains("Value") {
        quote! { runmat_builtins::Type::Unknown } // Value can be anything
    } else if path_str.starts_with("Result") {
        // Extract the Ok type from Result<T, E>
        if let syn::PathArguments::AngleBracketed(angle_bracketed) =
            &type_path.path.segments.last().unwrap().arguments
        {
            if let Some(syn::GenericArgument::Type(ty)) = angle_bracketed.args.first() {
                return infer_builtin_type(ty);
            }
        }
        quote! { runmat_builtins::Type::Unknown }
    } else if path_str.starts_with("Option") {
        // Extract the Some type from Option<T>
        if let syn::PathArguments::AngleBracketed(angle_bracketed) =
            &type_path.path.segments.last().unwrap().arguments
        {
            if let Some(syn::GenericArgument::Type(ty)) = angle_bracketed.args.first() {
                return infer_builtin_type(ty);
            }
        }
        quote! { runmat_builtins::Type::Unknown }
    } else if path_str.starts_with("Vec") {
        // Extract element type from Vec<T>
        if let syn::PathArguments::AngleBracketed(angle_bracketed) =
            &type_path.path.segments.last().unwrap().arguments
        {
            if let Some(syn::GenericArgument::Type(ty)) = angle_bracketed.args.first() {
                let element_type = infer_builtin_type(ty);
                return quote! { runmat_builtins::Type::Cell {
                    element_type: Some(Box::new(#element_type)),
                    length: None
                } };
            }
        }
        quote! { runmat_builtins::Type::cell() }
    } else {
        // Unknown type
        quote! { runmat_builtins::Type::Unknown }
    }
}
