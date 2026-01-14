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
    let mut sink_flag = false;
    let mut suppress_auto_output_flag = false;
    for arg in args {
        if let NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) = arg {
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
            } else if path.is_ident("builtin_path") {
                if let Lit::Str(ls) = lit {
                    builtin_path_lit = Some(ls);
                } else {
                    panic!("builtin_path must be a string literal");
                }
            } else {
                // Gracefully ignore unknown parameters for better IDE experience
            }
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
    let param_len = param_idents.len();

    // Infer parameter types for BuiltinFunction
    let inferred_param_types: Vec<proc_macro2::TokenStream> =
        param_types.iter().map(infer_builtin_type).collect();

    // Infer return type for BuiltinFunction
    let inferred_return_type = match &func.sig.output {
        syn::ReturnType::Default => quote! { runmat_builtins::Type::Void },
        syn::ReturnType::Type(_, ty) => infer_builtin_type(ty),
    };

    // Detect if last parameter is variadic Vec<Value>
    let is_last_variadic = param_types
        .last()
        .map(|ty| {
            // crude detection: type path starts with Vec and inner type is runmat_builtins::Value or Value
            if let syn::Type::Path(tp) = ty {
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

    let conv_stmts: Vec<proc_macro2::TokenStream> = if is_last_variadic && param_len > 0 {
        let mut stmts = Vec::new();
        // Convert fixed params (all but last)
        for (i, (ident, ty)) in param_idents
            .iter()
            .zip(param_types.iter())
            .enumerate()
            .take(param_len - 1)
        {
            stmts.push(quote! { let #ident : #ty = std::convert::TryInto::try_into(&args[#i])?; });
        }
        // Collect the rest into Vec<Value>
        let last_ident = &param_idents[param_len - 1];
        stmts.push(quote! {
            let #last_ident : Vec<runmat_builtins::Value> = {
                let mut v = Vec::new();
                for j in (#param_len-1)..args.len() {
                    let item : runmat_builtins::Value = std::convert::TryInto::try_into(&args[j])?;
                    v.push(item);
                }
                v
            };
        });
        stmts
    } else {
        param_idents
            .iter()
            .zip(param_types.iter())
            .enumerate()
            .map(|(i, (ident, ty))| {
                quote! { let #ident : #ty = std::convert::TryInto::try_into(&args[#i])?; }
            })
            .collect()
    };

    let wrapper = quote! {
        fn #wrapper_ident(args: &[runmat_builtins::Value]) -> Result<runmat_builtins::Value, runmat_async::RuntimeControlFlow> {
            #![allow(unused_variables)]
            if #is_last_variadic {
                if args.len() < #param_len - 1 { return Err(format!("expected at least {} args, got {}", #param_len - 1, args.len()).into()); }
            } else {
                if args.len() != #param_len { return Err(format!("expected {} args, got {}", #param_len, args.len()).into()); }
            }
            #(#conv_stmts)*
            let res = match #ident(#(#param_idents),*) {
                Ok(value) => value,
                Err(message) => {
                    if message == crate::interaction::PENDING_INTERACTION_ERR {
                        if let Some(pending) = crate::interaction::take_pending_interaction() {
                            return Err(runmat_async::RuntimeControlFlow::Suspend(pending));
                        }
                    }
                    return Err(runmat_async::RuntimeControlFlow::Error(message));
                }
            };
            Ok(runmat_builtins::Value::from(res))
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
    let sink_bool = sink_flag;
    let suppress_auto_output_bool = suppress_auto_output_flag;

    let builtin_expr = quote! {
        runmat_builtins::BuiltinFunction::new(
            #name_str,
            #summary_tok,
            #category_tok,
            "",
            "",
            vec![#(#inferred_param_types),*],
            #inferred_return_type,
            #wrapper_ident,
            #accel_slice,
            #sink_bool,
            #suppress_auto_output_bool,
        )
    };

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
    let builtin_path: syn::Path = syn::parse_str(&builtin_path_lit.value())
        .expect("runtime_builtin `builtin_path` must be a valid path");
    let helper_ident = format_ident!("__runmat_wasm_register_builtin_{}", ident);
    let builtin_expr_helper = builtin_expr.clone();
    let doc_expr_helper = doc_expr.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            runmat_builtins::wasm_registry::submit_builtin_function(#builtin_expr_helper);
            runmat_builtins::wasm_registry::submit_builtin_doc(#doc_expr_helper);
        }
    };
    let register_native = quote! {
        #[cfg(not(target_arch = "wasm32"))]
        runmat_builtins::inventory::submit! { #builtin_expr }
        #[cfg(not(target_arch = "wasm32"))]
        runmat_builtins::inventory::submit! { #doc_expr }
    };
    append_wasm_block(quote! {
        #builtin_path::#helper_ident();
    });

    TokenStream::from(quote! {
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        #func
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
/// use runmat_builtins::Value;
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
        runmat_builtins::Constant {
            name: #name,
            value: #value,
        }
    };

    let helper_ident = helper_ident_from_name("__runmat_wasm_register_const_", &name);
    let constant_expr_helper = constant_expr.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            runmat_builtins::wasm_registry::submit_constant(#constant_expr_helper);
        }
    };
    let register_native = quote! {
        #[cfg(not(target_arch = "wasm32"))]
        #[allow(non_upper_case_globals)]
        runmat_builtins::inventory::submit! { #constant_expr }
    };
    append_wasm_block(quote! {
        #builtin_path::#helper_ident();
    });

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
    let constant_expr = quote! {
        runmat_builtins::Constant {
            name: #name,
            value: #value,
        }
    };
    let helper_ident = helper_ident_from_name("__runmat_wasm_register_const_", &name.value());
    let builtin_path: syn::Path = syn::parse_str(&builtin_path.value())
        .expect("register_constant `builtin_path` must be a valid path");
    let constant_expr_helper = constant_expr.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            runmat_builtins::wasm_registry::submit_constant(#constant_expr_helper);
        }
    };
    append_wasm_block(quote! {
        #builtin_path::#helper_ident();
    });
    TokenStream::from(quote! {
        #wasm_helper
        #[cfg(not(target_arch = "wasm32"))]
        runmat_builtins::inventory::submit! { #constant_expr }
    })
}

struct RegisterSpecAttrArgs {
    spec_expr: Option<Expr>,
    builtin_path: Option<LitStr>,
}

impl Parse for RegisterSpecAttrArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut spec_expr = None;
        let mut builtin_path = None;
        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            if ident == "spec" {
                spec_expr = Some(input.parse()?);
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
        })
    }
}

struct RegisterDocAttrArgs {
    name: Expr,
    text: Option<Expr>,
    builtin_path: Option<LitStr>,
}

impl Parse for RegisterDocAttrArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut name = None;
        let mut text = None;
        let mut builtin_path = None;
        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            if ident == "name" {
                name = Some(input.parse()?);
            } else if ident == "text" {
                text = Some(input.parse()?);
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
            name: name.ok_or_else(|| input.error("missing `name` argument"))?,
            text,
            builtin_path,
        })
    }
}

#[proc_macro_attribute]
pub fn register_gpu_spec(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterSpecAttrArgs);
    let RegisterSpecAttrArgs {
        spec_expr,
        builtin_path,
    } = args;
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
    let spec_tokens_helper = spec_tokens.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            crate::builtins::common::spec::wasm_registry::submit_gpu_spec(&#spec_tokens_helper);
        }
    };
    append_wasm_block(quote! {
        #builtin_path::#helper_ident();
    });
    let expanded = quote! {
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        #item_const
        #wasm_helper
        #[cfg(not(target_arch = "wasm32"))]
        inventory::submit! {
            crate::builtins::common::spec::GpuSpecInventory { spec: &#spec_for_native }
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
    } = args;
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
    let spec_tokens_helper = spec_tokens.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            crate::builtins::common::spec::wasm_registry::submit_fusion_spec(&#spec_tokens_helper);
        }
    };
    append_wasm_block(quote! {
        #builtin_path::#helper_ident();
    });
    let expanded = quote! {
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        #item_const
        #wasm_helper
        #[cfg(not(target_arch = "wasm32"))]
        inventory::submit! {
            crate::builtins::common::spec::FusionSpecInventory { spec: &#spec_for_native }
        }
    };
    expanded.into()
}

#[proc_macro_attribute]
pub fn register_doc_text(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterDocAttrArgs);
    let RegisterDocAttrArgs {
        name,
        text,
        builtin_path,
    } = args;
    let item_const = parse_macro_input!(item as ItemConst);
    let name_tokens = quote! { #name };
    let text_tokens = text.map(|expr| quote! { #expr }).unwrap_or_else(|| {
        let ident = &item_const.ident;
        quote! { #ident }
    });
    let builtin_path_lit =
        builtin_path.expect("register_doc_text requires `builtin_path = \"...\"` argument");
    let builtin_path: syn::Path = syn::parse_str(&builtin_path_lit.value())
        .expect("register_doc_text `builtin_path` must be a valid path");
    let helper_ident = format_ident!(
        "__runmat_wasm_register_doc_text_{}",
        item_const.ident.to_string()
    );
    let wasm_name = name_tokens.clone();
    let wasm_text = text_tokens.clone();
    let wasm_helper = quote! {
        #[cfg(target_arch = "wasm32")]
        #[allow(non_snake_case)]
        pub(crate) fn #helper_ident() {
            const ENTRY: crate::builtins::common::spec::DocTextInventory =
                crate::builtins::common::spec::DocTextInventory {
                    name: #wasm_name,
                    text: #wasm_text,
                };
            crate::builtins::common::spec::wasm_registry::submit_doc_text(&ENTRY);
        }
    };
    append_wasm_block(quote! {
        #builtin_path::#helper_ident();
    });
    let expanded = quote! {
        #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
        #item_const
        #wasm_helper
        #[cfg(all(not(target_arch = "wasm32"), feature = "doc_export"))]
        inventory::submit! {
            crate::builtins::common::spec::DocTextInventory { name: #name_tokens, text: #text_tokens }
        }
    };
    expanded.into()
}

fn append_wasm_block(block: proc_macro2::TokenStream) {
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
    let insertion = format!("    {}\n", block);
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
        const HEADER: &str = "pub fn register_all() {\n}\n";
        fs::write(path, HEADER).expect("failed to create wasm registry file");
    });
}

fn should_generate_wasm_registry() -> bool {
    // Generate the registry file for all builds by default so that downstream
    // wasm consumers (which compile proc-macros for the host) still produce the table.
    // Allow opting out with RUNMAT_DISABLE_WASM_REGISTRY=1.
    !matches!(
        std::env::var("RUNMAT_DISABLE_WASM_REGISTRY"),
        Ok(ref value) if value == "1"
    )
}

fn workspace_registry_path() -> Option<PathBuf> {
    let mut dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    loop {
        if dir.join("Cargo.lock").exists() {
            return Some(dir.join("target").join("runmat_wasm_registry.rs"));
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
