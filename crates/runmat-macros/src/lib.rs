use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, AttributeArgs, Expr, FnArg, ItemFn, Lit, Meta, MetaNameValue, NestedMeta,
    Pat,
};

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
    let mut sink_flag = false;
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
        fn #wrapper_ident(args: &[runmat_builtins::Value]) -> Result<runmat_builtins::Value, String> {
            #![allow(unused_variables)]
            if #is_last_variadic {
                if args.len() < #param_len - 1 { return Err(format!("expected at least {} args, got {}", #param_len - 1, args.len())); }
            } else {
                if args.len() != #param_len { return Err(format!("expected {} args, got {}", #param_len, args.len())); }
            }
            #(#conv_stmts)*
            let res = #ident(#(#param_idents),*)?;
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

    let register = quote! {
        runmat_builtins::inventory::submit! {
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
            )
        }
        runmat_builtins::inventory::submit! {
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
        }
    };

    TokenStream::from(quote! {
        #func
        #wrapper
        #register
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

    for arg in args {
        match arg {
            NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) => {
                if path.is_ident("name") {
                    name_lit = Some(lit);
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

    let item = parse_macro_input!(input as syn::Item);

    let register = {
        quote! {
            #[allow(non_upper_case_globals)]
            runmat_builtins::inventory::submit! {
                runmat_builtins::Constant {
                    name: #name,
                    value: runmat_builtins::Value::Num(#value),
                }
            }
        }
    };

    TokenStream::from(quote! {
        #item
        #register
    })
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
