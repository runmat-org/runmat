use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, AttributeArgs, FnArg, ItemFn, Lit, Meta, MetaNameValue, NestedMeta, Pat, Expr,
};

/// Attribute used to mark functions as implementing a runtime builtin.
///
/// Example:
/// ```rust,ignore
/// use rustmat_macros::runtime_builtin;
///
/// #[runtime_builtin(name = "plot")]
/// pub fn plot_line(xs: &[f64]) {
///     /* implementation */
/// }
/// ```
///
/// This registers the function with the `rustmat-builtins` inventory
/// so the runtime can discover it at start-up.
#[proc_macro_attribute]
pub fn runtime_builtin(args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse attribute arguments as `name = "..."`
    let args = parse_macro_input!(args as AttributeArgs);
    let mut name_lit: Option<Lit> = None;
    for arg in args {
        if let NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) = arg {
            if path.is_ident("name") {
                name_lit = Some(lit);
            } else {
                panic!("unknown attribute parameter; only `name` is supported");
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
    let inferred_param_types: Vec<proc_macro2::TokenStream> = param_types
        .iter()
        .map(infer_builtin_type)
        .collect();
    
    // Infer return type for BuiltinFunction
    let inferred_return_type = match &func.sig.output {
        syn::ReturnType::Default => quote! { rustmat_builtins::Type::Void },
        syn::ReturnType::Type(_, ty) => infer_builtin_type(ty),
    };
    
    // Generate wrapper ident
    let wrapper_ident = format_ident!("__rt_wrap_{}", ident);

    let conv_stmts: Vec<proc_macro2::TokenStream> = param_idents
        .iter()
        .zip(param_types.iter())
        .enumerate()
        .map(|(i, (ident, ty))| {
            quote! { let #ident : #ty = std::convert::TryInto::try_into(&args[#i])?; }
        })
        .collect();

    let wrapper = quote! {
        fn #wrapper_ident(args: &[rustmat_builtins::Value]) -> Result<rustmat_builtins::Value, String> {
            if args.len() != #param_len {
                return Err(format!("expected {} args, got {}", #param_len, args.len()));
            }
            #(#conv_stmts)*
            let res = #ident(#(#param_idents),*)?;
            Ok(rustmat_builtins::Value::from(res))
        }
    };

    let register = quote! {
        rustmat_builtins::inventory::submit! {
            rustmat_builtins::BuiltinFunction::new(
                #name_str,
                "Runtime builtin function",
                vec![#(#inferred_param_types),*],
                #inferred_return_type,
                #wrapper_ident
            )
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
/// use rustmat_macros::runtime_constant;
/// use rustmat_builtins::Value;
///
/// #[runtime_constant(name = "pi", value = std::f64::consts::PI)]
/// const PI_CONSTANT: ();
/// ```
///
/// This registers the constant with the `rustmat-builtins` inventory
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
            rustmat_builtins::inventory::submit! {
                rustmat_builtins::Constant {
                    name: #name,
                    value: rustmat_builtins::Value::Num(#value),
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
                    "i32" | "i64" | "isize" => quote! { rustmat_builtins::Type::Int },
                    "f32" | "f64" => quote! { rustmat_builtins::Type::Num },
                    "bool" => quote! { rustmat_builtins::Type::Bool },
                    "String" => quote! { rustmat_builtins::Type::String },
                    _ => infer_complex_type(type_path),
                }
            } else {
                infer_complex_type(type_path)
            }
        }
        
        // Reference types like &str, &Value, &Matrix
        Type::Reference(type_ref) => {
            match type_ref.elem.as_ref() {
                Type::Path(type_path) => {
                    if let Some(ident) = type_path.path.get_ident() {
                        match ident.to_string().as_str() {
                            "str" => quote! { rustmat_builtins::Type::String },
                            _ => infer_builtin_type(&type_ref.elem),
                        }
                    } else {
                        infer_builtin_type(&type_ref.elem)
                    }
                }
                _ => infer_builtin_type(&type_ref.elem),
            }
        }
        
        // Slice types like &[Value], &[f64]
        Type::Slice(type_slice) => {
            let element_type = infer_builtin_type(&type_slice.elem);
            quote! { rustmat_builtins::Type::Cell { 
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
                        return quote! { rustmat_builtins::Type::Cell { 
                            element_type: Some(Box::new(#element_type)), 
                            length: Some(#length) 
                        } };
                    }
                }
            }
            // Fallback to unknown length
            quote! { rustmat_builtins::Type::Cell { 
                element_type: Some(Box::new(#element_type)), 
                length: None 
            } }
        }
        
        // Generic or complex types
        _ => quote! { rustmat_builtins::Type::Unknown },
    }
}

/// Infer types for complex path types like Result<T, E>, Option<T>, Matrix, Value
fn infer_complex_type(type_path: &syn::TypePath) -> proc_macro2::TokenStream {
    let path_str = quote! { #type_path }.to_string();
    
    // Handle common patterns
    if path_str.contains("Matrix") {
        quote! { rustmat_builtins::Type::matrix() }
    } else if path_str.contains("Value") {
        quote! { rustmat_builtins::Type::Unknown } // Value can be anything
    } else if path_str.starts_with("Result") {
        // Extract the Ok type from Result<T, E>
        if let syn::PathArguments::AngleBracketed(angle_bracketed) = &type_path.path.segments.last().unwrap().arguments {
            if let Some(syn::GenericArgument::Type(ty)) = angle_bracketed.args.first() {
                return infer_builtin_type(ty);
            }
        }
        quote! { rustmat_builtins::Type::Unknown }
    } else if path_str.starts_with("Option") {
        // Extract the Some type from Option<T>
        if let syn::PathArguments::AngleBracketed(angle_bracketed) = &type_path.path.segments.last().unwrap().arguments {
            if let Some(syn::GenericArgument::Type(ty)) = angle_bracketed.args.first() {
                return infer_builtin_type(ty);
            }
        }
        quote! { rustmat_builtins::Type::Unknown }
    } else if path_str.starts_with("Vec") {
        // Extract element type from Vec<T>
        if let syn::PathArguments::AngleBracketed(angle_bracketed) = &type_path.path.segments.last().unwrap().arguments {
            if let Some(syn::GenericArgument::Type(ty)) = angle_bracketed.args.first() {
                let element_type = infer_builtin_type(ty);
                return quote! { rustmat_builtins::Type::Cell { 
                    element_type: Some(Box::new(#element_type)), 
                    length: None 
                } };
            }
        }
        quote! { rustmat_builtins::Type::cell() }
    } else {
        // Unknown type
        quote! { rustmat_builtins::Type::Unknown }
    }
}
