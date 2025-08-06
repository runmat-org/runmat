//! Modern builtin function macro for RustMat

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, Meta, NestedMeta, AttributeArgs};

/// Macro for defining builtin functions with automatic registration
/// 
/// Usage:
/// ```rust
/// #[builtin_function(
///     name = "plot",
///     category = "Plotting", 
///     description = "Create a 2D line plot"
/// )]
/// fn plot_impl(args: &Arguments) -> BuiltinResult {
///     // Implementation here
/// }
/// ```
pub fn builtin_function_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as AttributeArgs);
    let input_fn = parse_macro_input!(input as ItemFn);
    
    // Extract attributes
    let mut name = None;
    let mut category = None;
    let mut description = None;
    
    for arg in args {
        match arg {
            NestedMeta::Meta(Meta::NameValue(nv)) if nv.path.is_ident("name") => {
                if let syn::Lit::Str(lit_str) = nv.lit {
                    name = Some(lit_str.value());
                }
            }
            NestedMeta::Meta(Meta::NameValue(nv)) if nv.path.is_ident("category") => {
                if let syn::Lit::Str(lit_str) = nv.lit {
                    category = Some(lit_str.value());
                }
            }
            NestedMeta::Meta(Meta::NameValue(nv)) if nv.path.is_ident("description") => {
                if let syn::Lit::Str(lit_str) = nv.lit {
                    description = Some(lit_str.value());
                }
            }
            _ => {}
        }
    }
    
    let name = name.expect("builtin_function macro requires 'name' attribute");
    let category = category.unwrap_or_else(|| "User".to_string());
    let description = description.unwrap_or_else(|| "User-defined function".to_string());
    
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_block = &input_fn.block;
    let fn_inputs = &input_fn.sig.inputs;
    let fn_output = &input_fn.sig.output;
    
    // Convert category string to enum variant
    let category_variant = match category.as_str() {
        "Mathematics" => quote! { FunctionCategory::Mathematics },
        "LinearAlgebra" => quote! { FunctionCategory::LinearAlgebra },
        "Statistics" => quote! { FunctionCategory::Statistics },
        "SignalProcessing" => quote! { FunctionCategory::SignalProcessing },
        "ImageProcessing" => quote! { FunctionCategory::ImageProcessing },
        "ControlSystems" => quote! { FunctionCategory::ControlSystems },
        "Optimization" => quote! { FunctionCategory::Optimization },
        "Plotting" => quote! { FunctionCategory::Plotting },
        "FileIO" => quote! { FunctionCategory::FileIO },
        "Strings" => quote! { FunctionCategory::Strings },
        "DataAnalysis" => quote! { FunctionCategory::DataAnalysis },
        "Numerical" => quote! { FunctionCategory::Numerical },
        "System" => quote! { FunctionCategory::System },
        _ => quote! { FunctionCategory::User },
    };
    
    let _static_name = format!("__{}_BUILTIN", name.to_uppercase().replace("-", "_"));
    
    // Extract parameter information from function signature
    let param_info = extract_parameter_info(&parsed.sig);
    let return_type_info = extract_return_type_info(&parsed.sig);

    let expanded = quote! {
        #fn_vis fn #fn_name(#fn_inputs) #fn_output #fn_block
        
        rustmat_builtins::inventory::submit! {
            #![crate = rustmat_builtins]
            rustmat_builtins::BuiltinFunction::new(
                #name,
                #description,
                #category_variant,
                #param_info,
                #return_type_info,
                #fn_name
            )
        }
    };
    
    TokenStream::from(expanded)
}

/// Extract parameter information from function signature
fn extract_parameter_info(sig: &syn::Signature) -> proc_macro2::TokenStream {
    let mut params = Vec::new();
    
    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let param_name = pat_ident.ident.to_string();
                    let param_type = infer_parameter_type(&pat_type.ty);
                    
                    params.push(quote! {
                        rustmat_builtins::ParameterInfo {
                            name: #param_name.to_string(),
                            param_type: #param_type,
                            description: String::new(),
                            required: true,
                        }
                    });
                }
            }
            syn::FnArg::Receiver(_) => {
                // Skip self parameters
            }
        }
    }
    
    quote! { vec![#(#params),*] }
}

/// Extract return type information from function signature
fn extract_return_type_info(sig: &syn::Signature) -> proc_macro2::TokenStream {
    match &sig.output {
        syn::ReturnType::Default => quote! { rustmat_builtins::ParameterType::Any },
        syn::ReturnType::Type(_, ty) => infer_parameter_type(ty),
    }
}

/// Infer parameter type from Rust type
fn infer_parameter_type(ty: &syn::Type) -> proc_macro2::TokenStream {
    // Convert type to string and analyze
    let type_str = quote! { #ty }.to_string();
    
    if type_str.contains("f64") || type_str.contains("f32") {
        quote! { rustmat_builtins::ParameterType::Number }
    } else if type_str.contains("i64") || type_str.contains("i32") || type_str.contains("usize") {
        quote! { rustmat_builtins::ParameterType::Integer }
    } else if type_str.contains("Matrix") {
        quote! { rustmat_builtins::ParameterType::Matrix }
    } else if type_str.contains("Value") {
        quote! { rustmat_builtins::ParameterType::Any }
    } else if type_str.contains("String") || type_str.contains("str") {
        quote! { rustmat_builtins::ParameterType::String }
    } else if type_str.contains("bool") {
        quote! { rustmat_builtins::ParameterType::Boolean }
    } else {
        quote! { rustmat_builtins::ParameterType::Any }
    }
}