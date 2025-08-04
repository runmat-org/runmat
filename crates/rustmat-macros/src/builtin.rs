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
    
    let expanded = quote! {
        #fn_vis fn #fn_name(#fn_inputs) #fn_output #fn_block
        
        rustmat_builtins::inventory::submit! {
            #![crate = rustmat_builtins]
            rustmat_builtins::BuiltinFunction::new(
                #name,
                #description,
                #category_variant,
                vec![], // TODO: Extract parameters from function signature
                rustmat_builtins::ParameterType::Any, // TODO: Extract return type
                #fn_name
            )
        }
    };
    
    TokenStream::from(expanded)
}