use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, AttributeArgs, FnArg, ItemFn, Lit, Meta, MetaNameValue, NestedMeta, Pat,
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
        const _: () = {
            rustmat_builtins::inventory::submit! {
                rustmat_builtins::Builtin {
                    name: #name_str,
                    func: #wrapper_ident as rustmat_builtins::BuiltinFn,
                }
            }
        };
    };

    TokenStream::from(quote! {
        #func
        #wrapper
        #register
    })
}
