use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, AttributeArgs, ItemFn, Lit, Meta, MetaNameValue, NestedMeta};

/// Attribute used to mark functions as implementing a MATLAB builtin.
///
/// Example:
/// ```rust,ignore
/// use rustmat_macros::matlab_fn;
///
/// #[matlab_fn(name = "plot")]
/// pub fn plot_line(xs: &[f64]) {
///     /* implementation */
/// }
/// ```
///
/// This attaches documentation and registers the function with the
/// `rustmat-builtins` inventory so the runtime can discover it at start-up.
#[proc_macro_attribute]
pub fn matlab_fn(args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse attribute arguments as `name = "..."`
    let args = parse_macro_input!(args as AttributeArgs);
    let mut name_lit: Option<Lit> = None;
    for arg in args {
        if let NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) = arg {
            if path.is_ident("name") {
                name_lit = Some(lit);
            }
        }
    }
    let name_lit = name_lit.expect("expected `name = \"...\"` argument");
    let name_str = if let Lit::Str(ref s) = name_lit {
        s.value()
    } else {
        panic!("name must be a string literal");
    };

    let mut func = parse_macro_input!(input as ItemFn);
    let ident = &func.sig.ident;

    // Gather documentation from existing #[doc] attributes
    let mut docs = Vec::new();
    for attr in &func.attrs {
        if attr.path.is_ident("doc") {
            if let Ok(Meta::NameValue(MetaNameValue {
                lit: Lit::Str(s), ..
            })) = attr.parse_meta()
            {
                docs.push(s.value());
            }
        }
    }
    let builtin_doc = format!("Matlab builtin `{name_str}`");
    func.attrs.push(syn::parse_quote!(#[doc = #builtin_doc]));
    let joined_docs = docs.join("\n");

    let register = quote! {
        const _: () = {
            inventory::submit! {
                rustmat_builtins::Builtin {
                    name: #name_str,
                    doc: #joined_docs,
                    func: #ident as *const (),
                }
            }
        };
    };

    TokenStream::from(quote! {
        #func
        #register
    })
}
