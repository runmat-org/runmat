use anyhow::{Context, Result, bail};
use clap::Parser;
use proc_macro2::Span;
use quote::quote;
use std::fs;
use std::path::{Path, PathBuf};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Attribute, File, ItemFn, ReturnType, Type, TypePath};
use walkdir::WalkDir;

#[derive(Parser)]
struct Args {
    /// Only report files that would change; do not modify them.
    #[arg(long)]
    check: bool,

    /// After converting return types, also coerce legacy `Result<_, String>` return expressions
    /// inside builtin functions by wrapping them with `.map_err(Into::into)`.
    #[arg(long)]
    coerce_returns: bool,

    /// Path to runmat-runtime/src (or a subdirectory like builtins/)
    #[arg(long, default_value = "crates/runmat-runtime/src")]
    src: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let src_root = args
        .src
        .canonicalize()
        .context("unable to resolve src directory")?;

    let mut rust_files = Vec::new();
    for entry in WalkDir::new(&src_root).into_iter().filter_entry(|e| !is_hidden(e.path())) {
        let entry = entry?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|ext| ext == "rs") {
            rust_files.push(entry.into_path());
        }
    }

    let mut changed_files = Vec::new();
    for file in rust_files {
        let changed = process_file(&file, args.check, args.coerce_returns)
            .with_context(|| format!("processing {}", file.display()))?;
        if changed {
            changed_files.push(file);
        }
    }

    if args.check && !changed_files.is_empty() {
        bail!(
            "builtin-controlflow-migrate would update {} files:\n{}",
            changed_files.len(),
            changed_files
                .iter()
                .map(|p| format!(" - {}", p.display()))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    Ok(())
}

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

fn process_file(path: &Path, check_only: bool, coerce_returns: bool) -> Result<bool> {
    let mut content = fs::read_to_string(path)?;
    let line_offsets = line_offsets(&content);
    let syntax: File = syn::parse_file(&content)?;

    let mut edits = Vec::new();
    let mut collector = BuiltinReturnCollector::new(&line_offsets, &content, false);
    collector.visit_file(&syntax);
    edits.extend(collector.into_edits());

    if coerce_returns {
        // Optional second pass: wrap return expressions for already-migrated functions.
        let mut coercer = BuiltinReturnCollector::new(&line_offsets, &content, true);
        coercer.visit_file(&syntax);
        edits.extend(coercer.into_edits());
    }

    if edits.is_empty() {
        return Ok(false);
    }
    if check_only {
        return Ok(true);
    }

    apply_edits(&mut content, edits);
    fs::write(path, content)?;
    Ok(true)
}

fn line_offsets(content: &str) -> Vec<usize> {
    let mut offsets = vec![0];
    for (idx, ch) in content.char_indices() {
        if ch == '\n' {
            offsets.push(idx + 1);
        }
    }
    offsets
}

fn span_start(span: Span, offsets: &[usize]) -> usize {
    let loc = span.start();
    offsets
        .get(loc.line - 1)
        .cloned()
        .unwrap_or(0)
        .saturating_add(loc.column)
}

fn span_end(span: Span, offsets: &[usize]) -> usize {
    let loc = span.end();
    offsets
        .get(loc.line - 1)
        .cloned()
        .unwrap_or(0)
        .saturating_add(loc.column)
}

#[derive(Debug)]
struct Edit {
    start: usize,
    end: usize,
    replacement: String,
}

fn apply_edits(content: &mut String, mut edits: Vec<Edit>) {
    edits.sort_by(|a, b| b.start.cmp(&a.start));
    for edit in edits {
        content.replace_range(edit.start..edit.end, &edit.replacement);
    }
}

struct BuiltinReturnCollector<'a> {
    line_offsets: &'a [usize],
    content: &'a str,
    coerce_returns: bool,
    in_builtin: bool,
    edits: Vec<Edit>,
}

impl<'a> BuiltinReturnCollector<'a> {
    fn new(line_offsets: &'a [usize], content: &'a str, coerce_returns: bool) -> Self {
        Self {
            line_offsets,
            content,
            coerce_returns,
            in_builtin: false,
            edits: Vec::new(),
        }
    }

    fn into_edits(self) -> Vec<Edit> {
        self.edits
    }

    fn has_runtime_builtin_attr(attrs: &[Attribute]) -> bool {
        attrs.iter().any(|attr| {
            let p = attr_path(attr);
            p == "runtime_builtin" || p == "runmat_macros::runtime_builtin"
        })
    }
}

impl<'a, 'ast> Visit<'ast> for BuiltinReturnCollector<'a> {
    fn visit_item_fn(&mut self, i: &'ast ItemFn) {
        if !Self::has_runtime_builtin_attr(&i.attrs) {
            return;
        }

        let ReturnType::Type(_, ty) = &i.sig.output else {
            // no return type: skip
            return;
        };

        if !self.coerce_returns {
            // Phase A: rewrite signature return types.
            // Match Result<Ok, String>
            let Some((ok_ty, err_ty)) = split_result_type(ty.as_ref()) else {
                return;
            };
            if !is_string_type(err_ty) {
                return;
            }

            // Replace only the type span, keeping everything else unchanged.
            let span = ty.span();
            let start = span_start(span, self.line_offsets);
            let end = span_end(span, self.line_offsets);
            let ok_tokens = quote! { #ok_ty }.to_string();
            let replacement = format!("crate::BuiltinResult<{ok_tokens}>");
            self.edits.push(Edit {
                start,
                end,
                replacement,
            });
            return;
        }

        // Phase B: for already-rewritten signatures, coerce legacy return expressions.
        if !is_builtin_result_type(ty.as_ref()) {
            return;
        }

        // Coerce the *function body* tail expression (only the outermost block).
        if let Some(syn::Stmt::Expr(expr)) = i.block.stmts.last() {
            if should_wrap(expr) {
                let replacement = wrap_expr(self.content, expr.span(), self.line_offsets);
                self.edits.push(Edit {
                    start: replacement.0,
                    end: replacement.1,
                    replacement: replacement.2,
                });
            }
        }

        // Visit nested expressions so we catch `return ...;` and `Err(...)` anywhere
        // (e.g. inside `if` blocks).
        self.in_builtin = true;
        syn::visit::visit_block(self, &i.block);
        self.in_builtin = false;
    }

    fn visit_expr_return(&mut self, i: &'ast syn::ExprReturn) {
        if self.coerce_returns && self.in_builtin {
            if let Some(inner) = &i.expr {
                if should_wrap(inner) {
                    let replacement = wrap_expr(self.content, inner.span(), self.line_offsets);
                    self.edits.push(Edit {
                        start: replacement.0,
                        end: replacement.1,
                        replacement: replacement.2,
                    });
                }
            }
        }
        syn::visit::visit_expr_return(self, i);
    }

    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        if self.coerce_returns && self.in_builtin {
            // Convert `Err(<string-ish>)` into `Err((<string-ish>).into())` so BuiltinResult
            // functions return `RuntimeControlFlow::Error(...)` instead of `String`.
            if let syn::Expr::Path(p) = i.func.as_ref() {
                if let Some(seg) = p.path.segments.last() {
                    if seg.ident == "Err" {
                        if let Some(arg0) = i.args.first() {
                            let span = arg0.span();
                            let start = span_start(span, self.line_offsets);
                            let end = span_end(span, self.line_offsets);
                            let original = self.content.get(start..end).unwrap_or("").to_string();
                            let replacement = format!("({original}).into()");
                            self.edits.push(Edit { start, end, replacement });
                        }
                    }
                }
            }
        }
        syn::visit::visit_expr_call(self, i);
    }

    fn visit_expr_match(&mut self, i: &'ast syn::ExprMatch) {
        if self.coerce_returns && self.in_builtin {
            for arm in &i.arms {
                let body = arm.body.as_ref();
                if should_wrap_result_expr(body) {
                    let replacement = wrap_expr(self.content, body.span(), self.line_offsets);
                    self.edits.push(Edit {
                        start: replacement.0,
                        end: replacement.1,
                        replacement: replacement.2,
                    });
                }
            }
        }
        syn::visit::visit_expr_match(self, i);
    }
}

fn split_result_type(ty: &Type) -> Option<(&Type, &Type)> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return None;
    };
    let seg = path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let mut iter = args.args.iter();
    let ok = match iter.next()? {
        syn::GenericArgument::Type(t) => t,
        _ => return None,
    };
    let err = match iter.next()? {
        syn::GenericArgument::Type(t) => t,
        _ => return None,
    };
    Some((ok, err))
}

fn is_string_type(ty: &Type) -> bool {
    match ty {
        Type::Path(tp) => {
            let last = tp.path.segments.last().map(|s| s.ident.to_string());
            matches!(last.as_deref(), Some("String"))
        }
        _ => false,
    }
}

fn is_builtin_result_type(ty: &Type) -> bool {
    let Type::Path(tp) = ty else {
        return false;
    };
    let last = tp.path.segments.last().map(|s| s.ident.to_string());
    matches!(last.as_deref(), Some("BuiltinResult"))
}

fn should_wrap(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::ForLoop(_) | syn::Expr::While(_) | syn::Expr::Loop(_) => false,
        // Don't wrap Ok(...) / Err(...); those are already correct.
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(p) = call.func.as_ref() {
                if let Some(seg) = p.path.segments.last() {
                    return seg.ident != "Ok" && seg.ident != "Err";
                }
            }
            true
        }
        syn::Expr::Try(_) => false, // already using `?`
        syn::Expr::Path(_) => true,
        syn::Expr::MethodCall(mc) => {
            // If it's already `...map_err(Into::into)`, don't wrap again.
            if mc.method == "map_err" {
                if let Some(arg) = mc.args.first() {
                    if let syn::Expr::Path(p) = arg {
                        if p.path.segments.iter().any(|s| s.ident == "Into") {
                            return false;
                        }
                    }
                }
            }
            true
        }
        syn::Expr::Block(_) | syn::Expr::If(_) | syn::Expr::Match(_) => false,
        _ => true,
    }
}

fn should_wrap_result_expr(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(p) = call.func.as_ref() {
                if let Some(seg) = p.path.segments.last() {
                    return seg.ident != "Ok" && seg.ident != "Err";
                }
            }
            true
        }
        syn::Expr::MethodCall(mc) => {
            // If it's already `...map_err(Into::into)`, don't wrap again.
            if mc.method == "map_err" {
                if let Some(arg) = mc.args.first() {
                    if let syn::Expr::Path(p) = arg {
                        if p.path.segments.iter().any(|s| s.ident == "Into") {
                            return false;
                        }
                    }
                }
            }
            true
        }
        _ => false,
    }
}

fn wrap_expr(content: &str, span: Span, offsets: &[usize]) -> (usize, usize, String) {
    let start = span_start(span, offsets);
    let end = span_end(span, offsets);
    let original = content.get(start..end).unwrap_or("").to_string();
    let replacement = format!("({original}).map_err(Into::into)");
    (start, end, replacement)
}

fn attr_path(attr: &Attribute) -> String {
    attr.path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}


