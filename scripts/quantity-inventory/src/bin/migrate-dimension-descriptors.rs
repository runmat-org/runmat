//! Checked source-type constructors adopt the shared dynamic descriptor explicitly.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
};
struct Rewrite<'a> {
    source: &'a str,
    offsets: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
}
impl Rewrite<'_> {
    fn range(&self, s: proc_macro2::Span) -> (usize, usize) {
        let (a, b) = (s.start(), s.end());
        (
            self.offsets[a.line - 1] + a.column,
            self.offsets[b.line - 1] + b.column,
        )
    }
}
impl<'ast> Visit<'ast> for Rewrite<'_> {
    fn visit_expr_binary(&mut self, x: &'ast syn::ExprBinary) {
        if env::var("DESCRIPTOR_COMPARISONS").as_deref() == Ok("1")
            && matches!(x.op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_))
        {
            fn quantity_arg(e: &syn::Expr) -> Option<&syn::Expr> {
                if let syn::Expr::Reference(r) = e {
                    return quantity_arg(&r.expr);
                }
                let syn::Expr::Call(call) = e else {
                    return None;
                };
                let syn::Expr::Path(p) = call.func.as_ref() else {
                    return None;
                };
                let parts = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>();
                (parts.len() >= 2
                    && parts[parts.len() - 2] == "FeaType"
                    && parts[parts.len() - 1] == "Quantity"
                    && call.args.len() == 1)
                    .then(|| &call.args[0])
            }
            let pair = quantity_arg(&x.right)
                .map(|arg| (&x.left, arg))
                .or_else(|| quantity_arg(&x.left).map(|arg| (&x.right, arg)));
            if let Some((actual, arg)) = pair {
                let (a, b) = self.range(actual.span());
                let actual = &self.source[a..b];
                let (a, b) = self.range(arg.span());
                let arg = &self.source[a..b];
                let replacement = format!(
                    "{}({actual}).is_quantity({arg})",
                    if matches!(x.op, syn::BinOp::Ne(_)) {
                        "!"
                    } else {
                        ""
                    }
                );
                let (a, b) = self.range(x.span());
                self.edits.push((a, b, replacement));
                return;
            }
        }
        visit::visit_expr_binary(self, x)
    }

    fn visit_expr_call(&mut self, x: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = x.func.as_ref() {
            let s = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>();
            if s.len() >= 2
                && s[s.len() - 1] == "Quantity"
                && matches!(s[s.len() - 2].as_str(), "FeaType" | "AuthoredInputType")
                && x.args.len() == 1
            {
                let (a, b) = self.range(x.args[0].span());
                if !self.source[a..b].ends_with(".into()") {
                    self.edits.push((a, a, "(".into()));
                    self.edits.push((b, b, ").into()".into()));
                }
            }
        }
        visit::visit_expr_call(self, x)
    }
    fn visit_macro(&mut self, x: &'ast syn::Macro) {
        if x.path.segments.last().is_some_and(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "assert_eq" | "assert_ne" | "assert" | "debug_assert" | "format"
            )
        }) {
            use syn::parse::Parser;
            if let Ok(args) =
                syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
                    .parse2(x.tokens.clone())
            {
                for arg in args {
                    self.visit_expr(&arg)
                }
            }
        }
    }
}
fn paths(p: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if p.is_file() {
        if p.extension().is_some_and(|e| e == "rs") {
            out.push(p.into())
        }
        return Ok(());
    }
    for e in fs::read_dir(p)? {
        let e = e?;
        if e.file_type()?.is_symlink() {
            continue;
        }
        if e.file_type()?.is_dir() {
            if !matches!(
                e.file_name().to_str(),
                Some("target" | ".git" | "node_modules")
            ) {
                paths(&e.path(), out)?
            }
        } else if e.path().extension().is_some_and(|e| e == "rs") {
            out.push(e.path())
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut files = vec![];
    for arg in env::args().skip(1) {
        paths(Path::new(&arg), &mut files)?
    }
    let mut count = 0;
    for path in files {
        let source = fs::read_to_string(&path)?;
        let syntax = syn::parse_file(&source)?;
        let mut offsets = vec![0];
        offsets.extend(source.match_indices('\n').map(|(i, _)| i + 1));
        let mut r = Rewrite {
            source: &source,
            offsets,
            edits: vec![],
        };
        r.visit_file(&syntax);
        if r.edits.is_empty() {
            continue;
        }
        let mut output = source.clone();
        r.edits.sort_by_key(|e| e.0);
        r.edits.dedup();
        for (a, b, s) in r.edits.iter().rev() {
            output.replace_range(*a..*b, s)
        }
        fs::write(path, output)?;
        count += r.edits.len();
    }
    println!("adopted descriptor constructors in {count} source-position edits");
    Ok(())
}
