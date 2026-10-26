//! Repair mutable canonical projections explicitly; no mutable raw-storage accessor is introduced.
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
    fn text(&self, s: proc_macro2::Span) -> &str {
        let (a, b) = self.range(s);
        &self.source[a..b]
    }
}
impl<'ast> Visit<'ast> for Rewrite<'_> {
    fn visit_expr_binary(&mut self, x: &'ast syn::ExprBinary) {
        let op = match &x.op {
            syn::BinOp::AddAssign(_) => "+",
            syn::BinOp::SubAssign(_) => "-",
            syn::BinOp::MulAssign(_) => "*",
            syn::BinOp::DivAssign(_) => "/",
            _ => {
                visit::visit_expr_binary(self, x);
                return;
            }
        };
        if let syn::Expr::MethodCall(m) = x.left.as_ref() {
            if m.method == "canonical_value" {
                let recv = self.text(m.receiver.span());
                let lhs = if matches!(m.receiver.as_ref(), syn::Expr::Path(_)) {
                    format!("*{recv}")
                } else {
                    recv.into()
                };
                let rhs = self.text(x.right.span());
                let (a, b) = self.range(x.span());
                self.edits.push((a,b,format!("{lhs} = runmat_types::physical::Quantity::new({recv}.canonical_value() {op} ({rhs}))")));
                return;
            }
        }
        visit::visit_expr_binary(self, x);
    }
    fn visit_expr_struct(&mut self, x: &'ast syn::ExprStruct) {
        if x.path
            .segments
            .last()
            .is_some_and(|s| s.ident == "GeometryRevisionRef")
            && !x.fields.iter().any(
                |f| matches!(&f.member, syn::Member::Named(name) if name == "source_unit_evidence"),
            )
        {
            let (start, _) = self.range(x.brace_token.span.open());
            self.edits.push((start+1,start+1,"source_unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(runmat_geometry_core::LengthUnit::Meter, \"canonical solver fixture\"),".into()));
        }
        visit::visit_expr_struct(self, x);
    }
    fn visit_expr_call(&mut self, x: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = x.func.as_ref() {
            if p.path
                .segments
                .last()
                .is_some_and(|s| s.ident == "validate_model_against_geometry")
                && x.args.len() == 3
            {
                let (a, _) = self.range(x.args[1].span());
                let (b, _) = self.range(x.args[2].span());
                self.edits.push((a, b, String::new()));
            }
        }
        visit::visit_expr_call(self, x);
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
        if path.to_string_lossy().contains("quantity-inventory/") {
            continue;
        }
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
        for (a, b, s) in r.edits.iter().rev() {
            output.replace_range(*a..*b, s)
        }
        fs::write(path, output)?;
        count += r.edits.len();
    }
    println!("repaired {count} mutable projections/API arguments");
    Ok(())
}
