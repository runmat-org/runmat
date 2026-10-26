//! Reviewed canonical-field migration. The mapping is semantic, not a suffix heuristic.
//! Existing canonical numeric wire fields retain their wire shape through one shared adapter.
use quote::ToTokens;
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
};
struct Rewrite<'a> {
    offsets: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
    mapping: &'a BTreeMap<&'static str, &'static str>,
    serializable: bool,
    lhs: bool,
}
impl Rewrite<'_> {
    fn range(&self, s: proc_macro2::Span) -> (usize, usize) {
        let (a, b) = (s.start(), s.end());
        (
            self.offsets[a.line - 1] + a.column,
            self.offsets[b.line - 1] + b.column,
        )
    }
    fn insert(&mut self, pos: usize, s: impl Into<String>) {
        self.edits.push((pos, pos, s.into()));
    }
    fn wrap(&mut self, e: &syn::Expr, kind: &str, prefix: &str) {
        let (a, b) = self.range(e.span());
        self.insert(a, format!("{prefix}("));
        self.insert(b, format!(").map(runmat_types::physical::{kind}::new)"));
    }
    fn macro_members(&mut self, ts: proc_macro2::TokenStream) {
        let mut dot = false;
        for token in ts {
            match token {
                proc_macro2::TokenTree::Group(g) => {
                    self.macro_members(g.stream());
                    dot = false
                }
                proc_macro2::TokenTree::Punct(p) => dot = p.as_char() == '.',
                proc_macro2::TokenTree::Ident(i) => {
                    if dot && self.mapping.contains_key(i.to_string().as_str()) {
                        let (_, b) = self.range(i.span());
                        self.insert(b, ".map(|quantity| quantity.canonical_value())");
                    }
                    dot = false
                }
                _ => dot = false,
            }
        }
    }
}
impl<'ast> Visit<'ast> for Rewrite<'_> {
    fn visit_item_struct(&mut self, x: &'ast syn::ItemStruct) {
        let old = self.serializable;
        self.serializable = x
            .attrs
            .iter()
            .any(|a| a.meta.to_token_stream().to_string().contains("Serialize"));
        visit::visit_item_struct(self, x);
        self.serializable = old;
    }
    fn visit_item_enum(&mut self, x: &'ast syn::ItemEnum) {
        let old = self.serializable;
        self.serializable = x
            .attrs
            .iter()
            .any(|a| a.meta.to_token_stream().to_string().contains("Serialize"));
        visit::visit_item_enum(self, x);
        self.serializable = old;
    }
    fn visit_field(&mut self, x: &'ast syn::Field) {
        if let Some(name) = &x.ident {
            if let Some(kind) = self.mapping.get(name.to_string().as_str()) {
                if x.ty.to_token_stream().to_string() == "Option < f64 >" {
                    let (a, b) = self.range(x.ty.span());
                    self.edits
                        .push((a, b, format!("Option<runmat_types::physical::{kind}>")));
                    if self.serializable {
                        let (a, _) = self.range(x.span());
                        self.insert(
                            a,
                            "#[serde(with = \"runmat_types::physical::canonical_scalar::optional\")]\n",
                        );
                    }
                }
            }
        }
        visit::visit_field(self, x);
    }
    fn visit_expr_field(&mut self, x: &'ast syn::ExprField) {
        if !self.lhs {
            if let syn::Member::Named(name) = &x.member {
                if self.mapping.contains_key(name.to_string().as_str()) {
                    let (_, b) = self.range(x.span());
                    self.insert(b, ".map(|quantity| quantity.canonical_value())");
                }
            }
        }
        visit::visit_expr_field(self, x);
    }
    fn visit_expr_assign(&mut self, x: &'ast syn::ExprAssign) {
        let old = self.lhs;
        self.lhs = true;
        self.visit_expr(&x.left);
        self.lhs = old;
        if let syn::Expr::Field(f) = x.left.as_ref() {
            if let syn::Member::Named(name) = &f.member {
                if let Some(kind) = self.mapping.get(name.to_string().as_str()) {
                    self.wrap(&x.right, kind, "");
                }
            }
        }
        self.visit_expr(&x.right);
    }
    fn visit_expr_struct(&mut self, x: &'ast syn::ExprStruct) {
        for f in &x.fields {
            if let syn::Member::Named(name) = &f.member {
                if let Some(kind) = self.mapping.get(name.to_string().as_str()) {
                    let prefix = if f.colon_token.is_none() {
                        format!("{name}: ")
                    } else {
                        String::new()
                    };
                    self.wrap(&f.expr, kind, &prefix);
                }
            }
        }
        visit::visit_expr_struct(self, x);
    }
    fn visit_macro(&mut self, x: &'ast syn::Macro) {
        self.macro_members(x.tokens.clone());
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
        } else if e.path().extension().is_some_and(|s| s == "rs") {
            out.push(e.path())
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mapping = if env::var("QUANTITY_FIELD_BATCH").as_deref() == Ok("signal-waveform") {
        BTreeMap::from([("duration_seconds", "Time"), ("frequency_hz", "Frequency")])
    } else {
        BTreeMap::from([("mass_density_kg_per_m3", "MassDensity")])
    };
    let mut files = vec![];
    for arg in env::args().skip(1) {
        paths(Path::new(&arg), &mut files)?
    }
    let (mut count, mut total) = (0, 0);
    let mut crates = BTreeSet::new();
    for path in files {
        if path.to_string_lossy().contains("quantity-inventory/") {
            continue;
        }
        let source = fs::read_to_string(&path)?;
        let syntax = syn::parse_file(&source)?;
        let mut offsets = vec![0];
        offsets.extend(source.match_indices('\n').map(|(i, _)| i + 1));
        let mut r = Rewrite {
            offsets,
            edits: vec![],
            mapping: &mapping,
            serializable: false,
            lhs: false,
        };
        r.visit_file(&syntax);
        if r.edits.is_empty() {
            continue;
        }
        r.edits
            .sort_by_key(|e| (e.0, e.1, usize::from(e.2.starts_with(").map("))));
        r.edits.dedup();
        let mut output = source.clone();
        let mut upper = source.len();
        for (a, b, s) in r.edits.iter().rev() {
            if *b > upper {
                return Err(format!("overlap {}", path.display()).into());
            }
            output.replace_range(*a..*b, s);
            upper = *a;
        }
        fs::write(&path, output)?;
        count += 1;
        total += r.edits.len();
        if let Some(manifest) = path
            .ancestors()
            .find_map(|a| a.join("Cargo.toml").is_file().then(|| a.join("Cargo.toml")))
        {
            crates.insert(manifest);
        }
    }
    println!("{total} edits in {count} files");
    for c in crates {
        println!("{}", c.display())
    }
    Ok(())
}
