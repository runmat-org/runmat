//! Compare original raw fields with current shared quantities. Preserve canonical
//! numeric serde shapes, including Deserialize-only owners with manual Serialize.
use quote::ToTokens;
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
};
const BASE: &str = "899b0b2605646e7f79924ac0acdb00562ffa855f";
struct Fields {
    owner: String,
    fields: BTreeMap<(String, String), String>,
}
impl<'a> Visit<'a> for Fields {
    fn visit_item_struct(&mut self, x: &'a syn::ItemStruct) {
        let old = self.owner.clone();
        self.owner = x.ident.to_string();
        visit::visit_item_struct(self, x);
        self.owner = old;
    }
    fn visit_item_enum(&mut self, x: &'a syn::ItemEnum) {
        let old = self.owner.clone();
        self.owner = x.ident.to_string();
        for v in &x.variants {
            self.owner = format!("{}::{}", x.ident, v.ident);
            for f in &v.fields {
                self.visit_field(f);
            }
        }
        self.owner = old;
    }
    fn visit_field(&mut self, f: &'a syn::Field) {
        if let Some(n) = &f.ident {
            self.fields.insert(
                (self.owner.clone(), n.to_string()),
                f.ty.to_token_stream().to_string(),
            );
        }
    }
}
struct R<'a> {
    base: &'a BTreeMap<(String, String), String>,
    owner: String,
    serde_owner: bool,
    lines: Vec<usize>,
    edits: Vec<(usize, String)>,
    names: Vec<String>,
}
impl<'a> R<'a> {
    fn enter(&mut self, name: String, attrs: &[syn::Attribute]) -> (String, bool) {
        let old = (self.owner.clone(), self.serde_owner);
        self.owner = name;
        self.serde_owner = attrs.iter().any(|a| {
            a.to_token_stream().to_string().contains("Deserialize")
                || a.to_token_stream().to_string().contains("Serialize")
        });
        old
    }
}
impl<'a> Visit<'a> for R<'_> {
    fn visit_item_struct(&mut self, x: &'a syn::ItemStruct) {
        let old = self.enter(x.ident.to_string(), &x.attrs);
        visit::visit_item_struct(self, x);
        (self.owner, self.serde_owner) = old;
    }
    fn visit_item_enum(&mut self, x: &'a syn::ItemEnum) {
        let old = self.enter(x.ident.to_string(), &x.attrs);
        for v in &x.variants {
            self.owner = format!("{}::{}", x.ident, v.ident);
            for f in &v.fields {
                self.visit_field(f);
            }
        }
        (self.owner, self.serde_owner) = old;
    }
    fn visit_field(&mut self, f: &'a syn::Field) {
        if !self.serde_owner {
            return;
        }
        let Some(n) = &f.ident else { return };
        let Some(old) = self.base.get(&(self.owner.clone(), n.to_string())) else {
            return;
        };
        let ty = f.ty.to_token_stream().to_string();
        if !ty.contains("runmat_types :: physical ::")
            || (ty.contains("DynamicDimension") || ty.contains("DimensionedScalar"))
        {
            return;
        }
        if f.attrs.iter().any(|a| {
            let s = a.to_token_stream().to_string();
            s.contains("with") || s.contains("skip")
        }) {
            return;
        }
        let numeric = old == "f64"
            || old == "Option < f64 >"
            || old.starts_with("[f64")
            || old.starts_with("Vec < f64")
            || old.starts_with("Vec < [f64");
        if !numeric {
            return;
        }
        let adapter = if ty.starts_with("Option <") {
            "canonical_scalar::optional"
        } else if old == "f64" {
            "canonical_scalar"
        } else {
            "canonical_storage"
        };
        let p = f.span().start();
        self.edits.push((
            self.lines[p.line - 1] + p.column,
            format!("#[serde(with = \"runmat_types::physical::{adapter}\")]\n"),
        ));
        self.names.push(format!("{}.{} ({old})", self.owner, n));
    }
}
fn paths(p: &Path, out: &mut Vec<PathBuf>) {
    if p.is_file() {
        if p.extension().is_some_and(|x| x == "rs") {
            out.push(p.into())
        }
        return;
    }
    for e in fs::read_dir(p).unwrap() {
        let e = e.unwrap();
        if !e.file_type().unwrap().is_symlink() {
            paths(&e.path(), out)
        }
    }
}
fn main() {
    let mut ps = vec![];
    for arg in env::args().skip(1) {
        paths(Path::new(&arg), &mut ps)
    }
    for p in ps {
        let s = fs::read_to_string(&p).unwrap();
        if !s.contains("Deserialize") {
            continue;
        }
        let Ok(ast) = syn::parse_file(&s) else {
            continue;
        };
        let baseline = Command::new("git")
            .args(["show", &format!("{BASE}:{}", p.display())])
            .output()
            .unwrap();
        if !baseline.status.success() {
            continue;
        }
        let Ok(old) = syn::parse_file(&String::from_utf8(baseline.stdout).unwrap()) else {
            continue;
        };
        let mut f = Fields {
            owner: String::new(),
            fields: BTreeMap::new(),
        };
        f.visit_file(&old);
        let mut r = R {
            base: &f.fields,
            owner: String::new(),
            serde_owner: false,
            lines: std::iter::once(0)
                .chain(s.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            edits: vec![],
            names: vec![],
        };
        r.visit_file(&ast);
        if r.edits.is_empty() {
            continue;
        }
        r.edits.sort_by_key(|e| e.0);
        let mut output = s;
        for (pos, v) in r.edits.into_iter().rev() {
            output.insert_str(pos, &v)
        }
        fs::write(&p, output).unwrap();
        println!("{}: {}", p.display(), r.names.join(", "));
    }
}
