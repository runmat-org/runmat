//! One-time source-position codemod. Exact AST members only; never rewrites
//! strings/comments or unrelated `fraction`, `ratio` and `seconds` fields.
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
    source: &'a str,
    offsets: Vec<usize>,
    edits: Vec<(usize, usize, String)>,
    path: &'a Path,
    fields: BTreeSet<&'static str>,
    quantities: BTreeMap<&'static str, &'static str>,
}
impl Rewrite<'_> {
    fn range(&self, span: proc_macro2::Span) -> (usize, usize) {
        let a = span.start();
        let b = span.end();
        (
            self.offsets[a.line - 1] + a.column,
            self.offsets[b.line - 1] + b.column,
        )
    }
    fn text(&self, span: proc_macro2::Span) -> &str {
        let (a, b) = self.range(span);
        &self.source[a..b]
    }
    fn apply(&mut self, span: proc_macro2::Span, value: String) {
        let (a, b) = self.range(span);
        self.edits.push((a, b, value));
    }
}
impl Rewrite<'_> {
    fn macro_members(&mut self, tokens: proc_macro2::TokenStream) {
        let context = tokens.to_string();
        let mut after_dot = false;
        for token in tokens {
            match token {
                proc_macro2::TokenTree::Group(group) => {
                    self.macro_members(group.stream());
                    after_dot = false;
                }
                proc_macro2::TokenTree::Punct(p) => {
                    after_dot = p.as_char() == '.';
                }
                proc_macro2::TokenTree::Ident(ident) => {
                    if after_dot {
                        let name = ident.to_string();
                        let special = match name.as_str() {
                            "ratio" => {
                                context.contains("turbulence_intensity")
                                    || context.contains("coefficient")
                            }
                            "fraction" => context.contains("emissivity"),
                            "seconds" => context.contains("stiffness_coefficient"),
                            _ => self.fields.contains(name.as_str()),
                        };
                        if special {
                            self.apply(ident.span(), "canonical_value()".into());
                        }
                    }
                    after_dot = false;
                }
                _ => after_dot = false,
            }
        }
    }
}
impl<'ast> Visit<'ast> for Rewrite<'_> {
    fn visit_macro(&mut self, item: &'ast syn::Macro) {
        self.macro_members(item.tokens.clone());
    }

    fn visit_expr_field(&mut self, e: &'ast syn::ExprField) {
        if let syn::Member::Named(m) = &e.member {
            let name = m.to_string();
            let base = self.text(e.base.span());
            let special = match name.as_str() {
                "ratio" => {
                    base.contains("turbulence_intensity")
                        || base.contains("coefficient")
                        || self.path.to_string_lossy().contains("contact")
                }
                "fraction" => base.contains("emissivity"),
                "seconds" => base.contains("stiffness_coefficient"),
                _ => self.fields.contains(name.as_str()),
            };
            if special {
                self.apply(m.span(), "canonical_value()".into());
            }
        }
        visit::visit_expr_field(self, e);
    }
    fn visit_expr_struct(&mut self, e: &'ast syn::ExprStruct) {
        if let Some(name) = e.path.segments.last().map(|s| s.ident.to_string()) {
            if matches!(
                name.as_str(),
                "AnalysisModel" | "SolverAnalysisModel" | "AnalysisModelRealizationInput"
            ) {
                for field in &e.fields {
                    if field.member.to_token_stream().to_string() == "units" {
                        let text = self.text(field.expr.span());
                        if !text.contains("canonical()") && text != "self.units" {
                            if field.colon_token.is_none() {
                                self.apply(
                                    field.span(),
                                    "units: runmat_types::physical::CoordinateUnits::canonical()"
                                        .into(),
                                );
                            } else {
                                self.apply(
                                    field.expr.span(),
                                    "runmat_types::physical::CoordinateUnits::canonical()".into(),
                                );
                            }
                        }
                    }
                }
            }
            if matches!(
                name.as_str(),
                "GeometryInspection" | "ImportedGeometryDefinition"
            ) && !e
                .fields
                .iter()
                .any(|f| f.member.to_token_stream().to_string() == "unit_evidence")
            {
                let (start, _) = self.range(e.brace_token.span.open());
                self.edits.push((
                    start + 1,
                    start + 1,
                    "unit_evidence: Default::default(),".into(),
                ));
            }
            if name == "GeometryImportOptions"
                && e.rest.is_none()
                && !e
                    .fields
                    .iter()
                    .any(|f| f.member.to_token_stream().to_string() == "source_units")
            {
                let (start, _) = self.range(e.brace_token.span.open());
                self.edits
                    .push((start + 1, start + 1, "source_units: None,".into()));
            }
            if name == "ImportedGeometryDefinition"
                && !self
                    .path
                    .to_string_lossy()
                    .contains("semantic/geometry/mod.rs")
            {
                for field in &e.fields {
                    if field.member.to_token_stream().to_string() == "units" {
                        let text = self.text(field.expr.span());
                        if text != "None"
                            && !text.starts_with("Some(")
                            && !text.starts_with("Some (")
                            && !text.contains("uniform_unit()")
                        {
                            let (a, b) = self.range(field.expr.span());
                            self.edits.push((
                                a,
                                a,
                                if field.colon_token.is_none() {
                                    "units: Some("
                                } else {
                                    "Some("
                                }
                                .into(),
                            ));
                            self.edits.push((b, b, ")".into()));
                        }
                    }
                }
            }

            if matches!(
                name.as_str(),
                "ExactCadImportOptions" | "FacetedSolidImportOptions"
            ) {
                for field in &e.fields {
                    if field.member.to_token_stream().to_string() == "source_units" {
                        let text = self.text(field.expr.span());
                        if text != "None"
                            && !text.starts_with("Some(")
                            && !text.starts_with("Some (")
                        {
                            let (a, b) = self.range(field.expr.span());
                            self.edits.push((
                                a,
                                a,
                                if field.colon_token.is_none() {
                                    "source_units: Some("
                                } else {
                                    "Some("
                                }
                                .into(),
                            ));
                            self.edits.push((b, b, ")".into()));
                        }
                    }
                }
            }
            if name == "GeometrySource"
                && !e
                    .fields
                    .iter()
                    .any(|f| f.member.to_token_stream().to_string() == "unit_evidence")
            {
                let (start, _) = self.range(e.brace_token.span.open());
                self.edits.push((
                    start + 1,
                    start + 1,
                    "unit_evidence: Default::default(),".into(),
                ));
            }

            if let Some(&field) = self.quantities.get(name.as_str()) {
                if e.fields.len() == 1
                    && e.rest.is_none()
                    && e.fields[0].member.to_token_stream().to_string() == field
                {
                    let expr = self.text(e.fields[0].expr.span());
                    let path = e.path.to_token_stream().to_string();
                    self.apply(e.span(), format!("{path}::new({expr})"));
                    return;
                }
            }
        }
        visit::visit_expr_struct(self, e);
    }
}
fn paths(p: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if p.is_file() {
        if p.extension().is_some_and(|s| s == "rs") {
            out.push(p.into());
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
                paths(&e.path(), out)?;
            }
        } else if e.path().extension().is_some_and(|s| s == "rs") {
            out.push(e.path());
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fields = BTreeSet::from([
        "meters",
        "radians",
        "kelvin",
        "pascals",
        "volts",
        "amperes",
        "coulombs",
        "webers_per_meter",
        "meters_per_second",
        "meters_per_second_squared",
        "radians_per_second",
        "radians_per_second_squared",
        "newtons",
        "newton_meters",
        "newtons_per_cubic_meter",
        "watts_per_square_meter",
        "watts_per_cubic_meter",
        "watts_per_square_meter_kelvin",
        "pascal_seconds_per_meter",
        "kilograms_per_second",
        "cubic_meters_per_second",
        "amperes_per_square_meter",
        "coulombs_per_cubic_meter",
        "newtons_per_meter",
        "newton_seconds_per_meter",
        "per_second",
        "square_meter_kelvin_per_watt",
        "hertz",
        "kilograms_per_cubic_meter",
        "pascal_seconds",
    ]);
    let quantities = BTreeMap::from([
        ("Length", "meters"),
        ("Displacement", "meters"),
        ("Position", "meters"),
        ("Translation", "meters"),
        ("Temperature", "kelvin"),
        ("TemperatureDifference", "kelvin"),
        ("Rotation", "radians"),
        ("ComplexPhase", "radians"),
        ("Frequency", "hertz"),
        ("Pressure", "pascals"),
    ]);
    let mut files = vec![];
    for root in env::args().skip(1) {
        paths(Path::new(&root), &mut files)?;
    }
    let (mut count, mut edits) = (0, 0);
    for path in files {
        if path
            .to_string_lossy()
            .contains("runmat-types/src/physical/")
        {
            continue;
        }
        let source = fs::read_to_string(&path)?;
        let syntax = syn::parse_file(&source)?;
        let mut offsets = vec![0];
        offsets.extend(source.match_indices('\n').map(|(i, _)| i + 1));
        let mut rewrite = Rewrite {
            source: &source,
            offsets,
            edits: vec![],
            path: &path,
            fields: fields.clone(),
            quantities: quantities.clone(),
        };
        rewrite.visit_file(&syntax);
        if rewrite.edits.is_empty() {
            continue;
        }
        rewrite.edits.sort_by_key(|e| e.0);
        rewrite.edits.dedup();
        let mut output = source.clone();
        let mut upper = source.len();
        for (a, b, value) in rewrite.edits.iter().rev() {
            if *b > upper {
                return Err(format!("overlapping edits {}", path.display()).into());
            }
            output.replace_range(*a..*b, value);
            upper = *a;
        }
        fs::write(&path, output)?;
        count += 1;
        edits += rewrite.edits.len();
    }
    println!("rewrote {edits} canonical accesses/constructors across {count} files");
    Ok(())
}
