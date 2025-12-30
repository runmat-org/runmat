use anyhow::{Context, Result, bail};
use clap::Parser;
use proc_macro2::Span;
use quote::quote;
use std::fs;
use std::path::{Path, PathBuf};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{
    Attribute, File, ItemConst, ItemFn, ItemMod, LitStr, Meta, MetaList, NestedMeta, parse_quote,
};
use walkdir::WalkDir;

#[derive(Parser)]
struct Args {
    /// Only check for missing builtin_path entries (do not modify files)
    #[arg(long)]
    check: bool,

    /// Path to the runmat-runtime/src directory
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
    for entry in WalkDir::new(&src_root)
        .into_iter()
        .filter_entry(|e| !is_hidden(e.path()))
    {
        let entry = entry?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|ext| ext == "rs") {
            rust_files.push(entry.into_path());
        }
    }

    let mut missing = Vec::new();
    for file in rust_files {
        let changed = process_file(&file, &src_root, args.check)
            .with_context(|| format!("processing {}", file.display()))?;
        if changed {
            missing.push(file);
        }
    }

    if args.check && !missing.is_empty() {
        bail!(
            "missing builtin_path in {} files:\n{}",
            missing.len(),
            missing
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

fn process_file(path: &Path, src_root: &Path, check_only: bool) -> Result<bool> {
    let mut content = fs::read_to_string(path)?;
    let relative = path.strip_prefix(src_root)?;
    let base_components = module_components(relative);
    let line_offsets = line_offsets(&content);
    let syntax: File = syn::parse_file(&content)?;

    let mut collector = AttrCollector::new(&line_offsets, base_components);
    collector.visit_file(&syntax);
    let edits = collector.into_edits();

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

fn module_components(rel_path: &Path) -> Vec<String> {
    let mut components = Vec::new();
    let mut parts: Vec<String> = rel_path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    if let Some(file) = parts.pop() {
        if file != "mod.rs" && file != "lib.rs" {
            components.extend(parts.into_iter().map(|p| sanitize_component(&p)));
            let name = sanitize_component(file.trim_end_matches(".rs"));
            components.push(name);
            return components;
        }
        components.extend(parts.into_iter().map(|p| sanitize_component(&p)));
        if file == "lib.rs" && components.is_empty() {
            return components;
        }
        return components;
    }
    components
}

fn sanitize_component(name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if is_keyword(&clean) {
        format!("r#{}", clean)
    } else {
        clean
    }
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

struct AttrCollector<'a> {
    line_offsets: &'a [usize],
    base_components: Vec<String>,
    module_stack: Vec<String>,
    edits: Vec<Edit>,
}

impl<'a> AttrCollector<'a> {
    fn new(line_offsets: &'a [usize], base_components: Vec<String>) -> Self {
        Self {
            line_offsets,
            base_components,
            module_stack: Vec::new(),
            edits: Vec::new(),
        }
    }

    fn module_path(&self) -> String {
        let mut parts = Vec::new();
        parts.push("crate".to_string());
        parts.extend(self.base_components.iter().cloned());
        parts.extend(self.module_stack.iter().cloned());
        parts.retain(|part| !part.is_empty());
        parts.join("::")
    }

    fn process_attrs(&mut self, attrs: &[Attribute]) {
        let module_path = self.module_path();
        for attr in attrs {
            let path = attr_path(attr);
            match path.as_str() {
                "runtime_builtin" | "runmat_macros::runtime_builtin" => {
                    self.inject_attr(attr, &module_path);
                }
                "runtime_constant" | "runmat_macros::runtime_constant" => {
                    self.inject_attr(attr, &module_path);
                }
                "register_gpu_spec" | "runmat_macros::register_gpu_spec" => {
                    self.inject_attr(attr, &module_path);
                }
                "register_fusion_spec" | "runmat_macros::register_fusion_spec" => {
                    self.inject_attr(attr, &module_path);
                }
                "register_doc_text" | "runmat_macros::register_doc_text" => {
                    self.inject_attr(attr, &module_path);
                }
                "cfg_attr" => {
                    self.inject_cfg_attr(attr, &module_path);
                }
                _ => {}
            }
        }
    }

    fn inject_attr(&mut self, attr: &Attribute, module_path: &str) {
        let start = span_start(attr.span(), self.line_offsets);
        let end = span_end(attr.span(), self.line_offsets);
        if let Some(replacement) = add_wasm_argument(attr, module_path) {
            self.edits.push(Edit {
                start,
                end,
                replacement,
            });
        }
    }

    fn inject_cfg_attr(&mut self, attr: &Attribute, module_path: &str) {
        let start = span_start(attr.span(), self.line_offsets);
        let end = span_end(attr.span(), self.line_offsets);
        if let Some(replacement) = add_wasm_to_cfg_attr(attr, module_path) {
            self.edits.push(Edit {
                start,
                end,
                replacement,
            });
        }
    }

    fn into_edits(self) -> Vec<Edit> {
        self.edits
    }
}

impl<'ast> Visit<'ast> for AttrCollector<'_> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.process_attrs(&node.attrs);
        syn::visit::visit_item_fn(self, node);
    }

    fn visit_item_const(&mut self, node: &'ast ItemConst) {
        self.process_attrs(&node.attrs);
        syn::visit::visit_item_const(self, node);
    }

    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        self.process_attrs(&node.attrs);
        if let Some((_, items)) = &node.content {
            self.module_stack.push(node.ident.to_string());
            for item in items {
                self.visit_item(item);
            }
            self.module_stack.pop();
        }
    }
}

fn attr_path(attr: &Attribute) -> String {
    attr.path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn add_wasm_argument(attr: &Attribute, module_path: &str) -> Option<String> {
    let meta = attr.parse_meta().ok()?;
    match meta {
        Meta::List(mut list) => {
            if ensure_wasm_meta(&mut list, module_path) {
                let tokens = quote!(#list);
                Some(format!("#[{}]", tokens))
            } else {
                None
            }
        }
        Meta::Path(path) => {
            let mut list = MetaList {
                path,
                paren_token: Default::default(),
                nested: Punctuated::new(),
            };
            if ensure_wasm_meta(&mut list, module_path) {
                let tokens = quote!(#list);
                Some(format!("#[{}]", tokens))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn add_wasm_to_cfg_attr(attr: &Attribute, module_path: &str) -> Option<String> {
    let meta = attr.parse_meta().ok()?;
    match meta {
        Meta::List(mut list) if path_to_string(&list.path) == "cfg_attr" => {
            let mut changed = false;
            for nested in list.nested.iter_mut() {
                if let NestedMeta::Meta(Meta::List(inner)) = nested {
                    let path = path_to_string(&inner.path);
                    if needs_builtin_path(&path) {
                        changed |= ensure_wasm_meta(inner, module_path);
                    }
                }
            }
            if changed {
                let tokens = quote!(#list);
                Some(format!("#[{}]", tokens))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn ensure_wasm_meta(list: &mut MetaList, module_path: &str) -> bool {
    let has = list.nested.iter().any(|nested| {
        matches!(
            nested,
            NestedMeta::Meta(Meta::NameValue(nv)) if nv.path.is_ident("builtin_path")
        )
    });
    if has {
        return false;
    }
    let lit = LitStr::new(module_path, Span::call_site());
    let new_meta: Meta = parse_quote!(builtin_path = #lit);
    list.nested.push(NestedMeta::Meta(new_meta));
    true
}

fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn needs_builtin_path(path: &str) -> bool {
    matches!(
        path,
        "runtime_builtin"
            | "runmat_macros::runtime_builtin"
            | "runtime_constant"
            | "runmat_macros::runtime_constant"
            | "register_gpu_spec"
            | "runmat_macros::register_gpu_spec"
            | "register_fusion_spec"
            | "runmat_macros::register_fusion_spec"
            | "register_doc_text"
            | "runmat_macros::register_doc_text"
    )
}

fn is_keyword(ident: &str) -> bool {
    matches!(
        ident,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
            | "abstract"
            | "become"
            | "box"
            | "do"
            | "final"
            | "macro"
            | "override"
            | "priv"
            | "try"
            | "typeof"
            | "unsized"
            | "virtual"
            | "yield"
    )
}
