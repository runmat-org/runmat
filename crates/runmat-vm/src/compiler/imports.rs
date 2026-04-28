//! Import and static-resolution lowering.

use crate::compiler::core::Compiler;
use crate::compiler::CompileError;

pub(crate) struct CallImportResolution {
    pub resolved: String,
    pub static_candidates: Vec<(String, String)>,
}

impl Compiler {
    fn join_import_path(parts: &[String]) -> String {
        parts.join(".")
    }

    pub(crate) fn resolve_unqualified_static_property(
        &self,
        name: &str,
    ) -> Result<Option<String>, CompileError> {
        let mut classes: Vec<String> = Vec::new();
        for (path, wildcard) in &self.imports {
            if !*wildcard || path.is_empty() {
                continue;
            }
            let cls = Self::join_import_path(path);
            if let Some((p, _owner)) = runmat_builtins::lookup_property(&cls, name) {
                if p.is_static {
                    classes.push(cls);
                }
            }
        }
        if classes.len() > 1 {
            return Err(self.compile_error(format!(
                "ambiguous unqualified static property '{}' via Class.* imports: {}",
                name,
                classes.join(", ")
            )));
        }
        Ok(classes.pop())
    }

    pub(crate) fn resolve_call_imports(
        &self,
        name: &str,
    ) -> Result<CallImportResolution, CompileError> {
        let mut resolved = name.to_string();
        let mut static_candidates: Vec<(String, String)> = Vec::new();

        if !runmat_builtins::builtin_functions()
            .iter()
            .any(|b| b.name == resolved)
        {
            let mut specific_candidates: Vec<String> = Vec::new();
            for (path, wildcard) in &self.imports {
                if *wildcard {
                    continue;
                }
                if path.last().map(|s| s.as_str()) == Some(name) {
                    let qual = Self::join_import_path(path);
                    if runmat_builtins::builtin_functions()
                        .iter()
                        .any(|b| b.name == qual)
                        || self.functions.contains_key(&qual)
                    {
                        specific_candidates.push(qual);
                    }
                }
            }
            if specific_candidates.len() > 1 {
                return Err(self.compile_error(format!(
                    "ambiguous unqualified reference '{}' via imports: {}",
                    name,
                    specific_candidates.join(", ")
                )));
            }
            if specific_candidates.len() == 1 {
                resolved = specific_candidates.remove(0);
            } else {
                let mut wildcard_candidates: Vec<String> = Vec::new();
                for (path, wildcard) in &self.imports {
                    if !*wildcard || path.is_empty() {
                        continue;
                    }
                    let cls = Self::join_import_path(path);
                    let qual = format!("{}.{}", cls, name);
                    if runmat_builtins::builtin_functions()
                        .iter()
                        .any(|b| b.name == qual)
                        || self.functions.contains_key(&qual)
                    {
                        wildcard_candidates.push(qual);
                    }
                    if let Some((m, _owner)) = runmat_builtins::lookup_method(&cls, name) {
                        if m.is_static {
                            static_candidates.push((cls.clone(), name.to_string()));
                        }
                    }
                }
                if wildcard_candidates.len() > 1 {
                    return Err(self.compile_error(format!(
                        "ambiguous unqualified reference '{}' via wildcard imports: {}",
                        name,
                        wildcard_candidates.join(", ")
                    )));
                }
                if wildcard_candidates.len() == 1 {
                    resolved = wildcard_candidates.remove(0);
                }
            }
        }

        Ok(CallImportResolution {
            resolved,
            static_candidates,
        })
    }
}
