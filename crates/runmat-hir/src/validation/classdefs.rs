use crate::{HirClassMember, HirProgram, HirStmt, SemanticError};

pub fn validate_classdefs(prog: &HirProgram) -> Result<(), SemanticError> {
    use std::collections::HashSet;

    fn norm_attr_value(v: &str) -> String {
        let t = v.trim();
        let t = t.trim_matches('\'');
        t.to_ascii_lowercase()
    }

    fn validate_access_value(ctx: &str, v: &str) -> Result<(), SemanticError> {
        match v {
            "public" | "private" => Ok(()),
            other => Err(SemanticError::new(format!(
                "invalid access value '{other}' in {ctx} (allowed: public, private)",
            ))),
        }
    }

    for stmt in &prog.body {
        if let HirStmt::ClassDef {
            name,
            super_class,
            members,
            ..
        } = stmt
        {
            if let Some(sup) = super_class {
                if sup == name {
                    return Err(SemanticError::new(format!(
                        "Class '{name}' cannot inherit from itself"
                    )));
                }
            }

            let mut prop_names: HashSet<String> = HashSet::new();
            let mut method_names: HashSet<String> = HashSet::new();
            for m in members {
                match m {
                    HirClassMember::Properties {
                        names: props,
                        attributes,
                    } => {
                        let mut has_static = false;
                        let mut has_constant = false;
                        let mut _has_transient = false;
                        let mut _has_hidden = false;
                        let mut has_dependent = false;
                        let mut access_default: Option<String> = None;
                        let mut get_access: Option<String> = None;
                        let mut set_access: Option<String> = None;
                        for a in attributes {
                            if a.name.eq_ignore_ascii_case("Static") {
                                has_static = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Constant") {
                                has_constant = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Transient") {
                                _has_transient = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Hidden") {
                                _has_hidden = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Dependent") {
                                has_dependent = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Access") {
                                let v = a.value.as_ref().ok_or_else(|| {
                                    format!(
                                        "Access requires value in class '{name}' properties block",
                                    )
                                })?;
                                let v = norm_attr_value(v);
                                validate_access_value(&format!("class '{name}' properties"), &v)?;
                                access_default = Some(v);
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("GetAccess") {
                                let v = a.value.as_ref().ok_or_else(|| {
                                    format!(
                                        "GetAccess requires value in class '{name}' properties block",
                                    )
                                })?;
                                let v = norm_attr_value(v);
                                validate_access_value(&format!("class '{name}' properties"), &v)?;
                                get_access = Some(v);
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("SetAccess") {
                                let v = a.value.as_ref().ok_or_else(|| {
                                    format!(
                                        "SetAccess requires value in class '{name}' properties block",
                                    )
                                })?;
                                let v = norm_attr_value(v);
                                validate_access_value(&format!("class '{name}' properties"), &v)?;
                                set_access = Some(v);
                                continue;
                            }
                        }
                        if has_static && has_dependent {
                            return Err(SemanticError::new(format!(
                                "class '{name}' properties: attributes 'Static' and 'Dependent' cannot be combined"
                            )));
                        }
                        if has_constant && has_dependent {
                            return Err(SemanticError::new(format!(
                                "class '{name}' properties: attributes 'Constant' and 'Dependent' cannot be combined"
                            )));
                        }
                        let _ = (access_default, get_access, set_access);
                        for p in props {
                            if !prop_names.insert(p.clone()) {
                                return Err(SemanticError::new(format!(
                                    "Duplicate property '{p}' in class {name}"
                                )));
                            }
                            if method_names.contains(p) {
                                return Err(SemanticError::new(format!(
                                    "Name '{p}' used for both property and method in class {name}"
                                )));
                            }
                        }
                    }
                    HirClassMember::Methods { body, attributes } => {
                        let mut _has_static = false;
                        let mut has_abstract = false;
                        let mut has_sealed = false;
                        let mut _has_hidden = false;
                        for a in attributes {
                            if a.name.eq_ignore_ascii_case("Static") {
                                _has_static = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Abstract") {
                                has_abstract = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Sealed") {
                                has_sealed = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Hidden") {
                                _has_hidden = true;
                                continue;
                            }
                            if a.name.eq_ignore_ascii_case("Access") {
                                let v =
                                    a.value.as_ref().ok_or_else(|| {
                                        format!(
                                            "Access requires value in class '{name}' methods block",
                                        )
                                    })?;
                                let v = norm_attr_value(v);
                                validate_access_value(&format!("class '{name}' methods"), &v)?;
                            }
                        }
                        if has_abstract && has_sealed {
                            return Err(SemanticError::new(format!(
                                "class '{name}' methods: attributes 'Abstract' and 'Sealed' cannot be combined"
                            )));
                        }
                        for s in body {
                            if let HirStmt::Function { name: fname, .. } = s {
                                if !method_names.insert(fname.clone()) {
                                    return Err(SemanticError::new(format!(
                                        "Duplicate method '{fname}' in class {name}"
                                    )));
                                }
                                if prop_names.contains(fname) {
                                    return Err(SemanticError::new(format!(
                                        "Name '{fname}' used for both property and method in class {name}"
                                    )));
                                }
                            }
                        }
                    }
                    HirClassMember::Events { attributes, names } => {
                        for ev in names {
                            if method_names.contains(ev) || prop_names.contains(ev) {
                                return Err(SemanticError::new(format!(
                                    "Name '{ev}' used for event conflicts with existing member in class {name}"
                                )));
                            }
                        }
                        let mut seen = std::collections::HashSet::new();
                        for ev in names {
                            if !seen.insert(ev) {
                                return Err(SemanticError::new(format!(
                                    "Duplicate event '{ev}' in class {name}"
                                )));
                            }
                        }
                        let _ = attributes;
                    }
                    HirClassMember::Enumeration { attributes, names } => {
                        for en in names {
                            if method_names.contains(en) || prop_names.contains(en) {
                                return Err(SemanticError::new(format!(
                                    "Name '{en}' used for enumeration conflicts with existing member in class {name}"
                                )));
                            }
                        }
                        let mut seen = std::collections::HashSet::new();
                        for en in names {
                            if !seen.insert(en) {
                                return Err(SemanticError::new(format!(
                                    "Duplicate enumeration '{en}' in class {name}"
                                )));
                            }
                        }
                        let _ = attributes;
                    }
                    HirClassMember::Arguments { attributes, names } => {
                        for ar in names {
                            if method_names.contains(ar) || prop_names.contains(ar) {
                                return Err(SemanticError::new(format!(
                                    "Name '{ar}' used for arguments conflicts with existing member in class {name}"
                                )));
                            }
                        }
                        let _ = attributes;
                    }
                }
            }
        }
    }
    Ok(())
}
