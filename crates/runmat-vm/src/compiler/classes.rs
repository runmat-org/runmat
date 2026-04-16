//! Class lowering.

use crate::compiler::core::Compiler;
use crate::compiler::CompileError;
use crate::instr::Instr;

impl Compiler {
    fn attr_access_from_str(s: &str) -> runmat_builtins::Access {
        match s.to_ascii_lowercase().as_str() {
            "private" => runmat_builtins::Access::Private,
            _ => runmat_builtins::Access::Public,
        }
    }

    pub(crate) fn parse_prop_attrs(
        attrs: &Vec<runmat_parser::Attr>,
    ) -> (bool, bool, String, String) {
        let mut is_static = false;
        let mut is_dependent = false;
        let mut get_acc = runmat_builtins::Access::Public;
        let mut set_acc = runmat_builtins::Access::Public;
        for a in attrs {
            if a.name.eq_ignore_ascii_case("Static") {
                is_static = true;
            }
            if a.name.eq_ignore_ascii_case("Dependent") {
                is_dependent = true;
            }
            if a.name.eq_ignore_ascii_case("Access") {
                if let Some(v) = &a.value {
                    let acc = Self::attr_access_from_str(v.trim_matches('\'').trim());
                    get_acc = acc.clone();
                    set_acc = acc;
                }
            }
            if a.name.eq_ignore_ascii_case("GetAccess") {
                if let Some(v) = &a.value {
                    get_acc = Self::attr_access_from_str(v.trim_matches('\'').trim());
                }
            }
            if a.name.eq_ignore_ascii_case("SetAccess") {
                if let Some(v) = &a.value {
                    set_acc = Self::attr_access_from_str(v.trim_matches('\'').trim());
                }
            }
        }
        let gs = match get_acc {
            runmat_builtins::Access::Private => "private".to_string(),
            _ => "public".to_string(),
        };
        let ss = match set_acc {
            runmat_builtins::Access::Private => "private".to_string(),
            _ => "public".to_string(),
        };
        (is_static, is_dependent, gs, ss)
    }

    pub(crate) fn parse_method_attrs(attrs: &Vec<runmat_parser::Attr>) -> (bool, String) {
        let mut is_static = false;
        let mut access = runmat_builtins::Access::Public;
        for a in attrs {
            if a.name.eq_ignore_ascii_case("Static") {
                is_static = true;
            }
            if a.name.eq_ignore_ascii_case("Access") {
                if let Some(v) = &a.value {
                    access = Self::attr_access_from_str(v.trim_matches('\'').trim());
                }
            }
        }
        let acc_str = match access {
            runmat_builtins::Access::Private => "private".to_string(),
            _ => "public".to_string(),
        };
        (is_static, acc_str)
    }

    pub(crate) fn compile_class_def(
        &mut self,
        name: &str,
        super_class: &Option<String>,
        members: &[runmat_hir::HirClassMember],
    ) -> Result<(), CompileError> {
        let mut props: Vec<(String, bool, String, String)> = Vec::new();
        let mut methods: Vec<(String, String, bool, String)> = Vec::new();
        for m in members {
            match m {
                runmat_hir::HirClassMember::Properties { names, attributes } => {
                    let (is_static, is_dependent, get_access, set_access) =
                        Self::parse_prop_attrs(attributes);
                    for n in names {
                        let enc = if is_dependent {
                            format!("@dep:{n}")
                        } else {
                            n.clone()
                        };
                        props.push((enc, is_static, get_access.clone(), set_access.clone()));
                    }
                }
                runmat_hir::HirClassMember::Methods { body, attributes } => {
                    let (is_static, access) = Self::parse_method_attrs(attributes);
                    for s in body {
                        if let runmat_hir::HirStmt::Function { name: mname, .. } = s {
                            methods.push((mname.clone(), mname.clone(), is_static, access.clone()));
                        }
                    }
                }
                _ => {}
            }
        }
        self.emit(Instr::RegisterClass {
            name: name.to_string(),
            super_class: super_class.clone(),
            properties: props,
            methods,
        });
        Ok(())
    }
}
