use runmat_builtins::{self, Access, ClassDef, MethodDef, PropertyDef};
use runmat_runtime::RuntimeError;

pub fn register_class(
    name: String,
    super_class: Option<String>,
    properties: Vec<(String, bool, String, String)>,
    methods: Vec<(String, String, bool, String)>,
) -> Result<(), RuntimeError> {
    let mut prop_map = std::collections::HashMap::new();
    for (p, is_static, get_access, set_access) in properties {
        let gacc = if get_access.eq_ignore_ascii_case("private") {
            Access::Private
        } else {
            Access::Public
        };
        let sacc = if set_access.eq_ignore_ascii_case("private") {
            Access::Private
        } else {
            Access::Public
        };
        let (is_dep, clean_name) = if let Some(stripped) = p.strip_prefix("@dep:") {
            (true, stripped.to_string())
        } else {
            (false, p.clone())
        };
        prop_map.insert(
            clean_name.clone(),
            PropertyDef {
                name: clean_name,
                is_static,
                is_dependent: is_dep,
                get_access: gacc,
                set_access: sacc,
                default_value: None,
            },
        );
    }
    let mut method_map = std::collections::HashMap::new();
    for (mname, fname, is_static, access) in methods {
        let access = if access.eq_ignore_ascii_case("private") {
            Access::Private
        } else {
            Access::Public
        };
        method_map.insert(
            mname.clone(),
            MethodDef {
                name: mname,
                is_static,
                access,
                function_name: fname,
            },
        );
    }
    let def = ClassDef {
        name: name.clone(),
        parent: super_class.clone(),
        properties: prop_map,
        methods: method_map,
    };
    runmat_builtins::register_class(def);
    Ok(())
}
