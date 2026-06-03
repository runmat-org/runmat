use crate::interpreter::errors::mex;
use runmat_builtins::{self, Access, ClassDef, MethodDef, PropertyDef, Value};
use runmat_runtime::RuntimeError;

pub fn register_class(
    name: String,
    super_class: Option<String>,
    is_sealed: bool,
    is_abstract: bool,
    properties: Vec<(String, bool, bool, Option<Value>, String, String)>,
    methods: Vec<(String, String, bool, bool, bool, String)>,
    enumerations: Vec<String>,
) -> Result<(), RuntimeError> {
    if let Some(parent) = super_class.as_deref() {
        if runmat_builtins::is_class_sealed(parent) {
            return Err(mex(
                "RunMat:ClassSealed",
                &format!("Cannot subclass sealed class '{}'.", parent),
            ));
        }
    }
    let mut prop_map = std::collections::HashMap::new();
    for (p, is_static, is_constant, default_value, get_access, set_access) in properties {
        let gacc = if get_access.eq_ignore_ascii_case("private") {
            Access::Private
        } else if get_access.eq_ignore_ascii_case("protected") {
            Access::Protected
        } else {
            Access::Public
        };
        let sacc = if set_access.eq_ignore_ascii_case("private") {
            Access::Private
        } else if set_access.eq_ignore_ascii_case("protected") {
            Access::Protected
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
                is_constant,
                is_dependent: is_dep,
                get_access: gacc,
                set_access: sacc,
                default_value,
            },
        );
    }
    let mut method_map = std::collections::HashMap::new();
    for (mname, fname, is_static, is_method_abstract, is_method_sealed, access) in methods {
        let access = if access.eq_ignore_ascii_case("private") {
            Access::Private
        } else if access.eq_ignore_ascii_case("protected") {
            Access::Protected
        } else {
            Access::Public
        };
        method_map.insert(
            mname.clone(),
            MethodDef {
                name: mname,
                is_static,
                is_abstract: is_method_abstract,
                is_sealed: is_method_sealed,
                access,
                function_name: fname,
                implicit_class_argument: None,
            },
        );
    }

    let inherited_sealed = collect_inherited_sealed_methods(super_class.as_deref());
    if let Some(conflict) = method_map
        .keys()
        .find(|method_name| inherited_sealed.contains(*method_name))
    {
        return Err(mex(
            "RunMat:MethodSealed",
            &format!(
                "Class '{}' cannot override sealed method '{}'.",
                name, conflict
            ),
        ));
    }

    if !is_abstract {
        let mut required_abstract = collect_required_abstract_methods(super_class.as_deref());
        for (method_name, method) in &method_map {
            if method.is_abstract {
                required_abstract.insert(method_name.clone());
            } else {
                required_abstract.remove(method_name);
            }
        }
        if let Some(missing) = required_abstract.into_iter().next() {
            return Err(mex(
                "RunMat:AbstractMethodMissing",
                &format!(
                    "Class '{}' must implement abstract method '{}'.",
                    name, missing
                ),
            ));
        }
    }
    let def = ClassDef {
        name: name.clone(),
        parent: super_class.clone(),
        properties: prop_map,
        methods: method_map,
    };
    runmat_builtins::register_class_with_modifiers(def, is_sealed, is_abstract);
    runmat_builtins::register_class_enumerations(&name, enumerations);
    Ok(())
}

fn collect_required_abstract_methods(
    super_class: Option<&str>,
) -> std::collections::HashSet<String> {
    let mut lineage = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut cursor = super_class.map(str::to_string);
    while let Some(class_name) = cursor {
        if !visited.insert(class_name.clone()) {
            break;
        }
        let Some(class_def) = runmat_builtins::get_class(&class_name) else {
            break;
        };
        cursor = class_def.parent.clone();
        lineage.push(class_def);
    }
    lineage.reverse();
    let mut required = std::collections::HashSet::new();
    for class_def in lineage {
        for (method_name, method) in class_def.methods {
            if method.is_abstract {
                required.insert(method_name);
            } else {
                required.remove(&method_name);
            }
        }
    }
    required
}

fn collect_inherited_sealed_methods(
    super_class: Option<&str>,
) -> std::collections::HashSet<String> {
    let mut sealed = std::collections::HashSet::new();
    let mut visited = std::collections::HashSet::new();
    let mut cursor = super_class.map(str::to_string);
    while let Some(class_name) = cursor {
        if !visited.insert(class_name.clone()) {
            break;
        }
        let Some(class_def) = runmat_builtins::get_class(&class_name) else {
            break;
        };
        for (method_name, method) in &class_def.methods {
            if method.is_sealed {
                sealed.insert(method_name.clone());
            }
        }
        cursor = class_def.parent;
    }
    sealed
}
