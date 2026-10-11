use crate::interpreter::errors::mex;
use runmat_runtime::RuntimeError;
use runmat_types::{ClassIdentity, MemberAccess};
use runmat_value::Value;

pub struct RuntimeClassPropertyRegistration {
    pub name: runmat_types::MemberName,
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub default_value: Option<Value>,
    pub get_access: MemberAccess,
    pub set_access: MemberAccess,
}

pub struct ClassRegistration<P> {
    pub name: ClassIdentity,
    pub super_class: Option<ClassIdentity>,
    pub is_sealed: bool,
    pub is_abstract: bool,
    pub properties: Vec<P>,
    pub methods: Vec<crate::bytecode::instr::BytecodeClassMethod>,
    pub events: Vec<runmat_types::EventDeclaration>,
    pub enumerations: Vec<String>,
}

impl<P> ClassRegistration<P> {
    pub fn map_properties<Q>(self, convert: impl FnMut(P) -> Q) -> ClassRegistration<Q> {
        ClassRegistration {
            name: self.name,
            super_class: self.super_class,
            is_sealed: self.is_sealed,
            is_abstract: self.is_abstract,
            properties: self.properties.into_iter().map(convert).collect(),
            methods: self.methods,
            events: self.events,
            enumerations: self.enumerations,
        }
    }
}

pub fn register_class(
    registration: ClassRegistration<RuntimeClassPropertyRegistration>,
) -> Result<(), RuntimeError> {
    let ClassRegistration {
        name,
        super_class,
        is_sealed,
        is_abstract,
        properties,
        methods,
        events,
        enumerations,
    } = registration;
    if let Some(parent) = super_class.as_ref() {
        if runmat_runtime::class_registry::is_class_sealed(parent) {
            return Err(mex(
                "RunMat:ClassSealed",
                &format!("Cannot subclass sealed class '{}'.", parent),
            ));
        }
    }
    let mut prop_map = std::collections::HashMap::new();
    for property in properties {
        prop_map.insert(
            property.name.clone(),
            runmat_runtime::class_registry::RuntimeProperty {
                name: property.name,
                is_static: property.is_static,
                is_constant: property.is_constant,
                is_dependent: property.is_dependent,
                get_access: property.get_access,
                set_access: property.set_access,
                default_value: property.default_value,
            },
        );
    }
    let mut method_map = std::collections::HashMap::new();
    for method in methods {
        method_map.insert(
            method.name.clone(),
            runmat_runtime::class_registry::RuntimeMethod {
                name: method.name,
                is_static: method.is_static,
                is_abstract: method.is_abstract,
                is_sealed: method.is_sealed,
                access: method.access,
                function_name: method.function_name,
                implicit_class_argument: None,
            },
        );
    }

    let inherited_sealed = collect_inherited_sealed_methods(super_class.as_ref());
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
        let mut required_abstract = collect_required_abstract_methods(super_class.as_ref());
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
    let def = runmat_runtime::class_registry::RuntimeClass {
        events: events
            .into_iter()
            .map(|event| (event.name.clone(), event))
            .collect(),
        name: name.clone(),
        parent: super_class.clone(),
        properties: prop_map,
        methods: method_map,
    };
    runmat_runtime::class_registry::register_class_with_modifiers(def, is_sealed, is_abstract);
    runmat_runtime::class_registry::register_class_enumerations(&name, enumerations);
    Ok(())
}

fn collect_required_abstract_methods(
    super_class: Option<&ClassIdentity>,
) -> std::collections::HashSet<runmat_types::MethodName> {
    let mut lineage = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut cursor = super_class.cloned();
    while let Some(class_name) = cursor {
        if !visited.insert(class_name.clone()) {
            break;
        }
        let Some(class_def) = runmat_runtime::class_registry::get_class(&class_name) else {
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
    super_class: Option<&ClassIdentity>,
) -> std::collections::HashSet<runmat_types::MethodName> {
    let mut sealed = std::collections::HashSet::new();
    let mut visited = std::collections::HashSet::new();
    let mut cursor = super_class.cloned();
    while let Some(class_name) = cursor {
        if !visited.insert(class_name.clone()) {
            break;
        }
        let Some(class_def) = runmat_runtime::class_registry::get_class(&class_name) else {
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
