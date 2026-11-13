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

pub(crate) fn checked_class_registration(
    registration: ClassRegistration<RuntimeClassPropertyRegistration>,
) -> runmat_runtime::class_registry::CheckedClassRegistration {
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
                role: method.role,
                is_static: method.is_static,
                is_abstract: method.is_abstract,
                is_sealed: method.is_sealed,
                access: method.access,
                function_name: method.function_name,
                implicit_class_argument: None,
            },
        );
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
    runmat_runtime::class_registry::CheckedClassRegistration {
        class: def,
        method_bindings: std::collections::HashMap::new(),
        is_sealed,
        is_abstract,
        enumerations,
    }
}
