//! Install the linked Native IR's immutable class declarations in this session.

use std::collections::{HashMap, HashSet};

use runmat_native_codegen::NativeClassDeclaration;
use runmat_runtime::class_registry::{
    CheckedClassRegistration, RuntimeClass, RuntimeMethod, RuntimeProperty,
};
use runmat_types::ClassIdentity;
use runmat_value::{IntValue, Value};

use super::native_error;

pub(super) fn install(
    executor: &runmat_native_executor::NativeExecutor,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<(), Box<runmat_runtime::RuntimeError>> {
    let classes = executor.class_declarations();
    admit_supported_defaults(classes)?;
    let by_name = classes
        .iter()
        .map(|class| (&class.class, class))
        .collect::<HashMap<_, _>>();
    let mut installed = HashSet::new();
    let mut visiting = HashSet::new();
    let mut batch = Vec::with_capacity(classes.len());
    let _scope = runtime.enter();
    for class in classes {
        visit(
            class,
            executor,
            &by_name,
            &mut installed,
            &mut visiting,
            &mut batch,
        )?;
    }
    runmat_runtime::class_registry::register_checked_classes(batch).map_err(Box::new)
}

fn admit_supported_defaults(
    classes: &[NativeClassDeclaration],
) -> Result<(), Box<runmat_runtime::RuntimeError>> {
    if classes
        .iter()
        .flat_map(|class| &class.properties)
        .any(|property| property.has_default && property.default_literal.is_none())
    {
        return Err(Box::new(native_error(
            "standalone class property default has no supported literal projection",
        )));
    }
    Ok(())
}

fn visit<'a>(
    class: &'a NativeClassDeclaration,
    executor: &runmat_native_executor::NativeExecutor,
    by_name: &HashMap<&'a ClassIdentity, &'a NativeClassDeclaration>,
    installed: &mut HashSet<ClassIdentity>,
    visiting: &mut HashSet<ClassIdentity>,
    batch: &mut Vec<CheckedClassRegistration>,
) -> Result<(), Box<runmat_runtime::RuntimeError>> {
    if installed.contains(&class.class) {
        return Ok(());
    }
    if !visiting.insert(class.class.clone()) {
        return Err(Box::new(native_error(
            "linked class inheritance contains a cycle",
        )));
    }
    if let Some(parent) = class.parent.as_ref().and_then(|parent| by_name.get(parent)) {
        visit(parent, executor, by_name, installed, visiting, batch)?;
    }
    let mut properties = HashMap::new();
    for property in &class.properties {
        let default_value = property
            .default_literal
            .as_ref()
            .map(|literal| match literal {
                runmat_native_codegen::NativeClassDefaultLiteral::Num(value) => Value::Num(*value),
                runmat_native_codegen::NativeClassDefaultLiteral::Int(value) => {
                    Value::Int(IntValue::from(value))
                }
                runmat_native_codegen::NativeClassDefaultLiteral::Bool(value) => {
                    Value::Bool(*value)
                }
                runmat_native_codegen::NativeClassDefaultLiteral::String(value) => {
                    Value::String(value.clone())
                }
            });
        let attributes = &property.attributes;
        if properties
            .insert(
                property.name.clone(),
                RuntimeProperty {
                    name: property.name.clone(),
                    is_static: attributes.is_static,
                    is_constant: attributes.is_constant,
                    is_dependent: attributes.is_dependent,
                    get_access: attributes.get_access,
                    set_access: attributes.set_access,
                    default_value,
                },
            )
            .is_some()
        {
            return Err(Box::new(native_error("linked class repeats a property")));
        }
    }
    let mut methods = HashMap::new();
    for method in &class.methods {
        let function_name = executor
            .function_metadata(method.function)
            .ok_or_else(|| native_error("linked class method has no executable body"))?
            .name
            .clone();
        if methods
            .insert(
                method.name.clone(),
                RuntimeMethod {
                    name: method.name.clone(),
                    is_static: method.is_static,
                    is_abstract: method.attributes.is_abstract,
                    is_sealed: method.attributes.is_sealed,
                    access: method.attributes.access,
                    function_name,
                    implicit_class_argument: None,
                    role: if class
                        .constructor
                        .as_ref()
                        .is_some_and(|constructor| constructor.function == method.function)
                    {
                        runmat_types::ClassMethodRole::Constructor
                    } else {
                        runmat_types::ClassMethodRole::Ordinary
                    },
                },
            )
            .is_some()
        {
            return Err(Box::new(native_error("linked class repeats a method")));
        }
    }
    let events = class
        .events
        .iter()
        .map(|event| (event.name.clone(), event.clone()))
        .collect();
    let definition = RuntimeClass {
        name: class.class.clone(),
        parent: class.parent.clone(),
        properties,
        methods,
        events,
    };
    batch.push(CheckedClassRegistration {
        class: definition,
        is_sealed: class.is_sealed,
        is_abstract: class.is_abstract,
        enumerations: class
            .enumerations
            .iter()
            .map(|name| name.display_name().to_string())
            .collect(),
    });
    visiting.remove(&class.class);
    installed.insert(class.class.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_class_default_rejects_before_any_registration() {
        let class = NativeClassDeclaration {
            class: ClassIdentity::new("DefaultClass").expect("class"),
            parent: None,
            is_sealed: false,
            is_abstract: false,
            properties: vec![runmat_native_codegen::NativeClassProperty {
                name: runmat_types::MemberName::from("value"),
                attributes: runmat_types::PropertyAttributes::default(),
                has_default: true,
                default_literal: None,
            }],
            methods: Vec::new(),
            events: Vec::new(),
            enumerations: Vec::new(),
            constructor: None,
        };
        assert!(admit_supported_defaults(&[class]).is_err());
    }
}
