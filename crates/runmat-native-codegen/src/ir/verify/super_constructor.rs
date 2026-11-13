//! Check the immutable class projection against Native IR owners and calls.

use std::collections::BTreeMap;

use runmat_types::{ClassIdentity, ClassMethodRole, FunctionId, MemberAccess};

use crate::{NativeAssembly, NativeCodegenError, NativeCodegenResult, NativeFunction};

pub(super) fn verify(
    assembly: &NativeAssembly,
    functions: &BTreeMap<FunctionId, &NativeFunction>,
) -> NativeCodegenResult<()> {
    let declarations = &assembly.class_declarations;
    if declarations
        .windows(2)
        .any(|pair| pair[0].class >= pair[1].class)
    {
        return Err(error("class declarations are not unique and ordered"));
    }
    let classes = declarations
        .iter()
        .map(|declaration| (&declaration.class, declaration))
        .collect::<BTreeMap<_, _>>();
    for class in classes.keys() {
        let mut visited = std::collections::BTreeSet::new();
        let mut cursor = Some(*class);
        while let Some(identity) = cursor {
            if !visited.insert(identity) {
                return Err(error("linked class inheritance contains a cycle"));
            }
            cursor = classes
                .get(identity)
                .and_then(|declaration| declaration.parent.as_ref());
        }
    }
    for declaration in declarations {
        if ClassIdentity::new(declaration.class.display_name())
            .ok()
            .as_ref()
            != Some(&declaration.class)
            || declaration.parent.as_ref() == Some(&declaration.class)
            || declaration.parent.as_ref().is_some_and(|parent| {
                ClassIdentity::new(parent.display_name()).ok().as_ref() != Some(parent)
            })
        {
            return Err(error("class or parent identity is invalid"));
        }
        // An unused companion declaration may have an external parent that
        // generic native never executes. The selected constructor call is
        // checked below; standalone AOT installation closes every class
        // parent through the checked runtime batch registrar before effects.
        let mut properties = std::collections::HashSet::new();
        for property in &declaration.properties {
            if !properties.insert(&property.name) {
                return Err(error("class repeats a property"));
            }
            if !property.has_default && property.default_literal.is_some() {
                return Err(error(
                    "class property has a default absent from its declaration",
                ));
            }
        }
        let mut methods = std::collections::HashSet::new();
        for method in &declaration.methods {
            if !methods.insert(&method.name) {
                return Err(error("class repeats a method"));
            }
            let Some(function) = functions.get(&method.function) else {
                return Err(error("linked class method is absent from Native IR"));
            };
            if function.class_method_owner.as_ref().is_none_or(|owner| {
                owner.declaring_class != declaration.class
                    || owner.method != method.name
                    || owner.is_static != method.is_static
            }) {
                return Err(error(
                    "class method disagrees with its typed function owner",
                ));
            }
        }
        let mut events = std::collections::HashSet::new();
        if declaration
            .events
            .iter()
            .any(|event| !events.insert(&event.name))
        {
            return Err(error("class repeats an event"));
        }
        let mut enumerations = std::collections::HashSet::new();
        if declaration
            .enumerations
            .iter()
            .any(|name| !enumerations.insert(name))
        {
            return Err(error("class repeats an enumeration"));
        }
        if let Some(constructor) = &declaration.constructor {
            let Some(function) = functions.get(&constructor.function) else {
                return Err(error("declared constructor is absent from Native IR"));
            };
            let declared_method = declaration.methods.iter().find(|method| {
                method.function == constructor.function
                    && method.attributes.access == constructor.access
                    && !method.is_static
            });
            if function.class_method_owner.as_ref().is_none_or(|owner| {
                owner.role != ClassMethodRole::Constructor
                    || owner.is_static
                    || owner.declaring_class != declaration.class
                    || declared_method.is_none_or(|method| owner.method != method.name)
            }) {
                return Err(error(
                    "declared constructor disagrees with its typed function owner",
                ));
            }
        }
    }
    for function in &assembly.functions {
        if let Some(owner) = &function.class_method_owner {
            if owner.role == ClassMethodRole::Constructor
                && classes
                    .get(&owner.declaring_class)
                    .is_none_or(|declaration| {
                        declaration
                            .constructor
                            .as_ref()
                            .map(|method| method.function)
                            != Some(function.id)
                    })
            {
                return Err(error("constructor function lacks its class declaration"));
            }
        }
        for (caller, parent) in function.super_constructor_calls() {
            verify_call(function, &caller, &parent, &classes)?;
        }
    }
    Ok(())
}

fn verify_call(
    function: &NativeFunction,
    caller: &ClassIdentity,
    parent: &ClassIdentity,
    classes: &BTreeMap<&ClassIdentity, &crate::NativeClassDeclaration>,
) -> NativeCodegenResult<()> {
    // Generic native still supports a host-registered class call without a
    // compiled class product. Standalone admission separately requires a
    // linked declaration before executing such a call.
    if classes.is_empty() && function.class_method_owner.is_none() {
        return Ok(());
    }
    if function.class_method_owner.as_ref().is_none_or(|owner| {
        owner.role != ClassMethodRole::Constructor
            || owner.is_static
            || &owner.declaring_class != caller
    }) || classes
        .get(caller)
        .is_none_or(|declaration| declaration.parent.as_ref() != Some(parent))
    {
        return Err(error(
            "super call disagrees with its compiled constructor and parent",
        ));
    }
    // A compiled but unused constructor can retain an unresolved external
    // parent in generic native. Actual selected-constructor admission closes
    // that dependency before receiver allocation; standalone AOT closes the
    // entire linked class batch before any entrypoint effect.
    if classes
        .get(parent)
        .and_then(|declaration| declaration.constructor.as_ref())
        .is_some_and(|constructor| constructor.access == MemberAccess::Private)
    {
        return Err(error("super call targets a private constructor"));
    }
    Ok(())
}

fn error(message: &str) -> NativeCodegenError {
    NativeCodegenError::new("native.ir.super_constructor", message)
}
