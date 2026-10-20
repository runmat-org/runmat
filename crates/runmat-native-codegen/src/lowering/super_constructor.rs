//! One projection of canonical MIR class declarations into Native IR.

use runmat_types::{ClassIdentity, ClassMethodRole};

use crate::{
    NativeClassDeclaration, NativeClassMethod, NativeClassProperty, NativeCodegenError,
    NativeCodegenResult, NativeConstructorMethod,
};

pub(super) fn lower(
    mir: &runmat_mir::MirAssembly,
) -> NativeCodegenResult<Vec<NativeClassDeclaration>> {
    let mut classes = Vec::with_capacity(mir.classes.len());
    for class in &mir.classes {
        let identity = ClassIdentity::from_qualified_name(&class.name).map_err(|error| {
            NativeCodegenError::new("native.lowering.class_identity", error.to_string())
        })?;
        let mut parent = if let Some(parent) = class.inheritance.resolved_super_class {
            let declaration = mir
                .classes
                .iter()
                .find(|candidate| candidate.id == parent)
                .ok_or_else(|| {
                    NativeCodegenError::new(
                        "native.lowering.class_parent",
                        "resolved superclass is absent from canonical MIR",
                    )
                })?;
            Some(
                ClassIdentity::from_qualified_name(&declaration.name).map_err(|error| {
                    NativeCodegenError::new("native.lowering.class_parent", error.to_string())
                })?,
            )
        } else {
            class
                .inheritance
                .builtin_super_class
                .as_ref()
                .or(class.inheritance.declared_super_class.as_ref())
                .map(|name| {
                    ClassIdentity::new(name.clone()).map_err(|error| {
                        NativeCodegenError::new("native.lowering.class_parent", error.to_string())
                    })
                })
                .transpose()?
        };
        if parent.is_none() && matches!(class.kind, runmat_types::ClassKind::Handle) {
            parent = Some(runmat_types::standard::HANDLE.owned());
        }
        let properties = class
            .properties
            .iter()
            .map(|property| {
                let default_literal = mir
                    .class_property_defaults
                    .iter()
                    .find(|candidate| {
                        candidate.class == class.id && candidate.property == property.name
                    })
                    .map(|candidate| candidate.literal.clone());
                NativeClassProperty {
                    name: property.name.clone(),
                    attributes: property.attributes.clone(),
                    has_default: property.has_default,
                    default_literal,
                }
            })
            .collect();
        let methods = class
            .methods
            .iter()
            .map(|method| {
                Ok(NativeClassMethod {
                    name: method.name.clone(),
                    function: method.function,
                    is_static: method.is_static,
                    attributes: method.attributes.clone(),
                })
            })
            .collect::<NativeCodegenResult<Vec<_>>>()?;
        let mut constructor = None;
        for method in &class.methods {
            let Some(metadata) = mir.functions.get(&method.function) else {
                continue;
            };
            let Some(owner) = &metadata.class_method_owner else {
                continue;
            };
            if owner.role != ClassMethodRole::Constructor {
                continue;
            }
            if owner.declaring_class != identity
                || owner.method != method.name
                || owner.is_static
                || method.is_static
                || method.attributes.is_abstract
            {
                return Err(NativeCodegenError::new(
                    "native.lowering.class_constructor",
                    "constructor metadata disagrees with its class declaration",
                ));
            }
            if constructor
                .replace(NativeConstructorMethod {
                    function: method.function,
                    access: method.attributes.access,
                })
                .is_some()
            {
                return Err(NativeCodegenError::new(
                    "native.lowering.class_constructor",
                    "class has multiple constructor declarations",
                ));
            }
        }
        classes.push(NativeClassDeclaration {
            class: identity,
            parent,
            is_sealed: class.is_sealed,
            is_abstract: class.is_abstract,
            properties,
            methods,
            events: class.events.clone(),
            enumerations: class
                .enumerations
                .iter()
                .map(|enumeration| enumeration.name.clone())
                .collect(),
            constructor,
        });
    }
    classes.sort_by(|left, right| left.class.cmp(&right.class));
    if classes
        .windows(2)
        .any(|pair| pair[0].class == pair[1].class)
    {
        return Err(NativeCodegenError::new(
            "native.lowering.class_identity",
            "canonical MIR repeats a class identity",
        ));
    }
    Ok(classes)
}
