use std::collections::HashMap;

use runmat_types::{
    ClassIdentity, ClassMethodOwner, ClassMethodRole, EventDeclaration, MemberAccess, MemberName,
    MethodName,
};
use std::ops::Deref;
use std::sync::Arc;

use crate::BoundFunctionBinding;

/// Receiver-owned, port-free selection facts. Each method has its own exact
/// binding, since inherited declarations can be admitted in a different V3
/// unit from the constructor. Local function numbers alone select nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedReceiverMethodDeclaration {
    pub owner: ClassMethodOwner,
    pub is_abstract: bool,
    pub access: MemberAccess,
    pub standard_substruct: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedReceiverPropertyDeclaration {
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub get_access: MemberAccess,
    pub set_access: MemberAccess,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedReceiverClassDeclaration {
    pub parent: Option<ClassIdentity>,
    pub properties: HashMap<MemberName, SelectedReceiverPropertyDeclaration>,
    pub events: HashMap<MemberName, EventDeclaration>,
}

/// The canonical selected receiver declaration projection. It owns no
/// default Values, source tables, definition leases, registry, or service
/// ports. The selected class product and every copied receiver share one Arc.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedReceiverDeclarations {
    class: ClassIdentity,
    is_handle_class: bool,
    ancestry: Vec<ClassIdentity>,
    classes: HashMap<ClassIdentity, SelectedReceiverClassDeclaration>,
    methods: HashMap<MethodName, Arc<SelectedReceiverMethodDeclaration>>,
}

impl SelectedReceiverDeclarations {
    pub fn new(
        class: ClassIdentity,
        is_handle_class: bool,
        ancestry: Vec<ClassIdentity>,
        classes: HashMap<ClassIdentity, SelectedReceiverClassDeclaration>,
        methods: HashMap<MethodName, Arc<SelectedReceiverMethodDeclaration>>,
    ) -> Option<Self> {
        let mut cursor = Some(&class);
        for name in &ancestry {
            if cursor != Some(name) {
                return None;
            }
            cursor = classes.get(name)?.parent.as_ref();
        }
        let ends_at_handle = cursor.is_some_and(|name| name.is(runmat_types::standard::HANDLE));
        if ancestry.first() != Some(&class)
            || (cursor.is_some() && !ends_at_handle)
            || ends_at_handle != is_handle_class
            || methods.iter().any(|(name, method)| {
                name != &method.owner.method
                    || method.owner.role != ClassMethodRole::Ordinary
                    || !ancestry.contains(&method.owner.declaring_class)
            })
        {
            return None;
        }
        Some(Self {
            class,
            is_handle_class,
            ancestry,
            classes,
            methods,
        })
    }

    pub fn class(&self) -> &ClassIdentity {
        &self.class
    }
    pub fn is_handle_class(&self) -> bool {
        self.is_handle_class
    }
    pub fn ancestry(&self) -> &[ClassIdentity] {
        &self.ancestry
    }
    pub fn method(&self, name: &MethodName) -> Option<&Arc<SelectedReceiverMethodDeclaration>> {
        self.methods.get(name)
    }
    pub fn methods(&self) -> &HashMap<MethodName, Arc<SelectedReceiverMethodDeclaration>> {
        &self.methods
    }
    pub fn class_declarations(&self) -> &HashMap<ClassIdentity, SelectedReceiverClassDeclaration> {
        &self.classes
    }
}

/// One receiver-specific execution lease paired with the shared canonical
/// method declaration. Deref exposes only immutable declaration fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedObjectMethod {
    pub declaration: Arc<SelectedReceiverMethodDeclaration>,
    /// Exact independently issued method body. Its numeric function is never
    /// interpreted without this binding's counted definition authority.
    pub binding: Option<BoundFunctionBinding>,
}

impl Deref for SelectedObjectMethod {
    type Target = SelectedReceiverMethodDeclaration;
    fn deref(&self) -> &Self::Target {
        &self.declaration
    }
}

/// Shared by value-object copies and by handle-object GC targets. No source
/// bytes, registry, runtime port or managed Value is retained here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedObjectClassFacts {
    declarations: Arc<SelectedReceiverDeclarations>,
    constructor: BoundFunctionBinding,
    methods: HashMap<MethodName, SelectedObjectMethod>,
}

impl SelectedObjectClassFacts {
    pub fn new(
        declarations: Arc<SelectedReceiverDeclarations>,
        constructor: BoundFunctionBinding,
        method_bindings: HashMap<MethodName, Option<BoundFunctionBinding>>,
    ) -> Option<Self> {
        if constructor
            .authority
            .local_lease(constructor.function)
            .is_none_or(|lease| {
                lease
                    .owner_arc()
                    .selected_receiver_declarations()
                    .is_none_or(|admitted| !Arc::ptr_eq(admitted, &declarations))
            })
            || method_bindings.len() != declarations.methods().len()
            || method_bindings.iter().any(|(name, binding)| {
                !declarations.methods().contains_key(name)
                    || binding.as_ref().is_some_and(|binding| {
                        binding.authority.local_lease(binding.function).is_none()
                            && binding.authority.portable_origin().is_none()
                    })
            })
        {
            return None;
        }
        let methods = method_bindings
            .into_iter()
            .map(|(name, binding)| {
                let declaration = Arc::clone(declarations.method(&name).expect("checked method"));
                (
                    name,
                    SelectedObjectMethod {
                        declaration,
                        binding,
                    },
                )
            })
            .collect();
        Some(Self {
            declarations,
            constructor,
            methods,
        })
    }

    pub fn class(&self) -> &ClassIdentity {
        self.declarations.class()
    }

    pub fn is_handle_class(&self) -> bool {
        self.declarations.is_handle_class()
    }

    pub fn constructor(&self) -> &BoundFunctionBinding {
        &self.constructor
    }

    pub fn ancestry(&self) -> &[ClassIdentity] {
        self.declarations.ancestry()
    }

    pub fn declarations(&self) -> &Arc<SelectedReceiverDeclarations> {
        &self.declarations
    }

    pub fn method(&self, name: &MethodName) -> Option<&SelectedObjectMethod> {
        self.methods.get(name)
    }

    pub fn methods(&self) -> impl Iterator<Item = &SelectedObjectMethod> {
        self.methods.values()
    }

    pub fn method_binding(&self, name: &MethodName) -> Option<BoundFunctionBinding> {
        self.methods.get(name)?.binding.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::{
        FunctionDefinitionOwner, LocalExport, ObjectInstance, ObjectPersistencePayloadKind,
        SelectedReceiverDeclarationSeal,
    };

    #[test]
    fn selected_receiver_copies_share_facts_without_reclassifying_plain_objects() {
        let class = ClassIdentity::new("SelectedBase").expect("class");
        let issuer = Arc::new(());
        let definition = Arc::new(FunctionDefinitionOwner::default());
        let constructor = BoundFunctionBinding::local(
            1,
            Arc::downgrade(&issuer),
            FunctionDefinitionOwner::issue(&definition, 1),
            Vec::new(),
            LocalExport::Ephemeral,
        );
        let declarations = Arc::new(
            SelectedReceiverDeclarations::new(
                class.clone(),
                false,
                vec![class.clone()],
                HashMap::from([(
                    class.clone(),
                    SelectedReceiverClassDeclaration {
                        parent: None,
                        properties: HashMap::new(),
                        events: HashMap::new(),
                    },
                )]),
                HashMap::new(),
            )
            .expect("selected declarations"),
        );
        let (seal, key) = SelectedReceiverDeclarationSeal::issue();
        assert!(definition.admit_selected_receiver_seal(Arc::clone(&seal)));
        assert!(seal.fill(&key, Arc::clone(&declarations)));
        let (replacement, _) = SelectedReceiverDeclarationSeal::issue();
        assert!(!definition.admit_selected_receiver_seal(replacement));
        let forged_same_content = Arc::new(declarations.as_ref().clone());
        assert!(SelectedObjectClassFacts::new(
            forged_same_content,
            constructor.clone(),
            HashMap::new(),
        )
        .is_none());
        let facts = Rc::new(
            SelectedObjectClassFacts::new(declarations, constructor, HashMap::new())
                .expect("typed facts"),
        );
        let selected = ObjectInstance::new(class.clone())
            .with_selected_class(Rc::clone(&facts))
            .expect("selected receiver");
        let copied = selected.clone();
        assert!(Rc::ptr_eq(
            selected.selected_class().expect("selected facts"),
            copied.selected_class().expect("copied facts")
        ));
        assert_eq!(
            selected.persistence_payload_kind(),
            ObjectPersistencePayloadKind::SelectedClass
        );
        assert_eq!(
            ObjectInstance::new(class).persistence_payload_kind(),
            ObjectPersistencePayloadKind::PlainValue
        );
        let mut attached = ObjectInstance::new("SelectedBase");
        assert!(attached.try_attach_selected_class(Rc::clone(&facts)));
        assert!(Rc::ptr_eq(
            attached.selected_class().expect("attached facts"),
            &facts
        ));
        assert!(!attached.try_attach_selected_class(Rc::clone(&facts)));
        let mut different = ObjectInstance::new("DifferentClass");
        assert!(!different.try_attach_selected_class(facts));
        assert!(different.selected_class().is_none());
    }
}
