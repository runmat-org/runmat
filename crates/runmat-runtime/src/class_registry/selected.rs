//! Immutable class ancestry for one exact selected constructor invocation.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use runmat_types::{
    ClassIdentity, ClassMethodOwner, ClassMethodRole, EventDeclaration, MemberAccess, MemberName,
    MethodName,
};

use super::{BoundRuntimeMethod, ClassRegistryRevision, RuntimeClass, RuntimeProperty};

/// Resolve one receiver's already-issued class product, never the mutable
/// same-named registry entry. The source-facts port proves the constructor's
/// counted body/source lease before this projection is used for metadata.
pub fn checked_receiver_projection(
    object: &runmat_value::ObjectInstance,
) -> Result<Option<SelectedReceiverView>, crate::RuntimeError> {
    let Some(facts) = object.selected_class() else {
        return Ok(None);
    };
    if object.has_native_payload() || object.class_name() != facts.class() {
        return Err(crate::build_runtime_error(
            "selected receiver has conflicting object authority",
        )
        .with_identifier("RunMat:SelectedObjectInvalid")
        .build());
    }
    Ok(Some(checked_receiver_facts(facts)?))
}

pub(crate) fn checked_receiver_facts(
    facts: &Rc<runmat_value::SelectedObjectClassFacts>,
) -> Result<SelectedReceiverView, crate::RuntimeError> {
    let runtime = crate::context::legacy::active().ok_or_else(|| {
        crate::build_runtime_error("selected receiver has no issuing session")
            .with_identifier("RunMat:SelectedObjectInvalid")
            .build()
    })?;
    if !facts
        .constructor()
        .authority
        .admits_local(&runtime.session_lease())
        || facts
            .constructor()
            .authority
            .local_lease(facts.constructor().function)
            .is_none()
    {
        return Err(crate::build_runtime_error(
            "selected receiver differs from its issuing definition",
        )
        .with_identifier("RunMat:SelectedObjectInvalid")
        .build());
    }
    let admitted = facts
        .constructor()
        .authority
        .local_lease(facts.constructor().function)
        .and_then(|lease| lease.owner_arc().selected_receiver_declarations());
    if admitted.is_none_or(|declarations| !Arc::ptr_eq(declarations, facts.declarations())) {
        return Err(crate::build_runtime_error(
            "selected receiver metadata differs from its constructor definition",
        )
        .with_identifier("RunMat:SelectedObjectInvalid")
        .build());
    }
    Ok(SelectedReceiverView {
        declarations: Arc::clone(facts.declarations()),
    })
}

/// Borrowed-payload receiver metadata issued once by the exact constructor
/// product. Public reads need no mutable current registry. Non-public reads
/// additionally require the active method's identical selected declaration.
#[derive(Clone, Debug)]
pub struct SelectedReceiverView {
    declarations: Arc<runmat_value::SelectedReceiverDeclarations>,
}

impl SelectedReceiverView {
    pub fn active_product_matches(&self) -> bool {
        crate::context::legacy::active().is_some_and(|runtime| {
            runtime
                .selected_class_projection()
                .and_then(|active| active.receiver_declarations(self.declarations.class()).ok())
                .is_some_and(|admitted| Arc::ptr_eq(&admitted, &self.declarations))
        })
    }

    pub fn class_is_or_subclass(&self, class: &ClassIdentity, ancestor: &ClassIdentity) -> bool {
        let mut cursor = Some(class);
        while let Some(name) = cursor {
            if name == ancestor {
                return true;
            }
            cursor = self
                .declarations
                .class_declarations()
                .get(name)
                .and_then(|selected| selected.parent.as_ref());
        }
        false
    }

    pub fn receiver_property(
        &self,
        class: &ClassIdentity,
        name: &MemberName,
    ) -> Option<(
        runmat_value::SelectedReceiverPropertyDeclaration,
        ClassIdentity,
    )> {
        let mut cursor = Some(class);
        while let Some(owner) = cursor {
            if owner.is(runmat_types::standard::HANDLE) {
                return None;
            }
            let selected = self.declarations.class_declarations().get(owner)?;
            if let Some(property) = selected.properties.get(name) {
                return Some((property.clone(), owner.clone()));
            }
            cursor = selected.parent.as_ref();
        }
        None
    }

    pub fn receiver_event(
        &self,
        class: &ClassIdentity,
        name: &MemberName,
    ) -> Option<(EventDeclaration, ClassIdentity)> {
        let mut cursor = Some(class);
        while let Some(owner) = cursor {
            if owner.is(runmat_types::standard::HANDLE) {
                return None;
            }
            let selected = self.declarations.class_declarations().get(owner)?;
            if let Some(event) = selected.events.get(name) {
                return Some((event.clone(), owner.clone()));
            }
            cursor = selected.parent.as_ref();
        }
        None
    }

    pub fn receiver_declares_method(&self, class: &ClassIdentity, name: &MethodName) -> bool {
        self.declarations.class() == class && self.declarations.method(name).is_some()
    }

    pub fn receiver_member_accessible(
        &self,
        owner: &ClassIdentity,
        access: MemberAccess,
        caller: Option<&ClassIdentity>,
    ) -> bool {
        if access == MemberAccess::Public {
            return true;
        }
        if !self.active_product_matches() {
            return false;
        }
        match access {
            MemberAccess::Public => true,
            MemberAccess::Private => caller == Some(owner),
            MemberAccess::Protected => {
                caller.is_some_and(|class| self.class_is_or_subclass(class, owner))
            }
        }
    }

    pub fn declarations(&self) -> &Arc<runmat_value::SelectedReceiverDeclarations> {
        &self.declarations
    }
}

#[derive(Debug)]
pub struct SelectedClassProjection {
    pub(super) classes: HashMap<ClassIdentity, RuntimeClass>,
    pub(super) method_bindings: HashMap<(ClassIdentity, MethodName), BoundRuntimeMethod>,
    pub(super) sealed: HashSet<ClassIdentity>,
    pub(super) abstract_classes: HashSet<ClassIdentity>,
    pub(super) enumerations: HashMap<ClassIdentity, HashSet<String>>,
    pub(super) generation: ClassRegistryRevision,
    /// One immutable declarative receiver view per constructor class. This
    /// contains no defaults, managed Values, source owners, or body leases.
    receiver_declarations:
        RefCell<HashMap<ClassIdentity, Arc<runmat_value::SelectedReceiverDeclarations>>>,
    // Defaults may contain managed values. The projection can outlive the
    // registry entry it shadows while a constructor is suspended.
    _roots: crate::execution::RootedValueLease,
}

#[derive(Clone, Debug)]
pub struct SelectedReceiverMethodDeclaration {
    pub owner: ClassMethodOwner,
    pub is_abstract: bool,
    pub access: MemberAccess,
}

impl SelectedClassProjection {
    /// Derive the canonical receiver declarations once from this already
    /// checked product. Every issued object shares the same Arc, while this
    /// projection alone retains rooted default/static Values.
    pub fn receiver_declarations(
        &self,
        class: &ClassIdentity,
    ) -> Result<Arc<runmat_value::SelectedReceiverDeclarations>, crate::RuntimeError> {
        if let Some(existing) = self.receiver_declarations.borrow().get(class) {
            return Ok(Arc::clone(existing));
        }
        let (ancestry, _) = self.receiver_method_declarations(class)?;
        let is_handle = self.receiver_is_handle_class(class)?;
        let mut classes = HashMap::with_capacity(ancestry.len());
        let mut methods = HashMap::new();
        for name in &ancestry {
            let selected = self.classes.get(name).ok_or_else(|| {
                crate::build_runtime_error("selected receiver declaration is incomplete")
                    .with_identifier("RunMat:ClassProductInvalid")
                    .build()
            })?;
            for method in selected
                .methods
                .values()
                .filter(|method| method.role == ClassMethodRole::Ordinary)
            {
                let owner = ClassMethodOwner {
                    declaring_class: name.clone(),
                    method: method.name.clone(),
                    is_static: method.is_static,
                    role: method.role,
                };
                let standard_substruct =
                    self.receiver_method_convention(&owner)
                        .is_some_and(|convention| {
                            matches!(
                                convention,
                                super::RuntimeMethodCallingConvention::StandardSubstruct
                            )
                        });
                methods.insert(
                    owner.clone(),
                    Arc::new(runmat_value::SelectedReceiverMethodDeclaration {
                        owner,
                        is_abstract: method.is_abstract,
                        access: method.access,
                        standard_substruct,
                    }),
                );
            }
            classes.insert(
                name.clone(),
                runmat_value::SelectedReceiverClassDeclaration {
                    parent: selected.parent.clone(),
                    properties: selected
                        .properties
                        .iter()
                        .map(|(name, property)| {
                            (
                                name.clone(),
                                runmat_value::SelectedReceiverPropertyDeclaration {
                                    is_static: property.is_static,
                                    is_constant: property.is_constant,
                                    is_dependent: property.is_dependent,
                                    get_access: property.get_access,
                                    set_access: property.set_access,
                                },
                            )
                        })
                        .collect(),
                    events: selected.events.clone(),
                },
            );
        }
        let declarations = runmat_value::SelectedReceiverDeclarations::new(
            class.clone(),
            is_handle,
            ancestry,
            classes,
            methods,
        )
        .ok_or_else(|| {
            crate::build_runtime_error("selected receiver declarations differ from class product")
                .with_identifier("RunMat:ClassProductInvalid")
                .build()
        })?;
        let declarations = Arc::new(declarations);
        self.receiver_declarations
            .borrow_mut()
            .insert(class.clone(), Arc::clone(&declarations));
        Ok(declarations)
    }
    pub fn receiver_class_matches(
        &self,
        class: &ClassIdentity,
        ancestry: &[ClassIdentity],
        is_handle_class: bool,
    ) -> bool {
        let mut cursor = Some(class);
        let mut position = 0usize;
        let mut handle = false;
        while let Some(name) = cursor {
            if name.is(runmat_types::standard::HANDLE) {
                handle = true;
                break;
            }
            if ancestry.get(position) != Some(name) {
                return false;
            }
            position += 1;
            cursor = match self.classes.get(name) {
                Some(selected) => selected.parent.as_ref(),
                None => return false,
            };
        }
        position == ancestry.len() && handle == is_handle_class
    }

    pub fn receiver_event(
        &self,
        class: &ClassIdentity,
        name: &MemberName,
    ) -> Option<(EventDeclaration, ClassIdentity)> {
        let mut cursor = Some(class);
        while let Some(owner) = cursor {
            if owner.is(runmat_types::standard::HANDLE) {
                return None;
            }
            let selected = self.classes.get(owner)?;
            if let Some(event) = selected.events.get(name) {
                return Some((event.clone(), owner.clone()));
            }
            cursor = selected.parent.as_ref();
        }
        None
    }

    pub fn receiver_property(
        &self,
        class: &ClassIdentity,
        name: &MemberName,
    ) -> Option<(RuntimeProperty, ClassIdentity)> {
        let mut cursor = Some(class);
        while let Some(owner) = cursor {
            if owner.is(runmat_types::standard::HANDLE) {
                return None;
            }
            let selected = self.classes.get(owner)?;
            if let Some(property) = selected.properties.get(name) {
                return Some((property.clone(), owner.clone()));
            }
            cursor = selected.parent.as_ref();
        }
        None
    }

    pub fn receiver_declares_method(&self, class: &ClassIdentity, name: &MethodName) -> bool {
        let mut cursor = Some(class);
        while let Some(owner) = cursor {
            if owner.is(runmat_types::standard::HANDLE) {
                return false;
            }
            let Some(selected) = self.classes.get(owner) else {
                return false;
            };
            if selected
                .methods
                .get(name)
                .is_some_and(|method| method.role == ClassMethodRole::Ordinary)
            {
                return true;
            }
            cursor = selected.parent.as_ref();
        }
        false
    }

    pub fn class_is_or_subclass(&self, class: &ClassIdentity, ancestor: &ClassIdentity) -> bool {
        let mut cursor = Some(class);
        while let Some(name) = cursor {
            if name == ancestor {
                return true;
            }
            cursor = match self.classes.get(name) {
                Some(selected) => selected.parent.as_ref(),
                None => return false,
            };
        }
        false
    }

    pub fn receiver_member_accessible(
        self: &std::rc::Rc<Self>,
        owner: &ClassIdentity,
        access: MemberAccess,
        caller: Option<&ClassIdentity>,
    ) -> bool {
        if access == MemberAccess::Public {
            return true;
        }
        if crate::context::legacy::active().is_none_or(|runtime| {
            runtime
                .selected_class_projection()
                .is_none_or(|active| !std::rc::Rc::ptr_eq(active, self))
        }) {
            return false;
        }
        match access {
            MemberAccess::Public => true,
            MemberAccess::Private => caller == Some(owner),
            MemberAccess::Protected => {
                caller.is_some_and(|class| self.class_is_or_subclass(class, owner))
            }
        }
    }

    /// Hot-path validation of one receiver method. The projection itself is
    /// cached by its selected product; this walk reads only the exact typed
    /// ancestry and one method entry per class, with no cloned declaration
    /// map or default values on each invocation.
    pub fn receiver_method_matches(
        &self,
        class: &ClassIdentity,
        ancestry: &[ClassIdentity],
        is_handle_class: bool,
        owner: &ClassMethodOwner,
        access: MemberAccess,
        is_abstract: bool,
        standard_substruct: bool,
    ) -> bool {
        let mut cursor = Some(class);
        let mut position = 0usize;
        let mut handle = false;
        let mut effective = None;
        while let Some(name) = cursor {
            if name.is(runmat_types::standard::HANDLE) {
                handle = true;
                break;
            }
            if ancestry.get(position) != Some(name) {
                return false;
            }
            position += 1;
            let Some(selected) = self.classes.get(name) else {
                return false;
            };
            if effective.is_none() {
                effective = selected
                    .methods
                    .get(&owner.method)
                    .filter(|method| method.role == ClassMethodRole::Ordinary)
                    .map(|method| (name, method.is_static, method.is_abstract, method.access));
            }
            cursor = selected.parent.as_ref();
        }
        let expected_convention =
            self.receiver_method_convention(owner)
                .is_some_and(|convention| {
                    matches!(
                        convention,
                        super::RuntimeMethodCallingConvention::StandardSubstruct
                    )
                });
        position == ancestry.len()
            && handle == is_handle_class
            && effective.is_some_and(
                |(declaring_class, is_static, abstract_method, method_access)| {
                    owner.role == ClassMethodRole::Ordinary
                        && declaring_class == &owner.declaring_class
                        && is_static == owner.is_static
                        && abstract_method == is_abstract
                        && method_access == access
                        && expected_convention == standard_substruct
                },
            )
    }

    pub fn receiver_is_handle_class(
        &self,
        class: &ClassIdentity,
    ) -> Result<bool, crate::RuntimeError> {
        let mut visited = HashSet::new();
        let mut cursor = Some(class.clone());
        while let Some(name) = cursor {
            if name.is(runmat_types::standard::HANDLE) {
                return Ok(true);
            }
            if !visited.insert(name.clone()) {
                return Err(
                    crate::build_runtime_error("selected class ancestry is cyclic")
                        .with_identifier("RunMat:ClassProductInvalid")
                        .build(),
                );
            }
            let selected = self.classes.get(&name).ok_or_else(|| {
                crate::build_runtime_error("selected class ancestry is incomplete")
                    .with_identifier("RunMat:ClassProductInvalid")
                    .build()
            })?;
            cursor = selected.parent.clone();
        }
        Ok(false)
    }

    pub fn receiver_method_convention(
        &self,
        owner: &ClassMethodOwner,
    ) -> Option<super::RuntimeMethodCallingConvention> {
        self.method_bindings
            .get(&(owner.declaring_class.clone(), owner.method.clone()))
            .filter(|bound| bound.declaring_class == owner.declaring_class)
            .map(|bound| bound.convention)
    }

    /// Effective method declarations for one selected receiver, child before
    /// parent. Only this product's admitted ancestry contributes; a later
    /// same-named ambient class cannot add or replace a method here.
    pub fn receiver_method_declarations(
        &self,
        class: &ClassIdentity,
    ) -> Result<
        (
            Vec<ClassIdentity>,
            HashMap<MethodName, SelectedReceiverMethodDeclaration>,
        ),
        crate::RuntimeError,
    > {
        let mut ancestry = Vec::new();
        let mut methods = HashMap::new();
        let mut visited = HashSet::new();
        let mut cursor = Some(class.clone());
        while let Some(name) = cursor {
            if name.is(runmat_types::standard::HANDLE) {
                break;
            }
            if !visited.insert(name.clone()) {
                return Err(
                    crate::build_runtime_error("selected class ancestry is cyclic")
                        .with_identifier("RunMat:ClassProductInvalid")
                        .build(),
                );
            }
            let selected = self.classes.get(&name).ok_or_else(|| {
                crate::build_runtime_error("selected class ancestry is incomplete")
                    .with_identifier("RunMat:ClassProductInvalid")
                    .build()
            })?;
            ancestry.push(name.clone());
            for method in selected
                .methods
                .values()
                .filter(|method| method.role == ClassMethodRole::Ordinary)
            {
                methods.entry(method.name.clone()).or_insert_with(|| {
                    SelectedReceiverMethodDeclaration {
                        owner: ClassMethodOwner {
                            declaring_class: name.clone(),
                            method: method.name.clone(),
                            is_static: method.is_static,
                            role: ClassMethodRole::Ordinary,
                        },
                        is_abstract: method.is_abstract,
                        access: method.access,
                    }
                });
            }
            cursor = selected.parent.clone();
        }
        Ok((ancestry, methods))
    }

    pub(super) fn new(
        classes: HashMap<ClassIdentity, RuntimeClass>,
        method_bindings: HashMap<(ClassIdentity, MethodName), BoundRuntimeMethod>,
        sealed: HashSet<ClassIdentity>,
        abstract_classes: HashSet<ClassIdentity>,
        enumerations: HashMap<ClassIdentity, HashSet<String>>,
        generation: ClassRegistryRevision,
    ) -> Result<Self, crate::RuntimeError> {
        let roots =
            crate::execution::RootedValueLease::for_values(classes.values().flat_map(|class| {
                class
                    .properties
                    .values()
                    .filter_map(|property| property.default_value.as_ref())
            }))
            .map_err(|error| error.into_runtime_error())?;
        Ok(Self {
            classes,
            method_bindings,
            sealed,
            abstract_classes,
            enumerations,
            generation,
            receiver_declarations: RefCell::new(HashMap::new()),
            _roots: roots,
        })
    }
}

#[cfg(test)]
#[path = "selected/tests.rs"]
mod tests;
