//! One-shot method intents and exact admitted frame ownership. Frame markers
//! stay inline; receiver facts and ordinary method declarations remain shared.
use super::RuntimeContext;
use std::{rc::Rc, sync::Arc};

pub type SelectedMethodBodyOwner = (
    Rc<runmat_value::SelectedObjectClassFacts>,
    runmat_types::ClassMethodOwner,
    SelectedMethodDispatch,
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectedMethodDispatch {
    #[default]
    Virtual,
    Superclass,
}

#[derive(Clone)]
pub(super) struct AdmittedSelectedMethodFrame {
    declaration: AdmittedFrameDeclaration,
    facts: Rc<runmat_value::SelectedObjectClassFacts>,
}

#[derive(Clone)]
enum AdmittedFrameDeclaration {
    Virtual(Arc<runmat_value::SelectedReceiverMethodDeclaration>),
    Superclass(Arc<runmat_value::SelectedReceiverMethodDeclaration>),
    Constructor(Rc<AdmittedConstructorFrame>),
}

// Constructor entry creates this immutable metadata once. Context clones share
// it rather than carrying the complete definition graph through every future.
struct AdmittedConstructorFrame {
    class: ConstructorClassSlot,
    binding: runmat_value::BoundFunctionBinding,
}

#[derive(Clone, Copy)]
struct ConstructorClassSlot(usize);

impl AdmittedSelectedMethodFrame {
    fn declaring_class(&self) -> &runmat_types::ClassIdentity {
        match &self.declaration {
            AdmittedFrameDeclaration::Virtual(method)
            | AdmittedFrameDeclaration::Superclass(method) => &method.owner.declaring_class,
            AdmittedFrameDeclaration::Constructor(frame) => &self.facts.ancestry()[frame.class.0],
        }
    }

    fn matches_owner(&self, owner: &runmat_types::ClassMethodOwner) -> bool {
        match &self.declaration {
            AdmittedFrameDeclaration::Virtual(method)
            | AdmittedFrameDeclaration::Superclass(method) => &method.owner == owner,
            AdmittedFrameDeclaration::Constructor(_) => {
                owner.role == runmat_types::ClassMethodRole::Constructor
                    && !owner.is_static
                    && &owner.declaring_class == self.declaring_class()
            }
        }
    }

    fn binding(&self) -> Option<&runmat_value::BoundFunctionBinding> {
        match &self.declaration {
            AdmittedFrameDeclaration::Virtual(method)
            | AdmittedFrameDeclaration::Superclass(method) => {
                self.facts.method_by_owner(&method.owner)?.binding.as_ref()
            }
            AdmittedFrameDeclaration::Constructor(frame) => Some(&frame.binding),
        }
    }

    fn dispatch(&self) -> SelectedMethodDispatch {
        match &self.declaration {
            AdmittedFrameDeclaration::Virtual(_) | AdmittedFrameDeclaration::Constructor(_) => {
                SelectedMethodDispatch::Virtual
            }
            AdmittedFrameDeclaration::Superclass(_) => SelectedMethodDispatch::Superclass,
        }
    }
}

impl RuntimeContext {
    pub(crate) fn with_selected_method_body(
        mut self,
        facts: Rc<runmat_value::SelectedObjectClassFacts>,
        owner: runmat_types::ClassMethodOwner,
        binding: runmat_value::BoundFunctionBinding,
    ) -> Self {
        self.selected_method_frame = None;
        self.selected_method_body = Some((facts, owner, binding, SelectedMethodDispatch::Virtual));
        self
    }

    pub(crate) fn with_selected_super_method_body(
        mut self,
        facts: Rc<runmat_value::SelectedObjectClassFacts>,
        owner: runmat_types::ClassMethodOwner,
        binding: runmat_value::BoundFunctionBinding,
    ) -> Self {
        self.selected_method_frame = None;
        self.selected_method_body =
            Some((facts, owner, binding, SelectedMethodDispatch::Superclass));
        self
    }

    /// The selected executor calls this after checking the body's exact
    /// definition lease and receiver declaration seal.
    pub fn with_admitted_selected_method_frame(
        mut self,
        owner: runmat_types::ClassMethodOwner,
        binding: runmat_value::BoundFunctionBinding,
        dispatch: SelectedMethodDispatch,
    ) -> Result<Self, crate::RuntimeError> {
        let facts = self.selected_object_class().ok_or_else(|| {
            crate::runtime_error::semantic_error(
                "ObjectMethodFrameIdentityMismatch",
                "superclass frame has no receiver product",
            )
        })?;
        match dispatch {
            SelectedMethodDispatch::Virtual => {
                crate::object::protocol::ObjectAccessContext::from_selected_method_frame(
                    &owner, facts,
                )
            }
            SelectedMethodDispatch::Superclass => {
                crate::object::protocol::ObjectAccessContext::from_selected_super_method_frame(
                    &owner, facts,
                )
            }
        }?;
        if facts
            .method_by_owner(&owner)
            .and_then(|method| method.binding.as_ref())
            != Some(&binding)
        {
            return Err(crate::runtime_error::semantic_error(
                "ObjectMethodFrameIdentityMismatch",
                "superclass frame differs from its receiver's exact method binding",
            ));
        }
        let method = facts
            .method_by_owner(&owner)
            .expect("binding and declaration checked");
        self.selected_method_frame = Some(AdmittedSelectedMethodFrame {
            declaration: match dispatch {
                SelectedMethodDispatch::Virtual => {
                    AdmittedFrameDeclaration::Virtual(Arc::clone(&method.declaration))
                }
                SelectedMethodDispatch::Superclass => {
                    AdmittedFrameDeclaration::Superclass(Arc::clone(&method.declaration))
                }
            },
            facts: Rc::clone(facts),
        });
        Ok(self)
    }

    /// Executor-only metadata projection after validating the entered body's
    /// typed constructor owner, source product and exact definition binding.
    /// A retained parent body keeps the dynamic receiver's existing facts.
    pub fn with_admitted_constructor_frame(
        mut self,
        owner: &runmat_types::ClassMethodOwner,
        binding: runmat_value::BoundFunctionBinding,
    ) -> Result<Self, crate::RuntimeError> {
        let Some(facts) = self.selected_object_class() else {
            return Ok(self);
        };
        if owner.role != runmat_types::ClassMethodRole::Constructor
            || owner.is_static
            || !facts.ancestry().contains(&owner.declaring_class)
            || !binding.authority.admits_local(&self.session_lease())
            || binding.authority.local_lease(binding.function).is_none()
        {
            return Err(crate::runtime_error::semantic_error(
                "ConstructorOwnerInvalid",
                "constructor frame differs from its admitted receiver or definition",
            ));
        }
        let class = ConstructorClassSlot(
            facts
                .ancestry()
                .iter()
                .position(|class| class == &owner.declaring_class)
                .expect("constructor class admitted above"),
        );
        self.selected_method_frame = Some(AdmittedSelectedMethodFrame {
            declaration: AdmittedFrameDeclaration::Constructor(Rc::new(AdmittedConstructorFrame {
                class,
                binding,
            })),
            facts: Rc::clone(facts),
        });
        Ok(self)
    }

    pub(crate) fn admits_selected_method_caller(
        &self,
        caller: &runmat_types::ClassIdentity,
        facts: &Rc<runmat_value::SelectedObjectClassFacts>,
    ) -> bool {
        self.current_selected_method_frame().is_some_and(|frame| {
            frame.declaring_class() == caller && Rc::ptr_eq(&frame.facts, facts)
        })
    }

    pub fn selected_method_frame_access(
        &self,
        owner: &runmat_types::ClassMethodOwner,
        function: runmat_types::FunctionId,
    ) -> Result<Option<crate::object::protocol::ObjectAccessContext>, crate::RuntimeError> {
        let Some(frame) = self.current_selected_method_frame().filter(|frame| {
            frame.matches_owner(owner)
                && frame
                    .binding()
                    .is_some_and(|binding| binding.function == function.0)
        }) else {
            return Ok(None);
        };
        let facts = self
            .selected_object_class()
            .filter(|facts| Rc::ptr_eq(facts, &frame.facts))
            .ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "ObjectMethodFrameIdentityMismatch",
                    "superclass frame changed its receiver product",
                )
            })?;
        match frame.dispatch() {
            SelectedMethodDispatch::Virtual => {
                crate::object::protocol::ObjectAccessContext::from_selected_method_frame(
                    owner, facts,
                )
            }
            SelectedMethodDispatch::Superclass => {
                crate::object::protocol::ObjectAccessContext::from_selected_super_method_frame(
                    owner, facts,
                )
            }
        }
        .map(Some)
    }

    fn current_selected_method_frame(&self) -> Option<&AdmittedSelectedMethodFrame> {
        let frame = self.selected_method_frame.as_ref()?;
        let facts = self.selected_object_class()?;
        (Rc::ptr_eq(facts, &frame.facts)
            && match &frame.declaration {
                AdmittedFrameDeclaration::Virtual(declaration)
                | AdmittedFrameDeclaration::Superclass(declaration) => facts
                    .method_by_owner(&declaration.owner)
                    .is_some_and(|method| Arc::ptr_eq(&method.declaration, declaration)),
                AdmittedFrameDeclaration::Constructor(frame) => {
                    facts.ancestry().get(frame.class.0).is_some()
                        && frame.binding.authority.admits_local(&self.session_lease())
                        && frame
                            .binding
                            .authority
                            .local_lease(frame.binding.function)
                            .is_some()
                }
            })
        .then_some(frame)
    }

    pub fn has_admitted_selected_method_frame_for(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> bool {
        self.current_selected_method_frame()
            .is_some_and(|frame| frame.binding() == Some(binding))
    }

    pub fn has_retained_super_body_for(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> bool {
        self.retained_super_body
            .as_ref()
            .is_some_and(|(_, expected)| expected == binding)
    }

    /// One selected receiver's call intent. The exact executable body still
    /// has to prove this typed owner at the bound-invoker admission boundary.
    pub fn take_selected_method_body_for(
        mut self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Result<(Self, Option<SelectedMethodBodyOwner>), crate::RuntimeError> {
        match self.selected_method_body.take() {
            None => Ok((self, None)),
            Some((facts, owner, expected, dispatch)) if &expected == binding => {
                Ok((self, Some((facts, owner, dispatch))))
            }
            Some(_) => Err(crate::build_runtime_error(
                "selected object method dispatch selected a different definition",
            )
            .with_identifier("RunMat:MethodOwnerInvalid")
            .build()),
        }
    }
}
