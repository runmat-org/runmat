//! Session-owned registration from the single canonical class declaration.
use super::*;
static CLASS_REGISTRATION: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CLASS);
pub(super) fn ensure() {
    CLASS_REGISTRATION.ensure(|| {
        let declaration = runmat_builtins::standard_class_declaration(&CLASS.owned())
            .expect("canonical optimization variable class");
        let properties = declaration
            .properties
            .into_iter()
            .map(|property| {
                let name = property.name;
                (
                    name.clone(),
                    crate::class_registry::RuntimeProperty {
                        name,
                        is_static: false,
                        is_constant: false,
                        is_dependent: true,
                        get_access: property.attributes.get_access,
                        set_access: property.attributes.set_access,
                        default_value: None,
                    },
                )
            })
            .collect();
        crate::class_registry::register_builtin_class(
            CLASS,
            crate::class_registry::RuntimeClass {
                name: CLASS.owned(),
                parent: None,
                properties,
                methods: crate::class_registry::standard_runtime_methods(CLASS),
                events: Default::default(),
            },
        );
    });
}

pub(super) enum Fence {
    Ordinary,
    Managed {
        session: crate::context::RuntimeContextLocalId,
        revision: u64,
    },
}
impl Fence {
    pub(super) fn prepare(
        cooperation: &super::super::common::ScalarCooperation,
    ) -> BuiltinResult<Self> {
        use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
        cooperation.check_allocation_epoch()?;
        match HostAllocationReceipt::policy(cooperation.runtime()).map_err(|source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                cooperation.runtime(),
                "RunMat:optimvar:Allocation",
                "variable registration fence",
                source,
            )
        })? {
            HostAllocationPolicy::Unbounded => Ok(Self::Ordinary),
            HostAllocationPolicy::Managed => {
                let runtime = cooperation
                    .runtime()
                    .expect("managed policy has its runtime");
                let revision = CLASS_REGISTRATION.current_builtin_revision_in_context(runtime);
                match revision {
                    Some(revision) => Ok(Self::Managed { session: runtime.local_identity(), revision }),
                    None => Err(crate::context::host_allocation::host_allocation_runtime_error(
                        Some(runtime), "RunMat:Placement:InsufficientResourceEvidence",
                        "optimvar: cold canonical class registration metadata requires unbounded workspace",
                        crate::context::host_allocation::HostAllocationError::Resource(
                            crate::context::host_effect::HostEffectAllocationError::Unavailable,
                        ),
                    )),
                }
            }
        }
    }
    pub(super) fn check(
        &self,
        cooperation: &super::super::common::ScalarCooperation,
    ) -> BuiltinResult<()> {
        cooperation.check_allocation_epoch()?;
        if let Self::Managed { session, revision } = self {
            let current = cooperation
                .runtime()
                .filter(|runtime| runtime.local_identity() == *session)
                .and_then(|runtime| {
                    CLASS_REGISTRATION.current_builtin_revision_in_context(runtime)
                });
            if current != Some(*revision) {
                return Err(
                    crate::context::host_allocation::host_allocation_runtime_error(
                        cooperation.runtime(),
                        "RunMat:optimvar:Allocation",
                        "original variable registration",
                        crate::context::host_allocation::HostAllocationError::StaleAuthority,
                    ),
                );
            }
        }
        Ok(())
    }
    pub(super) fn ensure_current(
        &self,
        cooperation: &super::super::common::ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.check(cooperation)?;
        if matches!(self, Self::Ordinary) {
            match cooperation.runtime() {
                Some(runtime) => {
                    let _scope = runtime.enter();
                    ensure();
                }
                None => {
                    let selected = crate::context::PreparedUnselectedScope::prepare(None)?;
                    let _scope = selected.enter();
                    ensure();
                }
            }
        }
        self.check(cooperation)
    }
}
