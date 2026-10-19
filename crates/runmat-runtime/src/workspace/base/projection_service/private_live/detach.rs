use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::workspace::{
    DetachedWorkspaceValues, GlobalBindingTarget, WorkspaceDetachSelection, WorkspaceName,
};
use crate::RuntimeError;

use super::super::super::{base_workspace_error_to_runtime, storage::BindingTreeValue};
use super::super::BaseWorkspaceProjectionService;
use super::private_storage_error;

impl BaseWorkspaceProjectionService {
    pub(in crate::workspace::base) fn private_detach_unbounded(
        &self,
        context: Option<&RuntimeContext>,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, RuntimeError> {
        match context {
            Some(context) => {
                self.ensure_mutation_context(context)?;
                if HostAllocationReceipt::policy(Some(context)).map_err(private_storage_error)?
                    != HostAllocationPolicy::Unbounded
                {
                    return Err(crate::runtime_error::semantic_error(
                        "RunMat:Workspace:ExactDetachAdmissionUnavailable",
                        "finite private workspace detachment requires admitted output and root storage",
                    ));
                }
            }
            None => self.ensure_unbounded_compatibility()?,
        }
        let (owner, owner_storage, projection_storage) = self.private_snapshots()?;
        let session = context.map(RuntimeContext::session_variable_authority);
        let mut namespace = Vec::with_capacity(projection_storage.len());
        let mut values = Vec::new();
        projection_storage.try_visit_sorted(|name, binding| {
            namespace.push(name.to_owned());
            match binding {
                BindingTreeValue::Local(entry) => {
                    values.push((name.to_owned(), entry.value().clone()))
                }
                BindingTreeValue::Global(target) => {
                    let value = match (session.as_ref(), target) {
                        (Some(session), GlobalBindingTarget::Named(target)) => {
                            session.global_value(target.as_str())?
                        }
                        (Some(session), GlobalBindingTarget::LegacySlot(slot)) => {
                            session.global_slot(*slot)?
                        }
                        (None, GlobalBindingTarget::Named(target)) => {
                            crate::workspace::session::global_value(target.as_str())
                        }
                        (None, GlobalBindingTarget::LegacySlot(slot)) => {
                            crate::workspace::session::global_slot_value(*slot)
                        }
                    };
                    if let Some(value) = value {
                        values.push((name.to_owned(), value));
                    }
                }
            }
            Ok::<_, RuntimeError>(())
        })?;
        let detached = DetachedWorkspaceValues::new(selection.select(values))
            .map_err(base_workspace_error_to_runtime)?;
        let selected_names = select_private_names(selection, namespace);
        let projection_prepared = if matches!(selection, WorkspaceDetachSelection::All) {
            projection_storage
                .prepare_clear(context)
                .map_err(private_storage_error)?
        } else {
            projection_storage
                .prepare_batch_with(context, selected_names.len(), |index| {
                    Ok((WorkspaceName::unbounded(&selected_names[index]), None))
                })
                .map_err(private_storage_error)?
        };
        let owner_prepared = owner_storage
            .prepare_adopt_pruned(context, &projection_prepared)
            .map_err(private_storage_error)?;
        self.install_private_transition(context, None, owner, owner_prepared, projection_prepared)?;
        Ok(detached)
    }
}

fn select_private_names(selection: &WorkspaceDetachSelection, names: Vec<String>) -> Vec<String> {
    match selection {
        WorkspaceDetachSelection::All => names,
        WorkspaceDetachSelection::Names(selected) => {
            let available: std::collections::HashSet<&str> =
                names.iter().map(String::as_str).collect();
            selected
                .iter()
                .filter(|name| available.contains(name.as_str()))
                .cloned()
                .collect()
        }
        WorkspaceDetachSelection::Except(excluded) => {
            let excluded: std::collections::HashSet<&str> =
                excluded.iter().map(String::as_str).collect();
            names
                .into_iter()
                .filter(|name| !excluded.contains(name.as_str()))
                .collect()
        }
    }
}
