//! Operation-admitted reconciliation of compiler binding slots at ingress.
//!
//! The canonical base owner lends names, never values. A replacement table is
//! fully constructed while those names are borrowed and becomes visible only
//! after its claims, slot mapping, and namespace checks have succeeded.

use super::super::{RunMatSession, SessionWorkspaceBinding};
use super::binding_control::{BindingLease, BindingPhase};
use super::binding_key::RetainedBindingKey;
use super::binding_scratch::ClaimedVec;
use super::binding_table::BindingTable;
use super::global_key::session_global_key;
use hashbrown::HashSet;
use runmat_runtime::context::host_allocation::{
    frame_hash_table_retained_bytes, FrameCollectionGate, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt,
};
use runmat_runtime::workspace::{BaseWorkspaceBindingNames, GlobalBindingTarget, WorkspaceName};
use runmat_runtime::{context::RuntimeContext, RuntimeError};
use std::alloc::Layout;
use std::collections::hash_map::RandomState;

type BorrowedBinding<'a> = (&'a str, Option<&'a GlobalBindingTarget>);

struct IngressQuote {
    revision: runmat_runtime::workspace::WorkspaceRevision,
    raw_names: usize,
    names: usize,
    largest_live: Option<usize>,
    retained: u64,
}

fn ingress_error(message: &str) -> RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:WorkspaceBindingSlots")
        .build()
}

fn allocation_error(context: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    runmat_runtime::build_runtime_error(format!(
        "workspace binding ingress admission failed: {cause}"
    ))
    .with_identifier(format!(
        "{}:{}",
        context.error_namespace(),
        cause.runtime_identifier_suffix()
    ))
    .with_source(cause)
    .build()
}

fn exact_retained(context: &RuntimeContext, bytes: usize) -> Result<u64, RuntimeError> {
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes)
            .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?,
        1,
    )
    .map_err(|error| allocation_error(context, error))
}

fn add_retained(total: &mut u64, added: u64, context: &RuntimeContext) -> Result<(), RuntimeError> {
    *total = total
        .checked_add(added)
        .ok_or_else(|| allocation_error(context, HostAllocationError::Overflow))?;
    Ok(())
}

fn eligible_local_key(binding: &SessionWorkspaceBinding) -> bool {
    !binding.key.is_global()
}

fn borrowed_name_count(
    context: &RuntimeContext,
    view: &BaseWorkspaceBindingNames<'_>,
) -> Result<usize, RuntimeError> {
    view.local_names()
        .count()
        .checked_add(view.global_names().count())
        .ok_or_else(|| allocation_error(context, HostAllocationError::Overflow))
}

fn collect_borrowed_names<'a>(
    context: &RuntimeContext,
    receipt: Option<&HostAllocationReceipt>,
    view: &'a BaseWorkspaceBindingNames<'_>,
    raw_names: usize,
) -> Result<ClaimedVec<BorrowedBinding<'a>>, RuntimeError> {
    if raw_names == 0 {
        return Ok(ClaimedVec::new(Vec::new(), None));
    }
    let (names, claim) = if let Some(receipt) = receipt {
        receipt
            .with_output_allocation(context, |permit| {
                let bytes = Layout::array::<BorrowedBinding<'_>>(raw_names)
                    .map_err(|_| runmat_value::ExactHostNumericAllocationError::LayoutOverflow)?
                    .size();
                let claim = permit.reserve_exact_claim(bytes)?;
                Ok((runmat_value::exact_capacity_vec(raw_names)?, Some(claim)))
            })
            .map_err(|error| allocation_error(context, error))?
    } else {
        (
            runmat_value::exact_capacity_vec(raw_names)
                .map_err(|error| allocation_error(context, HostAllocationError::Numeric(error)))?,
            None,
        )
    };
    let mut names = ClaimedVec::new(names, claim);
    names.extend(view.local_names().map(|name| (name, None)));
    for name in view.global_names() {
        let target = view
            .global_target(name)
            .ok_or_else(|| ingress_error("global binding has no canonical target"))?;
        names.push((name, Some(target)));
    }
    if names.len() != raw_names {
        return Err(ingress_error(
            "canonical workspace names changed during binding preparation",
        ));
    }
    // The managed and legacy iterators are each unique, but their chain need
    // not be. The final global namespace wins at a mixed boundary.
    names.sort_unstable_by(|left, right| {
        left.0
            .cmp(right.0)
            .then(right.1.is_some().cmp(&left.1.is_some()))
    });
    names.dedup_by(|current, previous| current.0 == previous.0);
    Ok(names)
}

impl RunMatSession {
    pub(super) fn prepare_import_binding_metadata(
        &self,
        entries: &[(String, runmat_value::Value)],
    ) -> Result<(BindingTable, BindingLease), RuntimeError> {
        let context = &self.runtime_context;
        let lease = self
            .workspace_bindings
            .reserve(BindingPhase::Import, Some(context))
            .map_err(|error| allocation_error(context, error))?;
        if entries.is_empty() {
            return Ok((
                BindingTable::new_for_context(context)
                    .map_err(|error| allocation_error(context, error))?,
                lease,
            ));
        }
        let index_bytes = Layout::array::<(&str, usize)>(entries.len())
            .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?
            .size();
        let index_plan = HostAllocationPlan::new(
            0,
            if entries.is_empty() {
                0
            } else {
                exact_retained(context, index_bytes)?
            },
        )
        .map_err(|error| allocation_error(context, error))?;
        let index_receipt = HostAllocationReceipt::admit(Some(context), index_plan)
            .map_err(|error| allocation_error(context, error))?;
        let (indexed, index_claim) = if let Some(receipt) = index_receipt.as_ref() {
            receipt
                .with_output_allocation(context, |permit| {
                    let claim = permit.reserve_exact_claim(index_bytes)?;
                    Ok((
                        runmat_value::exact_capacity_vec(entries.len())?,
                        Some(claim),
                    ))
                })
                .map_err(|error| allocation_error(context, error))?
        } else {
            (
                runmat_value::exact_capacity_vec(entries.len()).map_err(|error| {
                    allocation_error(context, HostAllocationError::Numeric(error))
                })?,
                None,
            )
        };
        let mut indexed = ClaimedVec::new(indexed, index_claim);
        indexed.extend(
            entries
                .iter()
                .enumerate()
                .map(|(slot, (name, _))| (name.as_str(), slot)),
        );
        // WorkspaceDelta's final assignment wins for a repeated imported name.
        indexed.sort_unstable_by(|left, right| left.0.cmp(right.0).then(right.1.cmp(&left.1)));
        indexed.dedup_by(|current, previous| current.0 == previous.0);
        let mut next = BindingTable::new_for_context(context)
            .map_err(|error| allocation_error(context, error))?;
        let mut retained = next
            .growth_retained_bytes(indexed.len())
            .map_err(|error| allocation_error(context, error))?;
        for &(name, _) in indexed.iter() {
            let name_bytes = WorkspaceName::requested_bytes(name)
                .map_err(|error| allocation_error(context, HostAllocationError::Numeric(error)))?;
            let key_bytes = RetainedBindingKey::interactive_requested_bytes(name)
                .map_err(|error| allocation_error(context, HostAllocationError::Numeric(error)))?;
            add_retained(&mut retained, exact_retained(context, name_bytes)?, context)?;
            add_retained(&mut retained, exact_retained(context, key_bytes)?, context)?;
        }
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, retained)
                .map_err(|error| allocation_error(context, error))?,
        )
        .map_err(|error| allocation_error(context, error))?;
        next.reserve_admitted(context, receipt.as_ref(), indexed.len())
            .map_err(|error| allocation_error(context, error))?;
        for &(name, slot) in indexed.iter() {
            let prepared_name = if let Some(receipt) = receipt.as_ref() {
                receipt
                    .with_output_allocation(context, |permit| WorkspaceName::prepare(name, permit))
                    .map_err(|error| allocation_error(context, error))?
            } else {
                WorkspaceName::unbounded(name)
            };
            let prepared_key = if let Some(receipt) = receipt.as_ref() {
                receipt
                    .with_output_allocation(context, |permit| {
                        RetainedBindingKey::prepare_interactive(
                            self.abi_workspace_handle.0,
                            name,
                            permit,
                        )
                    })
                    .map_err(|error| allocation_error(context, error))?
            } else {
                RetainedBindingKey::unbounded(self.interactive_workspace_binding_key(name))
            };
            next.insert_prepared(
                prepared_name,
                SessionWorkspaceBinding::admitted(prepared_key, slot),
            );
        }
        if let Some(receipt) = receipt.as_ref() {
            receipt
                .checkpoint(context)
                .map_err(|error| allocation_error(context, error))?;
        }
        lease
            .checkpoint(Some(context))
            .map_err(|error| allocation_error(context, error))?;
        Ok((next, lease))
    }

    /// Reconcile externally changed canonical workspace names without cloning
    /// language values or mutating the installed compiler map on failure.
    pub(crate) fn sync_workspace_binding_metadata(&mut self) -> Result<(), RuntimeError> {
        let context = &self.runtime_context;
        let lease = self
            .workspace_bindings
            .reserve(BindingPhase::Ingress, Some(context))
            .map_err(|error| allocation_error(context, error))?;
        let (revision, raw_names) = self
            .base_workspace
            .with_binding_names(|view| {
                Ok::<_, RuntimeError>((view.revision(), borrowed_name_count(context, &view)?))
            })
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)??;
        let scratch_bytes = Layout::array::<BorrowedBinding<'_>>(raw_names)
            .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?
            .size();
        let scratch_plan = HostAllocationPlan::new(
            0,
            if raw_names == 0 {
                0
            } else {
                exact_retained(context, scratch_bytes)?
            },
        )
        .map_err(|error| allocation_error(context, error))?;
        let scratch_receipt = HostAllocationReceipt::admit(Some(context), scratch_plan)
            .map_err(|error| allocation_error(context, error))?;
        let quote = self
            .base_workspace
            .with_binding_names(|view| {
                if view.revision() != revision {
                    return Err(ingress_error(
                        "canonical workspace changed during binding quotation",
                    ));
                }
                let names =
                    collect_borrowed_names(context, scratch_receipt.as_ref(), &view, raw_names)?;
                self.quote_binding_ingress(context, &lease, &view, raw_names, &names)
            })
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)??;
        drop(scratch_receipt);
        if quote.names == 0
            && lease
                .try_borrow()
                .map_err(|_| ingress_error("workspace binding ingress is reentrant"))?
                .is_empty()
        {
            return Ok(());
        }
        let mut next = BindingTable::new_for_context(context)
            .map_err(|error| allocation_error(context, error))?;
        let mut retained = quote.retained;
        add_retained(
            &mut retained,
            next.growth_retained_bytes(quote.names)
                .map_err(|error| allocation_error(context, error))?,
            context,
        )?;
        let names_layout = Layout::array::<BorrowedBinding<'_>>(quote.raw_names)
            .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?;
        if quote.raw_names != 0 {
            add_retained(
                &mut retained,
                exact_retained(context, names_layout.size())?,
                context,
            )?;
            add_retained(
                &mut retained,
                FrameCollectionGate::retained_bytes()
                    .map_err(|error| allocation_error(context, error))?,
                context,
            )?;
            add_retained(
                &mut retained,
                frame_hash_table_retained_bytes::<usize, ()>(quote.names)
                    .map_err(|error| allocation_error(context, error))?,
                context,
            )?;
        }
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|error| allocation_error(context, error))?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|error| allocation_error(context, error))?;
        next.reserve_admitted(context, receipt.as_ref(), quote.names)
            .map_err(|error| allocation_error(context, error))?;
        self.base_workspace
            .with_binding_names(|view| {
                if view.revision() != quote.revision {
                    return Err(ingress_error(
                        "canonical workspace changed during binding preparation",
                    ));
                }
                self.prepare_binding_ingress(
                    context,
                    &lease,
                    receipt.as_ref(),
                    &mut next,
                    &quote,
                    &view,
                )
            })
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)??;
        if let Some(receipt) = receipt.as_ref() {
            receipt
                .checkpoint(context)
                .map_err(|error| allocation_error(context, error))?;
        }
        lease
            .checkpoint(Some(context))
            .map_err(|error| allocation_error(context, error))?;
        if self
            .base_workspace
            .revision()
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)?
            != quote.revision
        {
            return Err(ingress_error(
                "canonical workspace changed before binding installation",
            ));
        }
        lease
            .replace_table(next)
            .map_err(|_| ingress_error("workspace binding ingress changed before installation"))?;
        Ok(())
    }

    fn quote_binding_ingress(
        &self,
        context: &RuntimeContext,
        lease: &BindingLease,
        view: &BaseWorkspaceBindingNames<'_>,
        raw_names: usize,
        names: &[BorrowedBinding<'_>],
    ) -> Result<IngressQuote, RuntimeError> {
        let old = lease
            .try_borrow()
            .map_err(|_| ingress_error("workspace binding ingress is reentrant"))?;
        let mut quote = IngressQuote {
            revision: view.revision(),
            raw_names,
            names: 0,
            largest_live: None,
            retained: 0,
        };
        for &(name, global) in names {
            if let Some(target) = global {
                if !context
                    .session_variable_authority()
                    .contains_global_target(target)?
                {
                    return Err(ingress_error(
                        "declared global has no canonical workspace value",
                    ));
                }
            }
            quote.measure_name(context, name, global, old.get(name))?;
        }
        if quote.largest_live == Some(usize::MAX) {
            return Err(ingress_error("workspace binding slot space is exhausted"));
        }
        Ok(quote)
    }

    fn prepare_binding_ingress(
        &self,
        context: &RuntimeContext,
        lease: &BindingLease,
        receipt: Option<&HostAllocationReceipt>,
        next: &mut BindingTable,
        quote: &IngressQuote,
        view: &BaseWorkspaceBindingNames<'_>,
    ) -> Result<(), RuntimeError> {
        if quote.names == 0 {
            return Ok(());
        }
        let names = collect_borrowed_names(context, receipt, view, quote.raw_names)?;
        if names.len() != quote.names {
            return Err(ingress_error(
                "canonical workspace names changed during binding preparation",
            ));
        }
        let gate = if let Some(receipt) = receipt {
            FrameCollectionGate::prepare(context, receipt)
                .map_err(|error| allocation_error(context, error))?
        } else {
            FrameCollectionGate::unbounded()
        };
        let mut used = HashSet::with_hasher_in(RandomState::new(), gate.allocator());
        {
            let _installed = if let Some(receipt) = receipt {
                gate.install(context, receipt)
            } else {
                gate.install_unbounded()
            }
            .map_err(|error| allocation_error(context, error))?;
            used.try_reserve(quote.names)
                .map_err(|_| allocation_error(context, HostAllocationError::Overflow))?;
        }
        let mut fresh = quote
            .largest_live
            .map_or(Some(0), |slot| slot.checked_add(1));
        let old = lease
            .try_borrow()
            .map_err(|_| ingress_error("workspace binding ingress is reentrant"))?;
        for &(name, global) in names.iter() {
            let prior = old.get(name);
            let slot = if let Some(prior) = prior.filter(|binding| used.insert(binding.slot)) {
                prior.slot
            } else {
                let slot = fresh
                    .filter(|slot| *slot != usize::MAX)
                    .ok_or_else(|| ingress_error("workspace binding slot space is exhausted"))?;
                used.insert(slot);
                fresh = slot.checked_add(1);
                slot
            };
            let prepared_name = if let Some(receipt) = receipt {
                receipt
                    .with_output_allocation(context, |permit| WorkspaceName::prepare(name, permit))
                    .map_err(|error| allocation_error(context, error))?
            } else {
                WorkspaceName::unbounded(name)
            };
            let prepared_key = if let Some(receipt) = receipt {
                receipt
                    .with_output_allocation(context, |permit| {
                        if let Some(global) = global {
                            RetainedBindingKey::prepare_session_global_target(
                                self.abi_workspace_handle.0,
                                global,
                                permit,
                            )
                        } else if let Some(prior) =
                            prior.filter(|binding| eligible_local_key(binding))
                        {
                            RetainedBindingKey::prepare(&prior.key, permit)
                        } else {
                            RetainedBindingKey::prepare_interactive(
                                self.abi_workspace_handle.0,
                                name,
                                permit,
                            )
                        }
                    })
                    .map_err(|error| allocation_error(context, error))?
            } else {
                let key = if let Some(global) = global {
                    session_global_key(self.abi_workspace_handle.0, global)
                } else {
                    prior
                        .filter(|binding| eligible_local_key(binding))
                        .map(|binding| binding.key.clone())
                        .unwrap_or_else(|| self.interactive_workspace_binding_key(name))
                };
                RetainedBindingKey::unbounded(key)
            };
            next.insert_prepared(
                prepared_name,
                SessionWorkspaceBinding::admitted(prepared_key, slot),
            );
        }
        Ok(())
    }
}

impl IngressQuote {
    fn measure_name(
        &mut self,
        context: &RuntimeContext,
        name: &str,
        global: Option<&GlobalBindingTarget>,
        prior: Option<&SessionWorkspaceBinding>,
    ) -> Result<(), RuntimeError> {
        self.names = self
            .names
            .checked_add(1)
            .ok_or_else(|| allocation_error(context, HostAllocationError::Overflow))?;
        if let Some(prior) = prior {
            self.largest_live = Some(
                self.largest_live
                    .map_or(prior.slot, |old| old.max(prior.slot)),
            );
        }
        let name_bytes = WorkspaceName::requested_bytes(name)
            .map_err(|error| allocation_error(context, HostAllocationError::Numeric(error)))?;
        add_retained(
            &mut self.retained,
            exact_retained(context, name_bytes)?,
            context,
        )?;
        let key_retained = if let Some(global) = global {
            RetainedBindingKey::session_global_retained_bytes(global)
                .map_err(|error| allocation_error(context, error))?
        } else {
            let bytes = if let Some(prior) = prior.filter(|binding| eligible_local_key(binding)) {
                RetainedBindingKey::requested_bytes(&prior.key)
            } else {
                RetainedBindingKey::interactive_requested_bytes(name)
            }
            .map_err(|error| allocation_error(context, HostAllocationError::Numeric(error)))?;
            exact_retained(context, bytes)?
        };
        add_retained(&mut self.retained, key_retained, context)
    }
}
