use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::value::{GraphProgramUnitId, GRAPH_MAX_SELECTED_UNITS};
use runmat_execution::FunctionBindingOrigin;
use runmat_runtime::execution::value_codec::{
    GraphFrameSourceRequirement, GraphSourceUnitFacts, ValueCodecError,
};
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_runtime::CallFrameSourceOwner;
use runmat_types::SourceId;
use runmat_vm::RetainedExecutableSourceOwner;

use super::super::super::generation::CallableArtifactIndex;
use super::super::SelectedGraphPrimary;
use super::{invalid, SelectedDependencyBody};

#[path = "units/declaration.rs"]
mod declaration;

pub(in crate::session::source_evaluation::selected_graph) struct SelectedGraphUnit {
    pub(in crate::session::source_evaluation::selected_graph) bytes: Arc<[u8]>,
    pub(in crate::session::source_evaluation::selected_graph) facts: GraphSourceUnitFacts,
    pub(in crate::session::source_evaluation::selected_graph) source_owner:
        SelectedProgramSourceOwner,
    fingerprints: HashMap<FunctionBindingOrigin, String>,
    interop_receipt: Option<std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>>,
}

fn checked_interop_receipt(
    admitted: &runmat_vm::AdmittedExecutableUnit,
    bytes: &Arc<[u8]>,
    source_owner: &SelectedProgramSourceOwner,
    receipt: Option<&std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>>,
) -> Result<(), ValueCodecError> {
    let interop = &admitted.envelope().manifest.interop;
    let declared = !interop.foreign_types.is_empty()
        || !interop.adapters.is_empty()
        || !interop.adapter_contracts.is_empty();
    if (declared && receipt.is_none())
        || receipt.is_some_and(|receipt| !receipt.matches_selected_unit(bytes, source_owner))
    {
        return Err(invalid(
            "selected V3 foreign receipt differs from its issued unit",
        ));
    }
    Ok(())
}

fn same_receipt(
    left: Option<&std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>>,
    right: Option<&std::rc::Rc<runmat_vm::SelectedUnitInteropReceipt>>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => left.same_selected_admission(right),
        _ => false,
    }
}

pub(in crate::session::source_evaluation::selected_graph) struct SelectedGraphUnits {
    units: Vec<SelectedGraphUnit>,
    total_bytes: usize,
}

impl SelectedGraphUnits {
    pub(in crate::session::source_evaluation::selected_graph) fn foreign_policies(
        &self,
    ) -> Vec<Option<runmat_execution::ForeignPolicyRef>> {
        // Pure captures use the canonical empty policy vector. Ordered holes
        // are meaningful only when another selected unit has an issued policy.
        if self.units.iter().all(|unit| unit.interop_receipt.is_none()) {
            return Vec::new();
        }
        self.units
            .iter()
            .map(|unit| {
                unit.interop_receipt
                    .as_ref()
                    .map(|receipt| receipt.foreign_policy())
            })
            .collect()
    }

    pub(in crate::session::source_evaluation::selected_graph) fn new(
        primary: &SelectedGraphPrimary,
    ) -> Result<Self, ValueCodecError> {
        let bytes = primary
            .canonical_v3()
            .map_err(|_| invalid("selected primary V3 product cannot be encoded"))?;
        let admitted = runmat_vm::admit_executable_unit(&bytes)
            .map_err(|_| invalid("selected primary V3 product is invalid"))?;
        if !admitted.linked_native_imports().is_empty() {
            return Err(invalid(
                "selected Graph transfer has no linked-native code lease handoff",
            ));
        }
        let revision = &admitted.envelope().manifest.identity.program;
        if revision
            .identity_digest()
            .map_err(|_| invalid("selected primary program identity cannot be encoded"))?
            != primary.expected_program()
            || primary
                .revision()
                .is_some_and(|expected| expected != revision)
        {
            return Err(invalid(
                "selected primary V3 revision differs from its code owner",
            ));
        }
        let fingerprints = admitted
            .retained_callable_origin_fingerprints()
            .map_err(|_| invalid("selected primary V3 definitions are invalid"))?;
        if primary.retained_origin().is_none() {
            let selected = primary.current_body_fingerprints().map_err(|reason| {
                invalid(format!(
                    "current selected V3 definitions are invalid: {reason}"
                ))
            })?;
            if selected != fingerprints {
                return Err(invalid(
                    "selected primary V3 bodies differ from their published definitions",
                ));
            }
        } else if let Some(origin) = primary.retained_origin() {
            let function = primary
                .retained_function()
                .ok_or_else(|| invalid("retained V3 body has no selected function"))?;
            let (selected_origin, selected_fingerprint) = primary
                .selected_registry()
                .selected_runtime_body_fingerprint(function)
                .map_err(|_| invalid("retained selected definition is invalid"))?;
            if selected_origin != origin || fingerprints.get(&origin) != Some(&selected_fingerprint)
            {
                return Err(invalid(
                    "selected primary V3 body differs from its retained definition",
                ));
            }
        }
        let facts = GraphSourceUnitFacts::capture_owner(
            primary.source_owner().clone(),
            revision,
            Arc::new(admitted.admitted_source_map().clone()),
        )?;
        let interop_receipt = primary.selected_interop_receipt().cloned();
        checked_interop_receipt(
            &admitted,
            &bytes,
            primary.source_owner(),
            interop_receipt.as_ref(),
        )?;
        let total_bytes = bytes.len();
        Ok(Self {
            units: vec![SelectedGraphUnit {
                bytes,
                facts,
                source_owner: primary.source_owner().clone(),
                fingerprints,
                interop_receipt,
            }],
            total_bytes,
        })
    }

    pub(in crate::session::source_evaluation::selected_graph) fn add_dependency(
        &mut self,
        artifacts: &CallableArtifactIndex,
        body: &SelectedDependencyBody,
    ) -> Result<GraphProgramUnitId, ValueCodecError> {
        if let Some((position, unit)) = self.units.iter().enumerate().find(|(_, unit)| {
            unit.facts.program() == body.origin.program
                && unit.source_owner.same_issuance(&body.source_owner)
        }) {
            if body.product.as_ref().is_some_and(|product| {
                runmat_execution::Digest::sha256(unit.bytes.as_ref()) != product.full_digest()
            }) {
                return Err(invalid(
                    "selected source issuance has conflicting executable products",
                ));
            }
            if unit.fingerprints.get(&body.origin) != Some(&body.fingerprint) {
                return Err(invalid(
                    "selected V3 unit differs from its leased definition",
                ));
            }
            if !same_receipt(unit.interop_receipt.as_ref(), body.interop_receipt.as_ref()) {
                return Err(invalid(
                    "one selected source issuance has conflicting foreign receipts",
                ));
            }
            return u16::try_from(position)
                .map(GraphProgramUnitId)
                .map_err(|_| invalid("selected graph unit ordinal exceeds u16"));
        }
        if self.units.len() >= GRAPH_MAX_SELECTED_UNITS {
            return Err(invalid("selected graph exceeds its unit bound"));
        }
        let unit = select_retained_unit(artifacts, body)?;
        let total_bytes = self
            .total_bytes
            .checked_add(unit.bytes.len())
            .ok_or_else(|| invalid("selected graph unit bytes overflow"))?;
        if total_bytes > runmat_execution::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES {
            return Err(invalid(
                "selected graph unit bytes exceed their shared bound",
            ));
        }
        let position = self.units.len();
        self.units.push(unit);
        self.total_bytes = total_bytes;
        u16::try_from(position)
            .map(GraphProgramUnitId)
            .map_err(|_| invalid("selected graph unit ordinal exceeds u16"))
    }

    /// Select a declaration-retained static target from its immutable code
    /// product, never from the mutable callable artifact index by origin.
    pub(in crate::session::source_evaluation::selected_graph) fn add_static_product_target(
        &mut self,
        artifacts: &CallableArtifactIndex,
        target: &runmat_vm::SelectedStaticProductTarget,
    ) -> Result<GraphProgramUnitId, ValueCodecError> {
        let origin = target.import().target;
        let product = target.product();
        let fingerprint = product
            .body_fingerprint(origin)
            .ok_or_else(|| invalid("selected static target has no admitted body fingerprint"))?;
        self.add_dependency(
            artifacts,
            &SelectedDependencyBody {
                origin,
                source_owner: product.source_owner().clone(),
                fingerprint: fingerprint.to_owned(),
                product: Some(Rc::clone(product)),
                static_capture: None,
                interop_receipt: product.interop_receipt().cloned(),
            },
        )
    }

    pub(in crate::session::source_evaluation::selected_graph) fn unit_for_product(
        &self,
        product: &Rc<runmat_vm::SelectedExecutableProduct>,
    ) -> Result<GraphProgramUnitId, ValueCodecError> {
        let position = self
            .units
            .iter()
            .position(|unit| {
                unit.source_owner.same_issuance(product.source_owner())
                    && (Arc::ptr_eq(&unit.bytes, product.canonical_v3())
                        || unit.bytes.as_ref() == product.canonical_v3().as_ref())
            })
            .ok_or_else(|| invalid("captured static caller product is not admitted"))?;
        u16::try_from(position)
            .map(GraphProgramUnitId)
            .map_err(|_| invalid("captured static caller unit exceeds u16"))
    }

    pub(in crate::session::source_evaluation::selected_graph) fn units(
        &self,
    ) -> &[SelectedGraphUnit] {
        &self.units
    }

    pub(in crate::session::source_evaluation::selected_graph) fn selected_bundle_archives(
        &self,
    ) -> Vec<Option<Arc<[u8]>>> {
        self.units
            .iter()
            .map(|unit| {
                unit.interop_receipt
                    .as_ref()
                    .map(|receipt| Arc::clone(receipt.archive()))
            })
            .collect()
    }

    pub(in crate::session::source_evaluation::selected_graph) fn source_facts(
        &self,
    ) -> Vec<GraphSourceUnitFacts> {
        self.units.iter().map(|unit| unit.facts.clone()).collect()
    }

    pub(in crate::session::source_evaluation::selected_graph) fn owner_units(
        &self,
    ) -> Result<Vec<(GraphProgramUnitId, SelectedProgramSourceOwner)>, ValueCodecError> {
        self.units
            .iter()
            .enumerate()
            .map(|(position, unit)| {
                let position = u16::try_from(position)
                    .map_err(|_| invalid("selected graph unit ordinal exceeds u16"))?;
                Ok((GraphProgramUnitId(position), unit.source_owner.clone()))
            })
            .collect()
    }

    /// A Local diagnostic never adds executable code solely for presentation.
    /// A frame with no diagnostic instead requires an exact, already-admitted
    /// unit. Its issuance token only narrows the source owner; the bound unit's
    /// canonical V3 and source map supply the executable proof.
    pub(in crate::session::source_evaluation::selected_graph) fn add_frame_source(
        &mut self,
        source: SourceId,
        frame_owner: Option<&CallFrameSourceOwner>,
        requirement: GraphFrameSourceRequirement,
        source_owners: &[std::rc::Rc<RetainedExecutableSourceOwner>],
    ) -> Result<(), ValueCodecError> {
        let already_selected = self
            .units
            .iter()
            .filter(|unit| owner_matches_frame(&unit.source_owner, source, frame_owner))
            .count();
        if requirement == GraphFrameSourceRequirement::MayDetach {
            // An unmatched or numerically reused Local frame is encoded as a
            // detached diagnostic by the Runtime leaf codec. It cannot cause
            // an extra V3 product to enter the worker bundle.
            return (frame_owner.is_none() || already_selected <= 1)
                .then_some(())
                .ok_or_else(|| invalid("diagnostic frame has ambiguous selected issuances"));
        }
        let candidates = self
            .units
            .iter()
            .map(|unit| unit.source_owner.clone())
            .chain(
                source_owners
                    .iter()
                    .map(|unit| unit.selected_source_owner().clone()),
            )
            .filter(|owner| owner_matches_frame(owner, source, frame_owner))
            .collect::<Vec<_>>();
        // The owners above need not be adjacent: a later generation may hold
        // an earlier source owner after a separately admitted product.
        let mut unique = Vec::new();
        for owner in candidates {
            if !unique
                .iter()
                .any(|prior: &SelectedProgramSourceOwner| prior.same_issuance(&owner))
            {
                unique.push(owner);
            }
        }
        let [selected] = unique.as_slice() else {
            return Err(invalid(
                "exception frame has no unique admitted source owner",
            ));
        };
        if self
            .units
            .iter()
            .any(|unit| unit.source_owner.same_issuance(selected))
        {
            return Ok(());
        }
        let bound = source_owners
            .iter()
            .find(|unit| unit.selected_source_owner().same_issuance(selected))
            .ok_or_else(|| invalid("exception source has no retained admitted unit"))?;
        if !bound.selected_source_owner().is_live() {
            return Err(invalid("exception source issuer has retired"));
        }
        let bytes = bound.canonical_v3();
        let admitted = runmat_vm::admit_executable_unit(bytes)
            .map_err(|_| invalid("exception source V3 product is invalid"))?;
        if !admitted.linked_native_imports().is_empty() {
            return Err(invalid(
                "exception graph source has linked-native imports without code handoff",
            ));
        }
        let admitted_map = admitted.admitted_source_map();
        if selected.portable_map() != admitted_map {
            return Err(invalid("exception source differs from its admitted V3 map"));
        }
        let revision = &admitted.envelope().manifest.identity.program;
        let facts = GraphSourceUnitFacts::capture_owner(
            selected.clone(),
            revision,
            Arc::new(admitted_map.clone()),
        )?;
        let fingerprints = admitted
            .retained_callable_origin_fingerprints()
            .map_err(|_| invalid("exception source V3 definitions are invalid"))?;
        checked_interop_receipt(&admitted, bytes, selected, None)?;
        let total_bytes = self
            .total_bytes
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("selected graph unit bytes overflow"))?;
        if self.units.len() >= GRAPH_MAX_SELECTED_UNITS
            || total_bytes > runmat_execution::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES
        {
            return Err(invalid(
                "exception source exceeds selected graph unit bounds",
            ));
        }
        self.units.push(SelectedGraphUnit {
            bytes: Arc::clone(bytes),
            facts,
            source_owner: selected.clone(),
            fingerprints,
            interop_receipt: None,
        });
        self.total_bytes = total_bytes;
        Ok(())
    }
}

fn owner_matches_frame(
    owner: &SelectedProgramSourceOwner,
    source: SourceId,
    frame_owner: Option<&CallFrameSourceOwner>,
) -> bool {
    owner
        .selected_sources()
        .map(|(ordinal, reference, _)| (ordinal, reference.source_id()))
        .chain(
            owner
                .durable_aliases()
                .map(|(reference, ordinal)| (ordinal, reference.source_id())),
        )
        .any(|(ordinal, candidate)| {
            candidate == source
                && frame_owner.is_none_or(|expected| {
                    owner
                        .call_frame_source_owner_for(ordinal)
                        .is_some_and(|actual| actual.same_issuance(expected))
                })
        })
}

/// A body carrier selects its exact immutable bytes and issuance. The legacy
/// artifact index is used only when that older body has no attached carrier.
/// Both paths still compare the VM's canonical definition fingerprint.
fn select_retained_unit(
    artifacts: &CallableArtifactIndex,
    body: &SelectedDependencyBody,
) -> Result<SelectedGraphUnit, ValueCodecError> {
    let bytes = if let Some(product) = body.product.as_ref() {
        if !product.source_owner().same_issuance(&body.source_owner) {
            return Err(invalid("retained callable carrier changed source issuance"));
        }
        Arc::clone(product.canonical_v3())
    } else {
        artifacts
            .canonical_unit_for_origin(&body.origin)
            .ok_or_else(|| invalid("retained callable has no selected V3 product"))?
    };
    let admitted = runmat_vm::admit_executable_unit(&bytes)
        .map_err(|_| invalid("retained callable V3 product is invalid"))?;
    if !admitted.linked_native_imports().is_empty() {
        return Err(invalid(
            "retained Graph callable has linked-native imports without code handoff",
        ));
    }
    let revision = &admitted.envelope().manifest.identity.program;
    if revision
        .identity_digest()
        .map_err(|_| invalid("retained program identity cannot be encoded"))?
        != body.origin.program
    {
        return Err(invalid("retained callable revision differs from its lease"));
    }
    let fingerprints = admitted
        .retained_callable_origin_fingerprints()
        .map_err(|_| invalid("retained callable definitions are invalid"))?;
    checked_interop_receipt(
        &admitted,
        &bytes,
        &body.source_owner,
        body.interop_receipt.as_ref(),
    )?;
    if fingerprints.get(&body.origin) != Some(&body.fingerprint) {
        return Err(invalid(
            "retained V3 body differs from its leased definition",
        ));
    }
    let facts = GraphSourceUnitFacts::capture_owner(
        body.source_owner.clone(),
        revision,
        Arc::new(admitted.admitted_source_map().clone()),
    )?;
    Ok(SelectedGraphUnit {
        bytes,
        facts,
        source_owner: body.source_owner.clone(),
        fingerprints,
        interop_receipt: body.interop_receipt.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{Digest, ExecutableSourceMapV2, ExecutableSourceRecordV2};
    use runmat_runtime::source_context::{
        bind_portable_source_map_v2, lease_source_context_in_context,
    };
    use runmat_types::ProgramSourceId;

    fn owner() -> (
        runmat_runtime::context::RuntimeContext,
        SelectedProgramSourceOwner,
    ) {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let text = "value = 1;";
        let map = ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "frame.m".into(),
            content_revision: Digest::sha256(text.as_bytes()),
            display_name: "frame.m".into(),
            text: text.into(),
        }])
        .expect("portable map");
        let runtime = bind_portable_source_map_v2(runtime, &map).expect("bound portable map");
        let owner = SelectedProgramSourceOwner::capture(&runtime, &map).expect("source owner");
        (runtime, owner)
    }

    #[test]
    fn frame_selection_requires_exact_issuance_and_accepts_validated_durable_alias() {
        let (runtime, selected) = owner();
        let (_, foreign) = owner();
        let issued = runtime
            .program_source_ref(ProgramSourceId(0))
            .expect("issued source");
        let selected_token = selected
            .call_frame_source_owner_for(ProgramSourceId(0))
            .expect("selected token");
        let foreign_token = foreign
            .call_frame_source_owner_for(ProgramSourceId(0))
            .expect("foreign token");
        assert!(owner_matches_frame(
            &selected,
            issued.source_id(),
            Some(&selected_token)
        ));
        assert!(!owner_matches_frame(
            &selected,
            issued.source_id(),
            Some(&foreign_token)
        ));
        let durable = lease_source_context_in_context(&runtime, "frame.m", None, "value = 1;")
            .expect("durable source");
        let with_alias = selected
            .with_durable_aliases(&runtime, [(issued, durable.source_ref())])
            .expect("validated alias");
        assert!(owner_matches_frame(
            &with_alias,
            durable.source_id(),
            Some(&selected_token)
        ));
        assert!(!owner_matches_frame(
            &selected,
            durable.source_id(),
            Some(&selected_token)
        ));
    }

    #[test]
    fn unselected_old_local_diagnostic_may_detach_but_a_required_frame_cannot() {
        let (old_runtime, old) = owner();
        let (new_runtime, new) = owner();
        let old_source = old_runtime
            .program_source_ref(ProgramSourceId(0))
            .expect("old issued source")
            .source_id();
        let new_source = new_runtime
            .program_source_ref(ProgramSourceId(0))
            .expect("new issued source")
            .source_id();
        assert_eq!(old_source, new_source, "numeric source IDs are reused");
        let old_token = old
            .call_frame_source_owner_for(ProgramSourceId(0))
            .expect("old issuer token");
        assert!(!owner_matches_frame(&new, old_source, Some(&old_token)));
        let mut units = SelectedGraphUnits {
            units: Vec::new(),
            total_bytes: 0,
        };
        assert!(units
            .add_frame_source(
                old_source,
                Some(&old_token),
                GraphFrameSourceRequirement::MayDetach,
                &[],
            )
            .is_ok());
        assert!(units
            .add_frame_source(
                old_source,
                Some(&old_token),
                GraphFrameSourceRequirement::RequiredAdmitted,
                &[],
            )
            .is_err());
    }
}
