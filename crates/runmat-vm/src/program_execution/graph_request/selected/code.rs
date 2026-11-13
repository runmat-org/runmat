//! The code half of an admitted selected graph has no RuntimeContext, service
//! port, resource grant, timing origin or cancellation token. Only its active
//! worker wrapper retains the contexts for that one execution.
use super::*;
pub(in super::super) struct RetainedSelectedGraph {
    pub units: Vec<RetainedExecutableUnit>,
    pub _receipts: Vec<Option<Rc<crate::SelectedUnitInteropReceipt>>>,
    pub issuer: GraphSourceIssuer,
    pub source_map: Arc<ExecutableSourceMapV2>,
    pub layouts: HashMap<GraphBoundOwner, GraphLexicalLayout>,
    pub proofs: HashMap<GraphBoundOwner, GraphCallableProof>,
    pub bindings: HashMap<GraphBoundOwner, BoundFunctionBinding>,
    pub class_products: HashMap<GraphBoundOwner, Rc<crate::SelectedClassProduct>>,
    pub object_classes: HashMap<GraphBoundOwner, Rc<runmat_value::SelectedObjectClassFacts>>,
    pub static_bindings: Rc<
        HashMap<
            (
                GraphStaticCaller,
                runmat_execution::value::GraphStaticReferenceSite,
            ),
            BoundFunctionBinding,
        >,
    >,
    pub sealed_sites: Rc<
        HashSet<(
            GraphStaticCaller,
            runmat_execution::value::GraphStaticReferenceSite,
        )>,
    >,
}
impl RetainedSelectedGraph {
    /// Project one already-admitted unit into this call's explicit current
    /// authority. Only its source/language/static selection changes; resource,
    /// cancellation, Clock, worker and output projections come from current.
    pub(in super::super) fn project_runtime(
        &self,
        unit: GraphProgramUnitId,
        current: &RuntimeContext,
        selected_foreign: Option<&Rc<dyn RuntimeForeignService>>,
    ) -> Result<RuntimeContext, String> {
        let position = usize::from(unit.0);
        let code = self
            .units
            .get(position)
            .ok_or("retained graph unit is absent")?;
        let owner = code
            .selected_source_owner()
            .ok_or("retained graph source owner is absent")?;
        if !owner.is_issued_in(current) {
            return Err("retained graph belongs to a different or retired session".into());
        }
        let port = [selected_foreign.cloned()];
        let runtime = selected_unit_runtime(
            current,
            Some(&port),
            0,
            code.interop(),
            code.revision().environment().compatibility_mode,
        )?;
        let runtime = owner
            .bind_runtime(runtime)
            .map_err(|error| error.to_string())?;
        let bindings = Rc::clone(&self.static_bindings);
        let sealed = Rc::clone(&self.sealed_sites);
        Ok(
            runtime.with_selected_static_site_resolver(Rc::new(move |origin, site| {
                let caller = match origin {
                    Some(origin) => GraphStaticCaller::Function(GraphBoundOwner { unit, origin }),
                    None => GraphStaticCaller::Script(unit),
                };
                if sealed.contains(&(caller, site)) {
                    bindings
                        .get(&(caller, site))
                        .cloned()
                        .map(Some)
                        .ok_or("sealed static site has no selected target")
                } else {
                    Ok(None)
                }
            })),
        )
    }
    pub(in super::super) fn owner_for_binding(
        &self,
        binding: &BoundFunctionBinding,
    ) -> Result<GraphBoundOwner, String> {
        let mut owners = self
            .bindings
            .iter()
            .filter(|(_, selected)| *selected == binding)
            .map(|(owner, _)| *owner);
        let owner = owners
            .next()
            .ok_or("bound callback has no exact selected graph unit")?;
        if owners.next().is_some() {
            return Err("bound callback has ambiguous selected graph units".into());
        }
        self.unit(owner)?;
        Ok(owner)
    }

    pub(in super::super) fn unit(
        &self,
        owner: GraphBoundOwner,
    ) -> Result<&RetainedExecutableUnit, String> {
        let selected = self
            .units
            .get(usize::from(owner.unit.0))
            .ok_or("graph bound owner selects an absent V3 unit")?;
        let binding = self
            .bindings
            .get(&owner)
            .ok_or("graph bound owner has no admitted definition")?;
        let body = selected_bound_body(selected.registry(), binding)?;
        if body.binding_origin != Some(owner.origin)
            || !body.selected_source_owner.as_ref().is_some_and(|issued| {
                selected
                    .selected_source_owner()
                    .is_some_and(|selected| issued.same_issuance(selected))
            })
        {
            return Err("graph bound owner differs from selected V3 body/source".into());
        }
        Ok(selected)
    }

    pub(in super::super) fn owner_for_authority(
        &self,
        origin: runmat_execution::FunctionBindingOrigin,
        authority: &BoundFunctionAuthority,
    ) -> Result<GraphBoundOwner, String> {
        let session = self
            .units
            .first()
            .and_then(RetainedExecutableUnit::selected_source_owner)
            .ok_or("graph has no selected issuing session")?
            .session_lease();
        if authority.export_origin(&session) != Some(origin) {
            return Err("returned callable differs from its issuing session/origin".into());
        }
        let mut found = None;
        for (index, unit) in self.units.iter().enumerate() {
            for body in unit.registry().selected_execution_bodies() {
                if body.binding_origin != Some(origin) {
                    continue;
                }
                let Some(selected_owner) = unit.selected_source_owner() else {
                    continue;
                };
                if !body
                    .selected_source_owner
                    .as_ref()
                    .is_some_and(|owner| owner.same_issuance(selected_owner))
                {
                    continue;
                }
                if let Some(lease) = authority.local_lease(body.function.0) {
                    if !lease.matches(&body.definition_owner) {
                        continue;
                    }
                } else if authority.local_definition().is_some() {
                    continue;
                }
                let owner = GraphBoundOwner {
                    unit: GraphProgramUnitId(
                        u16::try_from(index)
                            .map_err(|_| "graph unit index exceeds its ABI".to_string())?,
                    ),
                    origin,
                };
                if found.replace(owner).is_some() {
                    return Err("returned callable has ambiguous selected V3 issuances".into());
                }
            }
        }
        found.ok_or_else(|| "returned callable has no exact selected V3 owner".into())
    }
}
