use crate::model::validation_storage::{lookup, ValidationWork};

use super::super::{ExactBRepModel, ExactBRepTopology, GeometryContractError};
use super::{
    definition_validation::{
        curve_domain, curve_dynamic_value_count, curve_is_degenerate, curve_is_periodic,
        pcurve_domain, pcurve_dynamic_value_count, surface_dynamic_value_count,
        surface_periodicity, validate_curve, validate_mass_properties, validate_pcurve,
        validate_surface, validate_token, validate_trim_classifier,
    },
    CurveEvaluatorId, ExactEvaluatorRegistry, ExactMassPropertiesImplementation,
    MassPropertiesEvaluatorId, PcurveEvaluatorId, SurfaceEvaluatorId, TrimClassifierId,
    EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
};

const MAX_EVALUATOR_RECORDS: usize = 10_000_000;
const MAX_DYNAMIC_NUMERIC_VALUES: usize = 100_000_000;

impl ExactEvaluatorRegistry {
    pub fn validate_against(
        &self,
        topology: &ExactBRepTopology,
        model: &ExactBRepModel,
    ) -> Result<(), GeometryContractError> {
        self.validate_admitted(topology, model, ValidationWork(None))
    }
    pub fn validate_against_with_control(
        &self,
        topology: &ExactBRepTopology,
        model: &ExactBRepModel,
        control: &dyn crate::GeometryEvaluationControl,
    ) -> Result<(), GeometryContractError> {
        self.validate_admitted(topology, model, ValidationWork(Some(control)))
    }
    fn validate_admitted(
        &self,
        topology: &ExactBRepTopology,
        model: &ExactBRepModel,
        work: ValidationWork<'_>,
    ) -> Result<(), GeometryContractError> {
        if let Some(control) = work.0 {
            topology.validate_against_with_control(model, control)?;
        } else {
            topology.validate_against(model)?;
        }
        self.validate_for_validated_topology(topology, model, work)
    }
    /// Internal only: the geometry component owner has validated this exact
    /// immutable topology in the same call before reaching this validator.
    pub(crate) fn validate_for_validated_topology(
        &self,
        topology: &ExactBRepTopology,
        model: &ExactBRepModel,
        work: ValidationWork<'_>,
    ) -> Result<(), GeometryContractError> {
        work.checkpoint()?;
        if self.schema_version != EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION {
            return Err(invalid(
                "exact evaluator registry schema",
                "unsupported version",
            ));
        }
        validate_token("evaluator registry kernel ABI", &self.kernel_abi)?;
        if self.kernel_abi != model.kernel_abi {
            return Err(invalid(
                "evaluator registry kernel ABI",
                "must exactly match the admitted exact-model kernel ABI",
            ));
        }
        if !model.capabilities.complete_for_meshing() {
            return Err(invalid(
                "exact evaluator capabilities",
                "registry admission requires the complete exact-query capability contract",
            ));
        }
        let record_count = self
            .curves
            .len()
            .saturating_add(self.pcurves.len())
            .saturating_add(self.surfaces.len())
            .saturating_add(self.trim_classifiers.len())
            .saturating_add(self.mass_properties.len());
        if record_count == 0 || record_count > MAX_EVALUATOR_RECORDS {
            return Err(invalid(
                "exact evaluator record count",
                "registry must be nonempty and within the hard record bound",
            ));
        }
        let dynamic_value_count = self
            .curves
            .iter()
            .map(|record| curve_dynamic_value_count(&record.implementation))
            .chain(
                self.pcurves
                    .iter()
                    .map(|record| pcurve_dynamic_value_count(&record.implementation)),
            )
            .chain(
                self.surfaces
                    .iter()
                    .map(|record| surface_dynamic_value_count(&record.implementation)),
            )
            .fold(0usize, usize::saturating_add);
        if dynamic_value_count > MAX_DYNAMIC_NUMERIC_VALUES {
            return Err(invalid(
                "exact evaluator numeric payload",
                "registry exceeds the hard aggregate dynamic-value bound",
            ));
        }

        require_exact_inventory(
            "curve evaluator inventory",
            "curve evaluators",
            &self.curves,
            |record| &record.id,
            topology
                .edges
                .iter()
                .map(|entity| &entity.curve_evaluator_id),
            CurveEvaluatorId::validate,
            work,
        )?;
        require_exact_inventory(
            "pcurve evaluator inventory",
            "pcurve evaluators",
            &self.pcurves,
            |record| &record.id,
            topology
                .coedges
                .iter()
                .map(|entity| &entity.pcurve_evaluator_id),
            PcurveEvaluatorId::validate,
            work,
        )?;
        require_exact_inventory(
            "surface evaluator inventory",
            "surface evaluators",
            &self.surfaces,
            |record| &record.id,
            topology
                .faces
                .iter()
                .map(|entity| &entity.surface_evaluator_id),
            SurfaceEvaluatorId::validate,
            work,
        )?;
        require_exact_inventory(
            "trim classifier evaluator inventory",
            "trim classifiers",
            &self.trim_classifiers,
            |record| &record.id,
            topology
                .faces
                .iter()
                .map(|entity| &entity.trim_classifier_id),
            TrimClassifierId::validate,
            work,
        )?;
        require_exact_inventory(
            "mass-properties evaluator inventory",
            "mass-properties evaluators",
            &self.mass_properties,
            |record| &record.id,
            topology
                .bodies
                .iter()
                .map(|entity| &entity.mass_properties_evaluator_id),
            MassPropertiesEvaluatorId::validate,
            work,
        )?;

        for record in &self.curves {
            work.checkpoint()?;
            validate_curve(&record.implementation)?;
        }
        for record in &self.pcurves {
            work.checkpoint()?;
            validate_pcurve(&record.implementation)?;
        }
        for record in &self.surfaces {
            work.checkpoint()?;
            validate_surface(&record.implementation)?;
        }
        for record in &self.trim_classifiers {
            work.checkpoint()?;
            validate_trim_classifier(&record.implementation)?;
        }
        for record in &self.mass_properties {
            work.checkpoint()?;
            validate_mass_properties(&record.implementation)?;
        }
        self.kernel_representation_digest()?;

        validate_topology_claims(self, topology, work)
    }
}

fn validate_topology_claims(
    registry: &ExactEvaluatorRegistry,
    topology: &ExactBRepTopology,
    work: ValidationWork<'_>,
) -> Result<(), GeometryContractError> {
    for edge in &topology.edges {
        work.checkpoint()?;
        let implementation = lookup(&registry.curves, &edge.curve_evaluator_id, |record| {
            &record.id
        })
        .map(|record| &record.implementation)
        .ok_or_else(|| invalid("edge curve evaluator", "inventory index is incomplete"))?;
        if curve_is_periodic(implementation).is_some_and(|periodic| periodic != edge.is_periodic) {
            return Err(invalid(
                "edge periodicity",
                "topology must agree with the portable curve definition",
            ));
        }
        if curve_is_degenerate(implementation)
            .is_some_and(|degenerate| degenerate != edge.is_degenerate)
        {
            return Err(invalid(
                "edge degeneracy",
                "topology must agree with the portable curve definition",
            ));
        }
    }
    for coedge in &topology.coedges {
        work.checkpoint()?;
        let edge = lookup(&topology.edges, &coedge.edge_id, |edge| &edge.id)
            .ok_or_else(|| invalid("coedge edge", "topology index is incomplete"))?;
        let curve = lookup(&registry.curves, &edge.curve_evaluator_id, |record| {
            &record.id
        })
        .map(|record| &record.implementation)
        .ok_or_else(|| invalid("edge curve evaluator", "inventory index is incomplete"))?;
        let pcurve = lookup(&registry.pcurves, &coedge.pcurve_evaluator_id, |record| {
            &record.id
        })
        .map(|record| &record.implementation)
        .ok_or_else(|| invalid("coedge pcurve evaluator", "inventory index is incomplete"))?;
        if let (Some(curve_domain), Some(pcurve_domain)) =
            (curve_domain(curve), pcurve_domain(pcurve))
        {
            if curve_domain != pcurve_domain {
                return Err(invalid(
                    "coedge evaluator domain",
                    "portable 3D curve and face-use pcurve must share one parameter domain",
                ));
            }
        }
    }

    for face in &topology.faces {
        work.checkpoint()?;
        let implementation = lookup(&registry.surfaces, &face.surface_evaluator_id, |record| {
            &record.id
        })
        .map(|record| &record.implementation)
        .ok_or_else(|| invalid("face surface evaluator", "inventory index is incomplete"))?;
        if surface_periodicity(implementation)
            .is_some_and(|periodic| periodic != [face.periodic_u, face.periodic_v])
        {
            return Err(invalid(
                "face periodicity",
                "topology must agree with the portable surface definition",
            ));
        }
    }

    for body in &topology.bodies {
        work.checkpoint()?;
        let implementation = lookup(
            &registry.mass_properties,
            &body.mass_properties_evaluator_id,
            |record| &record.id,
        )
        .map(|record| &record.implementation)
        .ok_or_else(|| invalid("body mass properties", "inventory index is incomplete"))?;
        if let ExactMassPropertiesImplementation::KernelValidated { properties, .. } =
            implementation
        {
            if body.is_sheet_body != (properties.volume_m3.canonical_value() == 0.0) {
                return Err(invalid(
                    "body mass properties",
                    "sheet bodies require zero volume and solid bodies require positive volume",
                ));
            }
        }
    }
    Ok(())
}

fn require_exact_inventory<'a, Record, Id: Ord + 'a>(
    field: &str,
    canonical_field: &str,
    records: &[Record],
    id: impl Fn(&Record) -> &Id,
    expected: impl ExactSizeIterator<Item = &'a Id>,
    validate: impl Fn(&Id) -> Result<(), GeometryContractError>,
    work: ValidationWork<'a>,
) -> Result<(), GeometryContractError> {
    let mut prior = None;
    for record in records {
        work.checkpoint()?;
        let current = id(record);
        validate(current)?;
        if prior.is_some_and(|value| value >= current) {
            return Err(invalid(
                canonical_field,
                "records must be strictly canonical and unique",
            ));
        }
        prior = Some(current);
    }
    let mut referenced = work.buffer(expected.len())?;
    for value in expected {
        work.checkpoint()?;
        referenced.push(value);
    }
    referenced.sort_unstable();
    referenced.dedup();
    if !records.iter().map(id).eq(referenced.iter().copied()) {
        return Err(invalid(
            field,
            "must contain exactly every topology-referenced evaluator and no extras",
        ));
    }
    Ok(())
}

fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
