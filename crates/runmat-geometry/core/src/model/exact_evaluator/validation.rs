use std::collections::{BTreeMap, BTreeSet};

use super::super::{ExactBRepModel, ExactBRepTopology, GeometryContractError};
use super::{
    definition_validation::{
        curve_domain, curve_dynamic_value_count, curve_is_periodic, pcurve_domain,
        pcurve_dynamic_value_count, surface_dynamic_value_count, surface_periodicity,
        validate_curve, validate_mass_properties, validate_pcurve, validate_surface,
        validate_token, validate_trim_classifier,
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
        topology.validate_against(model)?;
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

        let curves = collect_records(
            "curve evaluators",
            &self.curves,
            |record| &record.id,
            CurveEvaluatorId::validate,
        )?;
        let pcurves = collect_records(
            "pcurve evaluators",
            &self.pcurves,
            |record| &record.id,
            PcurveEvaluatorId::validate,
        )?;
        let surfaces = collect_records(
            "surface evaluators",
            &self.surfaces,
            |record| &record.id,
            SurfaceEvaluatorId::validate,
        )?;
        let classifiers = collect_records(
            "trim classifiers",
            &self.trim_classifiers,
            |record| &record.id,
            TrimClassifierId::validate,
        )?;
        let mass_properties = collect_records(
            "mass-properties evaluators",
            &self.mass_properties,
            |record| &record.id,
            MassPropertiesEvaluatorId::validate,
        )?;

        require_exact_inventory(
            "curve evaluator inventory",
            curves,
            topology
                .edges
                .iter()
                .map(|edge| edge.curve_evaluator_id.clone())
                .collect(),
        )?;
        require_exact_inventory(
            "pcurve evaluator inventory",
            pcurves,
            topology
                .coedges
                .iter()
                .map(|coedge| coedge.pcurve_evaluator_id.clone())
                .collect(),
        )?;
        require_exact_inventory(
            "surface evaluator inventory",
            surfaces,
            topology
                .faces
                .iter()
                .map(|face| face.surface_evaluator_id.clone())
                .collect(),
        )?;
        require_exact_inventory(
            "trim classifier inventory",
            classifiers,
            topology
                .faces
                .iter()
                .map(|face| face.trim_classifier_id.clone())
                .collect(),
        )?;
        require_exact_inventory(
            "mass-properties evaluator inventory",
            mass_properties,
            topology
                .bodies
                .iter()
                .map(|body| body.mass_properties_evaluator_id.clone())
                .collect(),
        )?;

        for record in &self.curves {
            validate_curve(&record.implementation)?;
        }
        for record in &self.pcurves {
            validate_pcurve(&record.implementation)?;
        }
        for record in &self.surfaces {
            validate_surface(&record.implementation)?;
        }
        for record in &self.trim_classifiers {
            validate_trim_classifier(&record.implementation)?;
        }
        for record in &self.mass_properties {
            validate_mass_properties(&record.implementation)?;
        }
        self.kernel_representation_digest()?;

        validate_topology_claims(self, topology)
    }
}

fn validate_topology_claims(
    registry: &ExactEvaluatorRegistry,
    topology: &ExactBRepTopology,
) -> Result<(), GeometryContractError> {
    let curves = registry
        .curves
        .iter()
        .map(|record| (&record.id, &record.implementation))
        .collect::<BTreeMap<_, _>>();
    for edge in &topology.edges {
        let implementation = curves
            .get(&edge.curve_evaluator_id)
            .ok_or_else(|| invalid("edge curve evaluator", "inventory index is incomplete"))?;
        if curve_is_periodic(implementation).is_some_and(|periodic| periodic != edge.is_periodic) {
            return Err(invalid(
                "edge periodicity",
                "topology must agree with the portable curve definition",
            ));
        }
    }
    let pcurves = registry
        .pcurves
        .iter()
        .map(|record| (&record.id, &record.implementation))
        .collect::<BTreeMap<_, _>>();
    let edges = topology
        .edges
        .iter()
        .map(|edge| (&edge.id, edge))
        .collect::<BTreeMap<_, _>>();
    for coedge in &topology.coedges {
        let edge = edges
            .get(&coedge.edge_id)
            .ok_or_else(|| invalid("coedge edge", "topology index is incomplete"))?;
        let curve = curves
            .get(&edge.curve_evaluator_id)
            .ok_or_else(|| invalid("edge curve evaluator", "inventory index is incomplete"))?;
        let pcurve = pcurves
            .get(&coedge.pcurve_evaluator_id)
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

    let surfaces = registry
        .surfaces
        .iter()
        .map(|record| (&record.id, &record.implementation))
        .collect::<BTreeMap<_, _>>();
    for face in &topology.faces {
        let implementation = surfaces
            .get(&face.surface_evaluator_id)
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

    let mass_properties = registry
        .mass_properties
        .iter()
        .map(|record| (&record.id, &record.implementation))
        .collect::<BTreeMap<_, _>>();
    for body in &topology.bodies {
        let implementation = mass_properties
            .get(&body.mass_properties_evaluator_id)
            .ok_or_else(|| invalid("body mass properties", "inventory index is incomplete"))?;
        if let ExactMassPropertiesImplementation::KernelValidated { properties, .. } =
            implementation
        {
            if body.is_sheet_body != (properties.volume_m3 == 0.0) {
                return Err(invalid(
                    "body mass properties",
                    "sheet bodies require zero volume and solid bodies require positive volume",
                ));
            }
        }
    }
    Ok(())
}

fn collect_records<Record, Id>(
    field: &str,
    records: &[Record],
    id: impl Fn(&Record) -> &Id,
    validate: impl Fn(&Id) -> Result<(), GeometryContractError>,
) -> Result<BTreeSet<Id>, GeometryContractError>
where
    Id: Clone + Ord,
{
    let mut ids = BTreeSet::new();
    let mut previous = None;
    for record in records {
        let current = id(record);
        validate(current)?;
        if previous.is_some_and(|value| value >= current) || !ids.insert(current.clone()) {
            return Err(invalid(
                field,
                "records must be strictly canonical and unique",
            ));
        }
        previous = Some(current);
    }
    Ok(ids)
}

fn require_exact_inventory<Id: Ord>(
    field: &str,
    actual: BTreeSet<Id>,
    expected: BTreeSet<Id>,
) -> Result<(), GeometryContractError> {
    if actual != expected {
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
