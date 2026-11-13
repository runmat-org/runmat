//! Interface heat balances and global energy closure evaluated from retained temperatures.
use super::{
    assembly::{boundary_operator, element_operator, AssembledEnergy, BoundaryTerm, InterfaceLaw},
    ConjugateHeatError as Error, ConjugateHeatInput,
};
use crate::algebra::spmv;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConjugateInterfaceBalance {
    pub interface_id: String,
    pub face_count: u64,
    pub area_m2: f64,
    /// Heat entering the second side (the fluid, for a fluid–solid interface), W.
    pub heat_into_second_side_w: f64,
    /// Heat entering the first side (the solid) through the same faces, W.
    pub heat_into_first_side_w: f64,
    /// |first + second| relative to the heat crossing the interface.
    pub relative_imbalance: f64,
    /// Largest first-minus-second temperature difference on the interface, K.
    pub maximum_temperature_jump_k: f64,
    /// Finite-conductance interfaces: mismatch between h ∫ (T_a − T_b) and the heat received.
    pub law_relative_residual: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConjugateEnergyBalance {
    /// Volumetric generation and applied boundary heat flux, W.
    pub applied_heat_w: f64,
    /// Heat lost through convection boundaries, h ∫ (T − T_ambient), W.
    pub convective_loss_w: f64,
    /// Net enthalpy carried out of the fluid, ρ c ∮ T u·n over its boundary faces, W.
    pub enthalpy_outflow_w: f64,
    /// Heat removed at prescribed-temperature nodes (their equation reactions), W.
    pub prescribed_temperature_heat_w: f64,
    /// Rate of heat stored in fluid and solid over a transient interval, W (zero when steady).
    pub stored_heat_rate_w: f64,
    pub relative_imbalance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConjugateHeatEvidence {
    /// ||A T − b|| / ||b|| of the constrained system.
    pub algebraic_relative_residual: f64,
    pub interfaces: Vec<ConjugateInterfaceBalance>,
    pub energy: ConjugateEnergyBalance,
}

impl ConjugateHeatEvidence {
    pub(super) fn accepted(&self, linear_tolerance: f64) -> bool {
        self.algebraic_relative_residual.is_finite()
            && self.algebraic_relative_residual <= 10.0 * linear_tolerance
            && self.energy.relative_imbalance.is_finite()
            && self.interfaces.iter().all(|i| {
                i.relative_imbalance.is_finite()
                    && i.law_relative_residual.is_finite()
                    && i.heat_into_first_side_w.is_finite()
                    && i.heat_into_second_side_w.is_finite()
            })
    }
}

pub(super) fn evaluate(
    input: &ConjugateHeatInput<'_>,
    system: &AssembledEnergy,
    values: &[f64],
    departure: Option<(&[f64], f64)>,
) -> Result<ConjugateHeatEvidence, Error> {
    let p = input.projection;
    if values.len() != system.dofs.count() || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("temperature dof values"));
    }
    // Free equations: (A + C/Δt) T − b − C T_old/Δt; prescribed dofs must hold their values.
    let action = spmv(&system.full, values).map_err(|e| Error::Operator(e.to_string()))?;
    let storage = match departure {
        Some((previous, dt)) => {
            if previous.len() != values.len() || !(dt > 0.0) {
                return Err(Error::Invalid("transient departure temperatures"));
            }
            let change = values
                .iter()
                .zip(previous)
                .map(|(a, b)| (a - b) / dt)
                .collect::<Vec<_>>();
            Some(spmv(&system.capacity, &change).map_err(|e| Error::Operator(e.to_string()))?)
        }
        None => None,
    };
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for d in 0..values.len() {
        match system.dirichlet[d] {
            Some(value) => {
                if values[d] != value {
                    return Err(Error::Invalid("prescribed temperature not held"));
                }
            }
            None => {
                let stored = storage.as_ref().map_or(0.0, |s| s[d]);
                numerator = numerator.hypot(action[d] + stored - system.rhs[d]);
                denominator = denominator
                    .hypot(system.rhs[d])
                    .hypot(stored)
                    .hypot(action[d]);
            }
        }
    }
    let algebraic_relative_residual = relative(numerator, denominator);

    // Local residuals of the unconstrained operator, without jump coupling: each element's
    // and each boundary face's own contribution to its nodal equations.
    let n = system.dofs.count();
    let mut applied = 0.0;
    let mut stored_rate = 0.0;
    let mut contributions: Vec<(usize, Vec<usize>, Vec<f64>)> = Vec::new();
    for index in 0..p.elements.len() {
        let parts = element_operator(
            p,
            index,
            &system.elements[index],
            system.velocity[index].as_ref(),
            system.generation_w_per_m3[index],
        )?;
        let dofs = p.elements[index]
            .node_indices
            .iter()
            .map(|&node| system.dofs.dof(index, node))
            .collect::<Vec<_>>();
        let mut local = vec![0.0; dofs.len()];
        for i in 0..dofs.len() {
            applied += parts.load[i];
            local[i] = -parts.load[i];
            for j in 0..dofs.len() {
                local[i] += parts.total(i, j) * values[dofs[j]];
                if let Some((previous, dt)) = departure {
                    let rate = parts.capacity[i * dofs.len() + j]
                        * (values[dofs[j]] - previous[dofs[j]])
                        / dt;
                    local[i] += rate;
                    stored_rate += rate;
                }
            }
        }
        contributions.push((index, dofs, local));
    }
    let mut convective_loss = 0.0;
    for term in &system.boundary {
        let (face, matrix, load) = boundary_operator(p, term)?;
        let f = &p.boundary_faces[face];
        let element = f.adjacent_element_indices[0];
        let dofs = f
            .node_indices
            .iter()
            .map(|&node| system.dofs.dof(element, node))
            .collect::<Vec<_>>();
        let m = dofs.len();
        let mut local = vec![0.0; m];
        for i in 0..m {
            local[i] = -load[i];
            for j in 0..m {
                local[i] += matrix[i * m + j] * values[dofs[j]];
            }
            match term {
                BoundaryTerm::HeatFlux { .. } => applied += load[i],
                BoundaryTerm::Convection { .. } => convective_loss += local[i],
            }
        }
        contributions.push((element, dofs, local));
    }

    let mut interfaces = Vec::with_capacity(system.interfaces.len());
    for plan in &system.interfaces {
        // The second side of a fluid–solid interface is the whole fluid; between two solids
        // it is every element of the second-side regions.
        let fluid_second = plan
            .faces
            .first()
            .is_some_and(|&(_, _, b)| system.velocity[b].is_some());
        let mut second_regions = plan
            .faces
            .iter()
            .map(|&(_, _, b)| &p.elements[b].region_id)
            .collect::<Vec<_>>();
        second_regions.sort();
        second_regions.dedup();
        let in_second = |e: usize| {
            if fluid_second {
                system.velocity[e].is_some()
            } else {
                second_regions
                    .binary_search(&&p.elements[e].region_id)
                    .is_ok()
            }
        };
        let mut area = 0.0;
        let mut first_mask = vec![false; n];
        let mut second_mask = vec![false; n];
        let mut jump = 0.0f64;
        let mut transferred = 0.0;
        for &(face, a, b) in &plan.faces {
            let f = &p.boundary_faces[face];
            area += f.area_m2.canonical_value();
            let da = f
                .node_indices
                .iter()
                .map(|&node| system.dofs.dof(a, node))
                .collect::<Vec<_>>();
            let db = f
                .node_indices
                .iter()
                .map(|&node| system.dofs.dof(b, node))
                .collect::<Vec<_>>();
            for (x, y) in da.iter().zip(&db) {
                jump = jump.max((values[*x] - values[*y]).abs());
                first_mask[*x] = system.dirichlet[*x].is_none();
                second_mask[*y] = system.dirichlet[*y].is_none();
            }
            if let InterfaceLaw::Conductance(h) = plan.law {
                let mass = crate::physics::thermal_interface::interface_mass(p, f)
                    .map_err(Error::InvalidDetail)?;
                let m = da.len();
                for i in 0..m {
                    for j in 0..m {
                        transferred += h * mass[i * m + j] * (values[da[j]] - values[db[j]]);
                    }
                }
            }
        }
        let mut into_first = 0.0;
        let mut into_second = 0.0;
        for (element, dofs, local) in &contributions {
            let second = in_second(*element);
            for (d, r) in dofs.iter().zip(local) {
                if second && second_mask[*d] {
                    into_second += r;
                } else if !second && first_mask[*d] {
                    into_first += r;
                }
            }
        }
        let scale = into_first.abs().max(into_second.abs());
        let law_relative_residual = match plan.law {
            InterfaceLaw::Perfect => 0.0,
            InterfaceLaw::Conductance(_) => relative(
                (transferred - into_second).abs(),
                transferred.abs().max(into_second.abs()),
            ),
        };
        interfaces.push(ConjugateInterfaceBalance {
            interface_id: plan.interface_id.clone(),
            face_count: plan.faces.len() as u64,
            area_m2: area,
            heat_into_second_side_w: into_second,
            heat_into_first_side_w: into_first,
            relative_imbalance: relative((into_first + into_second).abs(), scale),
            maximum_temperature_jump_k: match plan.law {
                InterfaceLaw::Perfect => 0.0,
                InterfaceLaw::Conductance(_) => jump,
            },
            law_relative_residual,
        });
    }

    // Independent boundary enthalpy flux: retained face flux times the face mean temperature.
    let mut enthalpy_outflow = 0.0;
    for (i, face) in input.fluid_topology.faces.iter().enumerate() {
        let q = input.face_volume_flux_m3_per_s[i];
        if face.neighbor.is_some() || q == 0.0 {
            continue;
        }
        let source = face.boundary_source.ok_or(Error::Invalid(
            "fluid exterior face without boundary source",
        ))?;
        let f = &p.boundary_faces[source.projection_boundary_index];
        let element = input.fluid_topology.cells[face.owner].source_element_index;
        let mut mean = 0.0;
        let mut weight = 0.0;
        let nodes = &f.node_indices;
        let mut failure = None;
        p.visit_boundary_quadrature(f, |point| {
            let t = point
                .shape_values()
                .iter()
                .zip(nodes)
                .map(|(s, &node)| s * values[system.dofs.dof(element, node)])
                .sum::<f64>();
            mean += t * point.area_weight_m2;
            weight += point.area_weight_m2;
        })
        .unwrap_or_else(|e| failure = Some(e));
        if let Some(e) = failure {
            return Err(Error::InvalidDetail(e.to_string()));
        }
        let rho_c = system.elements[element].volumetric_capacity_j_per_m3k;
        enthalpy_outflow += rho_c * q * mean / weight;
    }
    // The residual at a prescribed node is the heat the constraint supplies to the domain;
    // the balance books the heat leaving through prescribed temperatures, its negative.
    let reaction = -(0..n)
        .filter(|&d| system.dirichlet[d].is_some())
        .map(|d| action[d] + storage.as_ref().map_or(0.0, |s| s[d]) - system.rhs[d])
        .sum::<f64>();
    let imbalance = applied - convective_loss - enthalpy_outflow - reaction - stored_rate;
    let scale = applied.abs()
        + convective_loss.abs()
        + enthalpy_outflow.abs()
        + reaction.abs()
        + stored_rate.abs();
    Ok(ConjugateHeatEvidence {
        algebraic_relative_residual,
        interfaces,
        energy: ConjugateEnergyBalance {
            applied_heat_w: applied,
            convective_loss_w: convective_loss,
            enthalpy_outflow_w: enthalpy_outflow,
            prescribed_temperature_heat_w: reaction,
            stored_heat_rate_w: stored_rate,
            relative_imbalance: relative(imbalance.abs(), scale),
        },
    })
}

fn relative(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() {
        f64::INFINITY
    } else if denominator == 0.0 {
        if numerator == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        numerator / denominator
    }
}
