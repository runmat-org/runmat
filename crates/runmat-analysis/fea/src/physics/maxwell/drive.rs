//! Terminal-driven solid conductors for harmonic electromagnetics.
//!
//! A conductor is driven between two terminal faces, by a total current (`electric_current`
//! load or `electrical.integral_current`) with the return terminal at 0 V, or by two
//! prescribed terminal potentials. The drive is represented by an electrokinetic unit
//! potential v0 (div(sigma grad v0) = 0 in the conductor, v0 = 1 on the driven terminal and
//! 0 on the return terminal) whose field g = -grad v0 is scaled by the unknown (current
//! drive) or known (voltage drive) terminal voltage U.
//!
//! Current that leaves the conductor through a terminal lying on a grounded boundary
//! returns through that boundary. Current leaving through a terminal inside the model (a
//! gap-fed coil) continues as a discrete potential flow through the electrically
//! transparent medium, so the impressed current is solenoidal for every admissible
//! gradient test field. Without that closure the open terminals would charge like a
//! capacitor and the solution would be dominated by an unphysical gap field.
//!
//! A conductor is a connected conducting body that a terminal bounds, or one that is not
//! electrically transparent. A medium is electrically transparent at angular frequency omega
//! when its skin depth is larger than the model: omega * mu * sigma * L^2 < 1, with L the
//! model's bounding-box diagonal. Such a medium cannot carry significant induced current at
//! that frequency; terminal current closes through the non-conductor elements.
use super::{
    invalid, topology, ComplexPhasor, MaxwellDriveExcitation, MaxwellElementVector, MaxwellError,
    MaxwellTerminalDrive, MaxwellVolumeData,
};
use crate::excitation::{RealizedExcitationKind, RealizedExcitationTarget};
use crate::solve::sparse_direct;
use crate::topology::AnalysisProjection;
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, ElectricalConstraintKind, EntityDimension, ProcedureStep,
    RegionTarget,
};
use std::collections::{BTreeMap, BTreeSet};

/// Corner-node P1 data of one affine tetrahedron.
pub(super) struct CornerElement {
    pub corners: [usize; 4],
    pub gradients: [[f64; 3]; 4],
    pub volume_m3: f64,
}

pub(super) fn corner_elements(
    projection: &AnalysisProjection,
) -> Result<Vec<CornerElement>, MaxwellError> {
    projection
        .elements
        .iter()
        .map(|element| {
            let mut gradients = None;
            let mut volume = 0.0;
            projection
                .visit_element_quadrature(element, |point| {
                    gradients.get_or_insert(*point.physical_barycentric_gradients());
                    volume += point.volume_weight_m3;
                })
                .map_err(topology)?;
            Ok(CornerElement {
                corners: std::array::from_fn(|i| element.node_indices[i]),
                gradients: gradients.ok_or_else(|| topology("empty element quadrature"))?,
                volume_m3: volume,
            })
        })
        .collect()
}

/// Bounding-box diagonal of the model, the length scale of electrical transparency.
pub(super) fn model_length_m(projection: &AnalysisProjection) -> f64 {
    projection
        .bounds_m
        .iter()
        .map(|[low, high]| (high - low) * (high - low))
        .sum::<f64>()
        .sqrt()
}

pub(super) fn transparent(volume: &MaxwellVolumeData, omega: f64, length_m: f64) -> bool {
    omega
        * volume.permeability_h_per_m.canonical_value()
        * volume.conductivity_s_per_m.canonical_value()
        * length_m
        * length_m
        < 1.0
}

struct Terminal {
    label: String,
    faces: Vec<usize>,
    kind: TerminalKind,
}

enum TerminalKind {
    Current(ComplexPhasor),
    Potential(ComplexPhasor),
}

pub(super) struct DriveContext<'a> {
    pub model: &'a AnalysisModel,
    pub step: &'a ProcedureStep,
    pub projection: &'a AnalysisProjection,
    pub volumes: &'a [MaxwellVolumeData],
    pub corner: &'a [CornerElement],
    pub grounded_nodes: &'a BTreeSet<usize>,
    pub omega: f64,
    pub frequency_hz: f64,
}

pub(super) fn compile(
    context: &DriveContext<'_>,
) -> Result<Vec<MaxwellTerminalDrive>, MaxwellError> {
    let terminals = terminals(context)?;
    if terminals.is_empty() {
        return Ok(Vec::new());
    }
    let length = model_length_m(context.projection);
    // A conductor is a connected conducting body that a terminal bounds or that is not
    // electrically transparent at this frequency; every other element is the medium the
    // terminal current may close through.
    let positive: Vec<bool> = context
        .volumes
        .iter()
        .map(|volume| volume.conductivity_s_per_m.canonical_value() > 0.0)
        .collect();
    let candidates = conductor_components(context, &positive);
    let mut kept = BTreeSet::new();
    for (element, owner) in candidates.iter().enumerate() {
        if let Some(owner) = owner {
            if !transparent(&context.volumes[element], context.omega, length) {
                kept.insert(*owner);
            }
        }
    }
    for terminal in &terminals {
        for &face in &terminal.faces {
            for &element in &context.projection.boundary_faces[face].adjacent_element_indices {
                if let Some(owner) = candidates[element] {
                    kept.insert(owner);
                }
            }
        }
    }
    let conducting: Vec<bool> = candidates
        .iter()
        .map(|owner| owner.is_some_and(|owner| kept.contains(&owner)))
        .collect();
    let component = conductor_components(context, &conducting);
    // Group terminals by the conductor component they bound.
    let mut groups: BTreeMap<usize, Vec<Terminal>> = BTreeMap::new();
    for terminal in terminals {
        let mut owners = BTreeSet::new();
        for &face in &terminal.faces {
            for &element in &context.projection.boundary_faces[face].adjacent_element_indices {
                if let Some(owner) = component[element] {
                    owners.insert(owner);
                }
            }
        }
        if owners.len() != 1 {
            return Err(MaxwellError::Unsupported(format!(
                "terminal {} must bound exactly one conducting body at {} Hz",
                terminal.label, context.frequency_hz
            )));
        }
        let owner = *owners.iter().next().expect("one owner");
        groups.entry(owner).or_default().push(terminal);
    }
    let mut drives = Vec::new();
    for (owner, mut group) in groups {
        group.sort_by(|a, b| a.label.cmp(&b.label));
        let currents: Vec<_> = group
            .iter()
            .filter(|t| matches!(t.kind, TerminalKind::Current(_)))
            .collect();
        let potentials: Vec<_> = group
            .iter()
            .filter(|t| matches!(t.kind, TerminalKind::Potential(_)))
            .collect();
        let (driven, ret, excitation) = match (currents.as_slice(), potentials.as_slice()) {
            ([current], [ret]) => {
                let (TerminalKind::Current(amperes), TerminalKind::Potential(volts)) =
                    (&current.kind, &ret.kind)
                else {
                    unreachable!()
                };
                if *volts != ComplexPhasor::ZERO {
                    return Err(MaxwellError::Unsupported(format!(
                        "the return terminal {} of a current-driven conductor is its 0 V reference",
                        ret.label
                    )));
                }
                (*current, *ret, MaxwellDriveExcitation::Current { amperes: *amperes })
            }
            ([], [a, b]) => {
                let (TerminalKind::Potential(va), TerminalKind::Potential(vb)) = (&a.kind, &b.kind)
                else {
                    unreachable!()
                };
                (
                    *a,
                    *b,
                    MaxwellDriveExcitation::Voltage {
                        volts: *va - *vb,
                    },
                )
            }
            _ => {
                return Err(MaxwellError::Unsupported(format!(
                    "a driven conductor needs one current terminal and one 0 V return terminal, or two potential terminals; found {} current and {} potential terminals ({})",
                    currents.len(),
                    potentials.len(),
                    group.iter().map(|t| t.label.as_str()).collect::<Vec<_>>().join(", ")
                )))
            }
        };
        let elements: Vec<usize> = (0..component.len())
            .filter(|&e| component[e] == Some(owner))
            .collect();
        drives.push(build_drive(
            context,
            &conducting,
            &elements,
            driven,
            ret,
            excitation,
        )?);
    }
    Ok(drives)
}

fn terminals(context: &DriveContext<'_>) -> Result<Vec<Terminal>, MaxwellError> {
    let mut terminals = Vec::new();
    let sample = crate::excitation::ExcitationSamplePoint::frequency(context.frequency_hz);
    for constraint in &context.model.constraints {
        let AnalysisConstraint::Electrical(constraint) = constraint else {
            continue;
        };
        let (target, kind) = match &constraint.kind {
            ElectricalConstraintKind::PrescribedPotential { target, value } => {
                let value = crate::excitation::sample_prescribed_potential(
                    value,
                    &context.model.signals,
                    sample.clone(),
                )
                .map_err(invalid)?;
                (
                    target,
                    TerminalKind::Potential(ComplexPhasor::new(value.real, value.imaginary)),
                )
            }
            ElectricalConstraintKind::PotentialReference { target } => {
                (target, TerminalKind::Potential(ComplexPhasor::ZERO))
            }
            ElectricalConstraintKind::IntegralCurrent { target, value } => {
                let value = crate::excitation::sample_prescribed_current(
                    value,
                    &context.model.signals,
                    sample.clone(),
                )
                .map_err(invalid)?;
                (
                    target,
                    TerminalKind::Current(ComplexPhasor::new(value.real, value.imaginary)),
                )
            }
            ElectricalConstraintKind::ElectricPerfectConductor { .. } => continue,
            _ => {
                return Err(MaxwellError::Unsupported(
                    "harmonic analysis admits potential, potential-reference, integral-current and perfect-conductor electrical constraints".into(),
                ))
            }
        };
        terminals.push(Terminal {
            label: constraint.constraint_id.clone(),
            faces: faces_for(context.projection, target, &constraint.constraint_id)?,
            kind,
        });
    }
    let loads = crate::excitation::realize_step_excitations(
        context.model,
        context.step,
        crate::excitation::ExcitationSamplePoint::frequency(context.frequency_hz),
    )
    .map_err(invalid)?;
    for load in loads.iter() {
        let kind = match load.kind {
            RealizedExcitationKind::ElectricCurrent(value) => {
                TerminalKind::Current(ComplexPhasor::new(value.real, value.imaginary))
            }
            RealizedExcitationKind::TerminalVoltage(value) => {
                TerminalKind::Potential(ComplexPhasor::new(value.real, value.imaginary))
            }
            _ => continue,
        };
        let RealizedExcitationTarget::Region(target) = &load.target else {
            return Err(invalid(
                "a terminal current or voltage requires a face region",
            ));
        };
        terminals.push(Terminal {
            label: load.excitation_id.0.clone(),
            faces: faces_for(context.projection, target, &load.excitation_id.0)?,
            kind,
        });
    }
    Ok(terminals)
}

fn faces_for(
    projection: &AnalysisProjection,
    target: &RegionTarget,
    label: &str,
) -> Result<Vec<usize>, MaxwellError> {
    target.validate_resolution().map_err(invalid)?;
    if target.entity_dimension != EntityDimension::Face {
        return Err(invalid(format!("terminal {label} must select faces")));
    }
    let faces: Vec<usize> = projection
        .boundary_faces
        .iter()
        .enumerate()
        .filter(|(_, face)| {
            face.provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
        })
        .map(|(index, _)| index)
        .collect();
    if faces.is_empty() {
        return Err(invalid(format!("terminal {label} matches no mesh face")));
    }
    Ok(faces)
}

fn conductor_components(context: &DriveContext<'_>, conducting: &[bool]) -> Vec<Option<usize>> {
    let count = context.corner.len();
    let mut parent: Vec<usize> = (0..count).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut owner_by_node: BTreeMap<usize, usize> = BTreeMap::new();
    for (element, data) in context.corner.iter().enumerate() {
        if !conducting[element] {
            continue;
        }
        for node in data.corners {
            match owner_by_node.get(&node) {
                Some(&other) => {
                    let (a, b) = (find(&mut parent, element), find(&mut parent, other));
                    if a != b {
                        parent[a.max(b)] = a.min(b);
                    }
                }
                None => {
                    owner_by_node.insert(node, element);
                }
            }
        }
    }
    (0..count)
        .map(|e| conducting[e].then(|| find(&mut parent, e)))
        .collect()
}

fn terminal_nodes(context: &DriveContext<'_>, terminal: &Terminal) -> BTreeSet<usize> {
    terminal
        .faces
        .iter()
        .flat_map(|&face| {
            context.projection.boundary_faces[face]
                .node_indices
                .iter()
                .take(3)
                .copied()
        })
        .collect()
}

fn build_drive(
    context: &DriveContext<'_>,
    conducting: &[bool],
    elements: &[usize],
    driven: &Terminal,
    ret: &Terminal,
    excitation: MaxwellDriveExcitation,
) -> Result<MaxwellTerminalDrive, MaxwellError> {
    let high = terminal_nodes(context, driven);
    let low = terminal_nodes(context, ret);
    if !high.is_disjoint(&low) {
        return Err(MaxwellError::Unsupported(format!(
            "terminals {} and {} touch; a driven conductor needs separated terminals",
            driven.label, ret.label
        )));
    }
    // Electrokinetic unit potential on the conductor's corner nodes.
    let nodes: BTreeSet<usize> = elements
        .iter()
        .flat_map(|&e| context.corner[e].corners)
        .collect();
    let prescribed = |node: usize| {
        if high.contains(&node) {
            Some(1.0)
        } else if low.contains(&node) {
            Some(0.0)
        } else {
            None
        }
    };
    let free: BTreeMap<usize, usize> = nodes
        .iter()
        .copied()
        .filter(|&n| prescribed(n).is_none())
        .enumerate()
        .map(|(row, node)| (node, row))
        .collect();
    let mut entries = Vec::new();
    let mut rhs = vec![0.0; free.len()];
    for &e in elements {
        let data = &context.corner[e];
        let sigma = context.volumes[e].conductivity_s_per_m.canonical_value();
        for i in 0..4 {
            let Some(&row) = free.get(&data.corners[i]) else {
                continue;
            };
            for j in 0..4 {
                let k = sigma * data.volume_m3 * dot(data.gradients[i], data.gradients[j]);
                match free.get(&data.corners[j]) {
                    Some(&column) => entries.push((row, column, k)),
                    None => rhs[row] -= k * prescribed(data.corners[j]).unwrap_or(0.0),
                }
            }
        }
    }
    let solved = sparse_direct::solve_real(free.len(), &entries, &rhs).map_err(invalid)?;
    if solved.relative_residual > 1e-9 {
        return Err(MaxwellError::NotConverged(format!(
            "terminal unit potential residual {:e}",
            solved.relative_residual
        )));
    }
    let v0 =
        |node: usize| -> f64 { prescribed(node).unwrap_or_else(|| solved.solution[free[&node]]) };
    let mut generator = Vec::with_capacity(elements.len());
    let mut conductance = 0.0;
    // reaction[n] = integral over the conductor of sigma grad v0 . grad lambda_n
    let mut reaction: BTreeMap<usize, f64> = BTreeMap::new();
    for &e in elements {
        let data = &context.corner[e];
        let sigma = context.volumes[e].conductivity_s_per_m.canonical_value();
        let mut gradient = [0.0; 3];
        for i in 0..4 {
            let value = v0(data.corners[i]);
            for axis in 0..3 {
                gradient[axis] += value * data.gradients[i][axis];
            }
        }
        conductance += sigma * data.volume_m3 * dot(gradient, gradient);
        for i in 0..4 {
            if prescribed(data.corners[i]).is_some() {
                *reaction.entry(data.corners[i]).or_default() +=
                    sigma * data.volume_m3 * dot(gradient, data.gradients[i]);
            }
        }
        generator.push(MaxwellElementVector {
            element_identity: context.projection.elements[e].stable_identity,
            vector: gradient.map(|g| -g),
        });
    }
    if !conductance.is_finite() || conductance <= 0.0 {
        return Err(invalid(format!(
            "conductor between {} and {} has no conductance",
            driven.label, ret.label
        )));
    }
    let closure = closure_current(context, conducting, &reaction, &driven.label)?;
    Ok(MaxwellTerminalDrive {
        drive_id: driven.label.clone(),
        excitation,
        conductance_s: conductance,
        unit_generator_field: generator,
        unit_closure_current: closure,
    })
}

/// Discrete potential flow in the transparent medium carrying the terminal current
/// between ungrounded terminals: solves L phi = -r on transparent corner nodes, with
/// phi = 0 on grounded nodes (current may return through a grounded boundary).
fn closure_current(
    context: &DriveContext<'_>,
    conducting: &[bool],
    reaction: &BTreeMap<usize, f64>,
    label: &str,
) -> Result<Vec<MaxwellElementVector>, MaxwellError> {
    let scale = reaction.values().fold(0.0_f64, |m, v| m.max(v.abs()));
    let open: BTreeMap<usize, f64> = reaction
        .iter()
        .filter(|(node, value)| {
            !context.grounded_nodes.contains(node) && value.abs() > 1e-12 * scale
        })
        .map(|(&node, &value)| (node, value))
        .collect();
    if open.is_empty() {
        return Ok(Vec::new());
    }
    let medium: Vec<usize> = (0..conducting.len()).filter(|&e| !conducting[e]).collect();
    let medium_nodes: BTreeSet<usize> = medium
        .iter()
        .flat_map(|&e| context.corner[e].corners)
        .collect();
    if let Some((&node, _)) = open.iter().find(|(node, _)| !medium_nodes.contains(node)) {
        return Err(MaxwellError::Unsupported(format!(
            "terminal {label} carries current through a boundary that is neither grounded nor in contact with an electrically transparent medium (node {node}); ground that boundary or model the medium the current returns through"
        )));
    }
    let grounded_medium: BTreeSet<usize> = medium_nodes
        .intersection(context.grounded_nodes)
        .copied()
        .collect();
    // Pure Neumann (no grounded node): pin the smallest node; the data sum to zero.
    let pinned: BTreeSet<usize> = if grounded_medium.is_empty() {
        let total: f64 = open.values().sum();
        if total.abs() > 1e-9 * scale {
            return Err(invalid(format!(
                "terminal {label} current does not balance through the medium"
            )));
        }
        medium_nodes.iter().next().copied().into_iter().collect()
    } else {
        grounded_medium
    };
    let free: BTreeMap<usize, usize> = medium_nodes
        .iter()
        .copied()
        .filter(|n| !pinned.contains(n))
        .enumerate()
        .map(|(row, node)| (node, row))
        .collect();
    let mut entries = Vec::new();
    for &e in &medium {
        let data = &context.corner[e];
        for i in 0..4 {
            let Some(&row) = free.get(&data.corners[i]) else {
                continue;
            };
            for j in 0..4 {
                if let Some(&column) = free.get(&data.corners[j]) {
                    entries.push((
                        row,
                        column,
                        data.volume_m3 * dot(data.gradients[i], data.gradients[j]),
                    ));
                }
            }
        }
    }
    let mut rhs = vec![0.0; free.len()];
    for (node, value) in &open {
        if let Some(&row) = free.get(node) {
            rhs[row] = -value;
        }
    }
    let solved = sparse_direct::solve_real(free.len(), &entries, &rhs).map_err(invalid)?;
    if solved.relative_residual > 1e-9 {
        return Err(MaxwellError::NotConverged(format!(
            "terminal closure potential residual {:e}",
            solved.relative_residual
        )));
    }
    let phi = |node: usize| free.get(&node).map_or(0.0, |&row| solved.solution[row]);
    Ok(medium
        .iter()
        .map(|&e| {
            let data = &context.corner[e];
            let mut gradient = [0.0; 3];
            for i in 0..4 {
                let value = phi(data.corners[i]);
                for axis in 0..3 {
                    gradient[axis] += value * data.gradients[i][axis];
                }
            }
            MaxwellElementVector {
                element_identity: context.projection.elements[e].stable_identity,
                vector: gradient.map(|g| -g),
            }
        })
        .filter(|value| value.vector != [0.0; 3])
        .collect())
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
