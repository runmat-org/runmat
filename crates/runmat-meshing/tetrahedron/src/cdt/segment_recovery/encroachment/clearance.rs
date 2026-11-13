//! Off-centre rounding is a search hint, not geometric admission. Both child
//! diametral spheres must clear the original witness under the same exact
//! materialized-point predicate used by physical segment recovery.
use super::*;
use crate::cdt::DelaunayVolumeNode;

pub(super) fn choose(
    endpoints: [DelaunayVolumeNode; 2],
    witness: DelaunayVolumeNode,
    context: SegmentContext,
    interval: [DyadicNode; 2],
    preferred: [f64; 3],
    radial_origin: Option<[f64; 3]>,
    work: &mut RecoveryWork<'_>,
) -> Result<DyadicNode, DelaunaySegmentRecoveryError> {
    let [left, right] = interval;
    for parameter in preferred {
        if !parameter.is_finite()
            || parameter <= left.parameter()?
            || parameter >= right.parameter()?
        {
            continue;
        }
        let mut admit = |candidate| {
            if !admits(endpoints, witness, context, candidate, work)? {
                return Ok(false);
            }
            admits_radius(witness, context, candidate, radial_origin, work)
        };
        let split = left.closest_parameter_with_clearance(
            right,
            parameter,
            radial_origin.is_some(),
            &mut admit,
        )?;
        if let Some(split) = split {
            return Ok(split);
        }
    }
    Err(error(
        DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
        Some(context.constraint_index),
        format!(
            "no representable off-centre split clears both child diametral spheres: segment \
             {:?}-{:?}, interval {:?}-{:?}, witness {:?} ({:?}), hints {preferred:?}, radial \
             origin {radial_origin:?}",
            context.first_coordinates,
            context.last_coordinates,
            endpoints[0].coordinates_m.canonical_value(),
            endpoints[1].coordinates_m.canonical_value(),
            witness.coordinates_m.canonical_value(),
            witness.construction,
        ),
    ))
}

fn admits_radius(
    witness: DelaunayVolumeNode,
    context: SegmentContext,
    split: DyadicNode,
    origin: Option<[f64; 3]>,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let Some(origin) = origin else {
        return Ok(true);
    };
    work.search_encroachment_step(context.constraint_index)?;
    let point = interpolate(
        context.first_coordinates,
        context.last_coordinates,
        split.parameter()?,
    );
    let explicit = |identity, coordinates| {
        DelaunayVolumeNode::explicit(identity, coordinates).predicate_point()
    };
    let origin_identity = if origin == context.first_coordinates {
        context.first_identity
    } else {
        context.last_identity
    };
    runmat_meshing_core::quality::predicate::compare_radial_distances_constructed(
        explicit(origin_identity, origin),
        explicit(witness.identity, witness.coordinates_m.canonical_value()),
        explicit(super::super::steiner_identity(context, split), point),
    )
    .map(|sign| sign != PredicateSign::Negative)
    .map_err(|failure| {
        invalid_topology(
            context.constraint_index,
            format!("acute radius predicate failed: {failure:?}"),
        )
    })
}

fn admits(
    endpoints: [DelaunayVolumeNode; 2],
    witness: DelaunayVolumeNode,
    context: SegmentContext,
    split: DyadicNode,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let candidate = DelaunayVolumeNode::explicit(
        super::super::steiner_identity(context, split),
        interpolate(
            context.first_coordinates,
            context.last_coordinates,
            split.parameter()?,
        ),
    );
    let original = endpoints.map(|node| {
        DelaunayVolumeNode::explicit(node.identity, node.coordinates_m.canonical_value())
    });
    let witness =
        DelaunayVolumeNode::explicit(witness.identity, witness.coordinates_m.canonical_value());
    for child in [[original[0], candidate], [candidate, original[1]]] {
        work.search_encroachment_step(context.constraint_index)?;
        let relation = diametral_sphere_position_constructed(
            child.map(DelaunayVolumeNode::predicate_point),
            witness.predicate_point(),
        )
        .map_err(|failure| {
            invalid_topology(
                context.constraint_index,
                format!("child-sphere clearance predicate failed: {failure:?}"),
            )
        })?;
        if relation == PredicateSign::Negative {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests;
