//! Authored field intent: no mesh digest, entity count or runtime data is inferred.
use super::*;
use runmat_analysis_core::{AuthoredSignalFieldLocation, AuthoredSignalTopologySelector};

pub(super) fn shape(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(SignalShape, Option<runmat_analysis_core::CoordinateFrame>)> {
    let before = diagnostics.len();
    let mut seen = BTreeMap::new();
    for statement in body {
        match &statement.kind {
            CheckedFeaStatementKind::GeneratedByFunction { .. } => {}
            CheckedFeaStatementKind::Assignment { target, .. }
                if target.len() == 1
                    && ["frame", "topology", "location", "components"]
                        .contains(&target[0].as_str()) =>
            {
                if seen.insert(target[0].as_str(), statement.span).is_some() {
                    error(
                        declaration,
                        statement.span,
                        "duplicate generated-signal shape member",
                        diagnostics,
                    );
                }
            }
            _ => error(
                declaration,
                statement.span,
                "unsupported generated-signal outer member",
                diagnostics,
            ),
        }
    }
    if diagnostics.len() != before {
        return None;
    }
    let topology = assignment(body, "topology");
    let location = assignment(body, "location");
    let components = assignment(body, "components");
    if topology.is_none() && location.is_none() && components.is_none() {
        let vector = seen.contains_key("frame");
        return Some((
            if vector {
                SignalShape::Vector3
            } else {
                SignalShape::Scalar
            },
            optional_frame(declaration, body, vector, diagnostics)?,
        ));
    }
    let (Some(topology), Some(location), Some(components)) = (topology, location, components)
    else {
        error(
            declaration,
            declaration.span,
            "generated topology field requires topology, location and components",
            diagnostics,
        );
        return None;
    };
    if !matches!(&topology.kind, CheckedExpressionKind::Enum(name) if name == "study_solver_mesh") {
        error(
            declaration,
            topology.span,
            "generated field topology must be study_solver_mesh",
            diagnostics,
        );
        return None;
    }
    let location = match &location.kind {
        CheckedExpressionKind::Enum(name) => match name.as_str() {
            "node" => Some(AuthoredSignalFieldLocation::Node),
            "cell" => Some(AuthoredSignalFieldLocation::Cell),
            "element" => Some(AuthoredSignalFieldLocation::Element),
            "face" => Some(AuthoredSignalFieldLocation::Face),
            "edge" => Some(AuthoredSignalFieldLocation::Edge),
            _ => None,
        },
        _ => None,
    }
    .or_else(|| {
        error(
            declaration,
            location.span,
            "generated field location must be node, cell, element, face or edge",
            diagnostics,
        );
        None
    })?;
    let width = match &components.kind {
        CheckedExpressionKind::Number(number) => number
            .parse::<usize>()
            .ok()
            .and_then(runmat_types::wire_integer::bounded)
            .filter(|width| *width > 0),
        _ => None,
    }
    .or_else(|| {
        error(
            declaration,
            components.span,
            "generated field components must be a positive integer literal",
            diagnostics,
        );
        None
    })?;
    Some((
        SignalShape::DeferredTopologyField {
            selector: AuthoredSignalTopologySelector::StudySolverMesh { location },
            component_count: width,
        },
        optional_frame(declaration, body, width != 1, diagnostics)?,
    ))
}
