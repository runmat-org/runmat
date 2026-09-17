use runmat_types::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisHeader {
    pub name: String,
    pub predecessor: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcedureHeader {
    pub family: String,
    pub procedure: String,
    pub span: Span,
}

pub fn analysis(header: &str, span: Span) -> Result<Option<AnalysisHeader>, String> {
    let words = header.split_whitespace().collect::<Vec<_>>();
    if words.first().copied() != Some("analysis") {
        return Ok(None);
    }
    let Some(name) = words.get(1) else {
        return Err("analysis declaration requires a name".to_string());
    };
    let predecessor = match words.as_slice() {
        [_, _] => None,
        [_, _, "after", reference] => Some(
            reference
                .strip_prefix("analysis.")
                .ok_or_else(|| "`after` requires an `analysis.<name>` reference".to_string())?
                .to_string(),
        ),
        _ => return Err(
            "analysis header must be `analysis <name>` or `analysis <name> after analysis.<name>`"
                .to_string(),
        ),
    };
    Ok(Some(AnalysisHeader {
        name: (*name).to_string(),
        predecessor,
        span,
    }))
}

pub fn procedure(header: &str, span: Span) -> Option<ProcedureHeader> {
    let words = header.split_whitespace().collect::<Vec<_>>();
    match words.as_slice() {
        [family, procedure] if is_family(family) => Some(ProcedureHeader {
            family: (*family).to_string(),
            procedure: (*procedure).to_string(),
            span,
        }),
        _ => None,
    }
}

fn is_family(value: &str) -> bool {
    matches!(
        value,
        "structural"
            | "thermal"
            | "electromagnetic"
            | "acoustic"
            | "fluid"
            | "thermomechanical"
            | "electrothermal"
            | "conjugate_heat_transfer"
            | "fluid_structure_interaction"
    )
}
