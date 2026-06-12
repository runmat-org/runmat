use runmat_geometry_core::{
    AssemblyNode, MaterialEvidence, MaterialEvidenceConfidence, Region, SourceGeometryKind,
};

use crate::report::{ImportDiagnostic, ImportDiagnosticSeverity};

#[derive(Debug)]
pub(crate) struct StepImportSummary {
    pub source_kind: SourceGeometryKind,
    pub assembly: Option<AssemblyNode>,
    pub material_evidence: Vec<MaterialEvidence>,
    pub regions: Vec<Region>,
    pub diagnostics: Vec<ImportDiagnostic>,
}

pub(crate) fn parse_step_summary(path: &str, text: &str) -> Result<StepImportSummary, String> {
    if !text.to_ascii_uppercase().contains("ISO-10303-21") {
        return Err("missing ISO-10303-21 STEP header".to_string());
    }

    let model_label = parse_model_label(path, text);
    let used_fallback_label = model_label == fallback_model_label(path);
    let product_labels = parse_product_labels(text);
    let materials = parse_material_evidence(text);

    let mut diagnostics = Vec::new();
    diagnostics.push(ImportDiagnostic {
        code: "CAD_METADATA_PRODUCT_COUNT".to_string(),
        severity: ImportDiagnosticSeverity::Info,
        message: format!(
            "STEP metadata resolved {} PRODUCT records",
            product_labels.len()
        ),
    });
    diagnostics.push(ImportDiagnostic {
        code: "CAD_METADATA_MATERIAL_EVIDENCE_COUNT".to_string(),
        severity: ImportDiagnosticSeverity::Info,
        message: format!(
            "STEP metadata resolved {} material evidence tokens",
            materials.len()
        ),
    });
    if used_fallback_label {
        diagnostics.push(ImportDiagnostic {
            code: "CAD_METADATA_FILE_NAME_FALLBACK".to_string(),
            severity: ImportDiagnosticSeverity::Info,
            message: "STEP FILE_NAME label missing; using path-derived assembly label".to_string(),
        });
    }
    if product_labels.is_empty() {
        diagnostics.push(ImportDiagnostic {
            code: "CAD_METADATA_PRODUCT_COUNT_FALLBACK".to_string(),
            severity: ImportDiagnosticSeverity::Info,
            message: "No explicit PRODUCT records found; generated single synthetic part"
                .to_string(),
        });
    }

    let regions = if product_labels.is_empty() {
        vec![Region {
            region_id: "region_1".to_string(),
            name: model_label.clone(),
            tag: Some("step_part".to_string()),
            cad_ownership: None,
        }]
    } else {
        product_labels
            .iter()
            .enumerate()
            .map(|(index, label)| Region {
                region_id: format!("region_{}", index + 1),
                name: label.clone(),
                tag: Some("step_part".to_string()),
                cad_ownership: None,
            })
            .collect()
    };

    let children = if product_labels.is_empty() {
        vec![AssemblyNode {
            node_id: "part_1".to_string(),
            label: model_label.clone(),
            children: Vec::new(),
        }]
    } else {
        product_labels
            .iter()
            .enumerate()
            .map(|(index, label)| AssemblyNode {
                node_id: format!("part_{}", index + 1),
                label: label.clone(),
                children: Vec::new(),
            })
            .collect()
    };

    Ok(StepImportSummary {
        source_kind: SourceGeometryKind::Cad,
        assembly: Some(AssemblyNode {
            node_id: "assembly_root".to_string(),
            label: model_label,
            children,
        }),
        material_evidence: materials,
        regions,
        diagnostics,
    })
}

fn parse_model_label(path: &str, text: &str) -> String {
    for line in text.lines() {
        let upper = line.to_ascii_uppercase();
        if upper.contains("FILE_NAME(") {
            if let Some(quoted) = first_quoted_value(line) {
                return quoted;
            }
        }
    }

    fallback_model_label(path)
}

fn fallback_model_label(path: &str) -> String {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .trim_end_matches(".step")
        .trim_end_matches(".stp")
        .to_string()
}

fn parse_product_labels(text: &str) -> Vec<String> {
    let mut labels = Vec::new();
    for line in text.lines() {
        let upper = line.to_ascii_uppercase();
        if !upper.contains("=PRODUCT(") {
            continue;
        }
        if let Some(label) = first_quoted_value(line) {
            labels.push(label);
        }
    }
    labels
}

fn parse_material_evidence(text: &str) -> Vec<MaterialEvidence> {
    let mut values = Vec::<String>::new();
    for line in text.lines() {
        let upper = line.to_ascii_uppercase();
        if !upper.contains("MATERIAL") {
            continue;
        }
        if let Some(value) = first_quoted_value(line) {
            values.push(value);
        }
    }

    values.sort();
    values.dedup();

    values
        .into_iter()
        .map(|value| MaterialEvidence {
            source_key: "STEP:MATERIAL".to_string(),
            normalized_key: "material_name".to_string(),
            value,
            confidence: MaterialEvidenceConfidence::Medium,
            unit_basis: None,
            assumptions: vec!["Imported from STEP metadata token".to_string()],
        })
        .collect()
}

fn first_quoted_value(line: &str) -> Option<String> {
    let start = line.find('\'')? + 1;
    let rest = &line[start..];
    let end = rest.find('\'')?;
    Some(rest[..end].trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::parse_step_summary;

    const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\n#11=PRODUCT('Bracket_B','',(#1));\n#20=MATERIAL_DESIGNATION('Aluminum 6061');\nENDSEC;\nEND-ISO-10303-21;\n";

    #[test]
    fn parse_step_summary_extracts_assembly_and_materials() {
        let summary = parse_step_summary("/fixtures/assy.step", SIMPLE_STEP).expect("parse");
        let assembly = summary.assembly.expect("assembly should exist");
        assert_eq!(assembly.label, "Assembly_A");
        assert_eq!(assembly.children.len(), 2);
        assert_eq!(summary.material_evidence.len(), 1);
        assert_eq!(summary.material_evidence[0].value, "Aluminum 6061");
        assert_eq!(summary.regions.len(), 2);
        let codes = summary
            .diagnostics
            .iter()
            .map(|item| item.code.as_str())
            .collect::<Vec<_>>();
        assert!(codes.contains(&"CAD_METADATA_PRODUCT_COUNT"));
        assert!(codes.contains(&"CAD_METADATA_MATERIAL_EVIDENCE_COUNT"));
    }

    #[test]
    fn parse_step_summary_rejects_missing_header() {
        let err = parse_step_summary("/fixtures/bad.step", "DATA;\n#1=PRODUCT('X','',(#1));")
            .expect_err("parse should fail");
        assert!(err.contains("ISO-10303-21"));
    }

    #[test]
    fn parse_step_summary_emits_file_name_fallback_diagnostic() {
        let payload = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#10=PRODUCT('Part_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";
        let summary = parse_step_summary("/fixtures/fallback.step", payload).expect("parse");
        let codes = summary
            .diagnostics
            .iter()
            .map(|item| item.code.as_str())
            .collect::<Vec<_>>();
        assert!(codes.contains(&"CAD_METADATA_FILE_NAME_FALLBACK"));
    }
}
