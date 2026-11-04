//! The OCCT kernel as runmat-cad's test harness sees it: the same `imports` fixtures as the
//! grid kernel (named boxes), plus per-operation outcome counting.

use runmat_cad::evaluate::{AnalysisRequest, AnalysisResult, GeometryAnalyzer};
use runmat_cad::regen::fake::FakeImportBox;
use runmat_cad::regen::kernel::{
    BodyHandle, CadKernel, KernelError, KernelId, KernelOp, KernelOutput, ToolShape,
};
use runmat_geometry_io::{ImportSource, NamedBox, OcctKernel};
use std::collections::BTreeMap;
use std::io::Write;

#[derive(Debug, Default)]
pub struct KernelUnderTest {
    /// Import fixtures by source path, as the grid kernel takes them.
    pub imports: BTreeMap<String, Vec<FakeImportBox>>,
}

impl KernelUnderTest {
    pub fn new() -> Self {
        Self::default()
    }

    fn occt(&self) -> OcctKernel {
        let mut k = OcctKernel::new();
        for (path, boxes) in &self.imports {
            k.imports.insert(
                path.clone(),
                ImportSource::Boxes(
                    boxes
                        .iter()
                        .map(|b| NamedBox {
                            name: b.name.clone(),
                            min: b.min,
                            max: b.max,
                            face_names: b.face_names.clone(),
                        })
                        .collect(),
                ),
            );
        }
        k
    }
}

/// A stable name for an operation and the case of it, for failure-rate tables.
pub fn op_case(op: &KernelOp) -> String {
    match op {
        KernelOp::Tool(t) => format!(
            "tool:{}",
            match &t.shape {
                ToolShape::Prism {
                    thin,
                    draft,
                    start_bound,
                    end_bound,
                    ..
                } => {
                    let mut s = String::from("prism");
                    if !start_bound.is_blind() || !end_bound.is_blind() {
                        s.push_str("+bound");
                    }
                    if draft.is_some() {
                        s.push_str("+draft");
                    }
                    if thin.is_some() {
                        s.push_str("+thin");
                    }
                    s
                }
                ToolShape::Revolve { .. } => "revolve".into(),
                ToolShape::Sweep { .. } => "sweep".into(),
                ToolShape::Loft { .. } => "loft".into(),
                ToolShape::Holes { .. } => "holes".into(),
            }
        ),
        KernelOp::Surface(s) => format!(
            "surface:{}",
            serde_json::to_value(s)
                .ok()
                .and_then(|v| v.get("surface").and_then(|x| x.as_str().map(String::from)))
                .unwrap_or_default()
        ),
        other => other.name().to_string(),
    }
}

fn record(case: &str, outcome: &str) {
    if let Ok(path) = std::env::var("RUNMAT_K1_STATS") {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            // One write per line: tests run in parallel and append to the same file.
            let _ = f.write_all(format!("{case}\t{outcome}\n").as_bytes());
        }
    }
}

impl CadKernel for KernelUnderTest {
    fn id(&self) -> KernelId {
        self.occt().id()
    }

    fn execute(
        &self,
        op: &KernelOp,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let r = self.occt().execute(op, operands);
        let outcome = match &r {
            Ok(_) => "ok".to_string(),
            Err(KernelError::NotSupported { what }) => format!("not_supported\t{what}"),
            Err(KernelError::Failed { message }) => format!("failed\t{message}"),
            Err(KernelError::HistoryIncomplete { message }) => {
                format!("history_incomplete\t{message}")
            }
            Err(KernelError::ForeignOperand { .. }) => "foreign_operand".to_string(),
        };
        record(&op_case(op), &outcome);
        r
    }
}

impl GeometryAnalyzer for KernelUnderTest {
    fn analyze(
        &self,
        request: &AnalysisRequest,
        bodies: &[&BodyHandle],
    ) -> Result<AnalysisResult, KernelError> {
        let r = self.occt().analyze(request, bodies);
        record(
            &format!(
                "analysis:{}",
                serde_json::to_value(request)
                    .ok()
                    .and_then(|v| v.get("analysis").and_then(|x| x.as_str().map(String::from)))
                    .unwrap_or_default()
            ),
            if r.is_ok() { "ok" } else { "failed" },
        );
        r
    }
}
