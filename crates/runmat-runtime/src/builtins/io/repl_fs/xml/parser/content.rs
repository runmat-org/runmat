//! One ordered semantic content store for an XML element. Public lightweight
//! ObjectInstance fields are projections, not an alternate parser authority.

use super::*;

#[derive(Clone, Debug)]
pub(super) enum XmlContent {
    Text(String),
    Element(XmlNode),
    Comment(String),
    ProcessingInstruction { target: String, content: String },
}

impl XmlNode {
    pub(super) fn push_content(&mut self, item: XmlContent) -> BuiltinResult<()> {
        self.content.try_reserve(1).map_err(|error| {
            build_runtime_error("xmlread: cannot reserve ordered XML content")
                .with_builtin("xmlread")
                .with_source(error)
                .build()
        })?;
        self.content.push(item);
        Ok(())
    }

    pub(super) fn text_run(&mut self) -> BuiltinResult<&mut String> {
        if !matches!(self.content.last(), Some(XmlContent::Text(_))) {
            self.push_content(XmlContent::Text(String::new()))?;
        }
        match self.content.last_mut() {
            Some(XmlContent::Text(text)) => Ok(text),
            _ => unreachable!("the new content item is a text run"),
        }
    }

    /// The legacy public TextContent/readstruct view contains direct text,
    /// but the ordered content remains the sole internal source of that view.
    pub(super) async fn direct_text(
        &self,
        cooperation: &mut XmlCooperation,
    ) -> BuiltinResult<String> {
        let mut out = String::new();
        for item in &self.content {
            let text = match item {
                XmlContent::Text(text) => text,
                // The legacy text view discards these ordered nodes. Their
                // payloads still belong to this source and are observed at
                // the projection checkpoint; no second text cache is kept.
                XmlContent::Comment(comment) => {
                    cooperation.completed(comment.len()).await?;
                    continue;
                }
                XmlContent::ProcessingInstruction { target, content } => {
                    cooperation.completed(target.len() + content.len()).await?;
                    continue;
                }
                XmlContent::Element(_) => continue,
            };
            let mut position = 0;
            while position < text.len() {
                let mut end = position.saturating_add(4096).min(text.len());
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                out.try_reserve(end - position).map_err(|error| {
                    build_runtime_error("xmlread: cannot reserve XML text projection")
                        .with_builtin("xmlread")
                        .with_source(error)
                        .build()
                })?;
                out.push_str(&text[position..end]);
                cooperation.completed(end - position).await?;
                position = end;
            }
        }
        Ok(out)
    }

    pub(super) fn element_count(&self) -> usize {
        self.content
            .iter()
            .filter(|item| matches!(item, XmlContent::Element(_)))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::num::NonZeroU64;

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn large_ordered_text_projection_returns_budget_on_cancellation() {
        let mut node = XmlNode::new("root".into(), BTreeMap::new());
        node.push_content(XmlContent::Text("x".repeat(16 * 1024)))
            .unwrap();
        node.push_content(XmlContent::Comment("comment".into()))
            .unwrap();
        node.push_content(XmlContent::Text("tail".into())).unwrap();
        let source = CancellationSource::new();
        let canceller = source.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                canceller.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlRead);
        let error = futures::executor::block_on(node.direct_text(&mut cooperation))
            .expect_err("projection must yield and observe cancellation");
        assert_eq!(error.identifier(), Some("RunMat:xmlread:Cancelled"));
        assert!(cooperation.budget_owner().unwrap().borrow().is_some());
    }
}
