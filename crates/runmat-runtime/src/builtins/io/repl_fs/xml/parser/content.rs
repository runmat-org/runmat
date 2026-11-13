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
            // Even empty text, child elements, comments, and instructions
            // have per-item projection work. A document can contain many of
            // them without any large payload to charge below.
            cooperation.completed(1).await?;
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

    pub(super) async fn has_element(
        &self,
        cooperation: &mut XmlCooperation,
    ) -> BuiltinResult<bool> {
        for item in &self.content {
            cooperation.completed(1).await?;
            if matches!(item, XmlContent::Element(_)) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn many_empty_items_charge_projection_work_without_a_text_payload() {
        let mut node = XmlNode::new("root".into(), BTreeMap::new());
        for _ in 0..128 {
            node.push_content(XmlContent::Comment(String::new()))
                .unwrap();
            node.push_content(XmlContent::Element(XmlNode::new(
                "empty".into(),
                BTreeMap::new(),
            )))
            .unwrap();
        }
        let turns = Arc::new(AtomicUsize::new(0));
        let observed = turns.clone();
        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                observed.fetch_add(1, Ordering::Relaxed);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlRead);
        let object = futures::executor::block_on(node.into_object(&mut cooperation, None))
            .expect("empty ordered items project to element children");
        let Value::Object(object) = object.value() else {
            panic!("projected XML element")
        };
        let Some(Value::Cell(children)) = object.property("ChildNodes") else {
            panic!("expected projected children");
        };
        assert_eq!(children.data().len(), 128);
        assert!(turns.load(Ordering::Relaxed) >= 256);
        assert!(cooperation.budget_owner().unwrap().borrow().is_some());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn readstruct_leaf_detection_yields_through_non_element_items() {
        let mut node = XmlNode::new("root".into(), BTreeMap::new());
        for _ in 0..128 {
            node.push_content(XmlContent::Comment(String::new()))
                .unwrap();
        }
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
        let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::ReadStruct);
        let error = futures::executor::block_on(node.has_element(&mut cooperation))
            .expect_err("leaf detection must yield even when there are no elements");
        assert_eq!(error.identifier(), Some("RunMat:readstruct:Cancelled"));
        assert!(cooperation.budget_owner().unwrap().borrow().is_some());
    }
}
