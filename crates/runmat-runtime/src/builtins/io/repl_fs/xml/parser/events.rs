//! Well-formed XML events projected into the existing lightweight node model.

use super::*;
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

use super::reader::CooperativeXmlBuf;
use super::{attributes, decode};

pub(in crate::builtins::io::repl_fs::xml) async fn parse_xml_document_with_context(
    text: &str,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<XmlNode> {
    parse_events(text, cooperation).await.map_err(|error| {
        if cooperation.is_execution_error(&error) {
            return error;
        }
        let operation = cooperation.operation();
        let message = error
            .message()
            .replace("xmlread:", &format!("{}:", operation.name()));
        build_runtime_error(message)
            .with_identifier(operation.identifier(XmlDiagnosticKind::MalformedXml))
            .with_builtin(operation.name())
            .with_source(error)
            .build()
    })
}

async fn parse_events(text: &str, cooperation: &mut XmlCooperation) -> BuiltinResult<XmlNode> {
    let source = CooperativeXmlBuf::new(text.as_bytes(), cooperation.budget_owner());
    let mut reader = Reader::from_reader(source);
    reader.config_mut().check_end_names = true;
    reader.config_mut().allow_unmatched_ends = false;
    reader.config_mut().check_comments = true;
    reader.config_mut().trim_text(false);
    let mut event_buf = Vec::new();
    // The reader may grow this for a single large event; do not eagerly retain
    // another full-document allocation before it has seen any XML.
    event_buf
        .try_reserve_exact(text.len().min(4096))
        .map_err(|error| {
            build_runtime_error("xmlread: cannot reserve XML event storage")
                .with_builtin("xmlread")
                .with_source(error)
                .build()
        })?;
    let mut stack = Vec::<XmlNode>::new();
    let mut root = None;
    let mut declaration_seen = false;
    let mut declaration_position_open = true;
    loop {
        let event = match reader.read_event_into_async(&mut event_buf).await {
            Ok(event) => event,
            Err(error) => {
                return Err(match reader.get_mut().take_execution_error() {
                    Some(execution) => cooperation.execution_error(execution),
                    None => quick_xml_error(error, cooperation),
                });
            }
        };
        match event {
            Event::Start(start) => {
                declaration_position_open = false;
                if stack.is_empty() && root.is_some() {
                    return Err(malformed("multiple root elements"));
                }
                stack.push(node_from_start(&start, cooperation).await?);
            }
            Event::Empty(start) => {
                declaration_position_open = false;
                if stack.is_empty() && root.is_some() {
                    return Err(malformed("multiple root elements"));
                }
                let node = node_from_start(&start, cooperation).await?;
                attach(&mut stack, &mut root, node)?;
            }
            Event::End(_) => {
                declaration_position_open = false;
                let node = stack
                    .pop()
                    .ok_or_else(|| malformed("unexpected closing tag"))?;
                attach(&mut stack, &mut root, node)?;
            }
            Event::Text(content) => {
                declaration_position_open = false;
                append_text(&mut stack, content.as_ref(), cooperation).await?;
            }
            Event::CData(content) => {
                declaration_position_open = false;
                if stack.is_empty() {
                    return Err(malformed("CDATA outside the root element"));
                }
                append_text(&mut stack, content.as_ref(), cooperation).await?;
            }
            Event::GeneralRef(reference) => {
                declaration_position_open = false;
                let node = stack
                    .last_mut()
                    .ok_or_else(|| malformed("entity outside the root element"))?;
                decode::append_reference(node.text_run()?, reference.as_ref(), cooperation).await?;
            }
            Event::Comment(comment) => {
                declaration_position_open = false;
                if let Some(node) = stack.last_mut() {
                    let mut text = String::new();
                    decode::append_text_bytes(&mut text, comment.as_ref(), cooperation).await?;
                    node.push_content(XmlContent::Comment(text))?;
                }
            }
            Event::PI(instruction) => {
                declaration_position_open = false;
                if instruction.target().eq_ignore_ascii_case(b"xml") {
                    return Err(malformed("XML processing-instruction target is reserved"));
                }
                let target = attributes::name(instruction.target(), cooperation).await?;
                if let Some(node) = stack.last_mut() {
                    let mut content = String::new();
                    decode::append_text_bytes(&mut content, instruction.content(), cooperation)
                        .await?;
                    node.push_content(XmlContent::ProcessingInstruction { target, content })?;
                }
            }
            Event::Decl(declaration) => {
                if declaration_seen || !declaration_position_open {
                    return Err(malformed("XML declaration must be first"));
                }
                declaration::parse(&declaration, cooperation).await?;
                declaration_seen = true;
                declaration_position_open = false;
            }
            Event::DocType(_) => return Err(malformed("DTD declarations are not supported")),
            Event::Eof => break,
        }
        event_buf.clear();
        cooperation.completed(0).await?;
    }
    if let Some(open) = stack.last() {
        return Err(malformed(&format!("unclosed tag <{}>", open.name)));
    }
    root.ok_or_else(|| malformed("no root element"))
}

async fn node_from_start(
    start: &BytesStart<'_>,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<XmlNode> {
    let name = attributes::name(start.name().as_ref(), cooperation).await?;
    let attributes = attributes::parse(start.attributes_raw(), cooperation).await?;
    Ok(XmlNode::new(name, attributes))
}

async fn append_text(
    stack: &mut [XmlNode],
    text: &[u8],
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<()> {
    let Some(node) = stack.last_mut() else {
        for chunk in text.chunks(4096) {
            if !chunk
                .iter()
                .all(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
            {
                return Err(malformed("text outside the root element"));
            }
            cooperation.completed(chunk.len()).await?;
        }
        return Ok(());
    };
    decode::append_text_bytes(node.text_run()?, text, cooperation).await
}

fn attach(stack: &mut [XmlNode], root: &mut Option<XmlNode>, node: XmlNode) -> BuiltinResult<()> {
    if let Some(parent) = stack.last_mut() {
        parent.push_content(XmlContent::Element(node))?;
    } else if root.replace(node).is_some() {
        return Err(malformed("multiple root elements"));
    }
    Ok(())
}

fn malformed(reason: &str) -> RuntimeError {
    compat_error("xmlread", format!("xmlread: malformed XML ({reason})"))
}

fn malformed_source(
    reason: &str,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(format!("xmlread: malformed XML ({reason})"))
        .with_builtin("xmlread")
        .with_source(source)
        .build()
}

pub(super) fn quick_xml_error(
    error: quick_xml::Error,
    cooperation: &XmlCooperation,
) -> RuntimeError {
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    while let Some(current) = cause {
        if let Some(execution) = current.downcast_ref::<crate::execution::ExecutionServiceError>() {
            return cooperation.execution_error(execution.clone());
        }
        cause = current.source();
    }
    malformed_source("invalid XML syntax", error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> BuiltinResult<XmlNode> {
        let mut cooperation = XmlCooperation::new(None, XmlOperation::XmlRead);
        futures::executor::block_on(parse_xml_document_with_context(source, &mut cooperation))
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn quoted_gt_cdata_comments_and_entities_retain_projected_content() {
        let root = parse(
            "<?xml version=\"1.0\"?><root note=\"a>b\"><!--x>y--><item><![CDATA[a>b]]>&amp;&#65;</item></root>",
        )
        .expect("valid XML");
        assert_eq!(root.name, "root");
        assert_eq!(root.attributes.get("note").map(String::as_str), Some("a>b"));
        assert_eq!(root.content.len(), 2);
        assert!(matches!(&root.content[0], XmlContent::Comment(text) if text == "x>y"));
        assert!(matches!(&root.content[1], XmlContent::Element(child)
            if child.name == "item" && matches!(&child.content[..], [XmlContent::Text(text)] if text == "a>b&A")));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn long_numeric_reference_and_literal_newlines_preserve_xml_text() {
        let zeros = "0".repeat(12 * 1024);
        let document = format!("<root>a\r\nb&#{zeros}65;\nc</root>");
        let root = parse(&document).expect("valid large character reference");
        assert!(matches!(&root.content[..], [XmlContent::Text(text)] if text == "a\nbA\nc"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn mixed_content_has_one_exact_ordered_authority() {
        let root = parse("<p> a<b/> c<![CDATA[d]]>&amp; e<!--note--><?pi data?>z</p>")
            .expect("valid mixed content");
        assert_eq!(root.content.len(), 6);
        assert!(matches!(&root.content[0], XmlContent::Text(text) if text == " a"));
        assert!(matches!(&root.content[1], XmlContent::Element(child) if child.name == "b"));
        assert!(matches!(&root.content[2], XmlContent::Text(text) if text == " cd& e"));
        assert!(matches!(&root.content[3], XmlContent::Comment(text) if text == "note"));
        assert!(
            matches!(&root.content[4], XmlContent::ProcessingInstruction { target, content }
            if target == "pi" && content == " data")
        );
        assert!(matches!(&root.content[5], XmlContent::Text(text) if text == "z"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn literal_newlines_normalize_before_character_references() {
        let root = parse("<root>a\r\nb&#13;c</root>").expect("valid XML text");
        assert!(matches!(&root.content[..], [XmlContent::Text(text)] if text == "a\nb\rc"));

        // Markup separates the two source line endings. A local CR state
        // must not collapse them merely because the semantic text run joins.
        let root = parse("<root>a\r<![CDATA[\n]]>b</root>").expect("valid adjacent text");
        assert!(matches!(&root.content[..], [XmlContent::Text(text)] if text == "a\n\nb"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn readstruct_projection_groups_elements_and_loses_mixed_order_explicitly() {
        let root = parse("<p>before<b>inner</b>after<b>last</b></p>").expect("valid XML");
        let mut cooperation = XmlCooperation::new(None, XmlOperation::ReadStruct);
        let projected = futures::executor::block_on(xml_node_to_struct(
            &root,
            &ReadStructOptions::default(),
            &mut cooperation,
        ))
        .expect("legacy grouped projection");
        assert_eq!(
            projected.fields.get("Text"),
            Some(&char_value("beforeafter"))
        );
        let Some(Value::Cell(elements)) = projected.fields.get("b") else {
            panic!("expected grouped repeated elements");
        };
        assert_eq!(
            elements.data(),
            vec![char_value("inner"), char_value("last")]
        );
        assert!(
            matches!(&root.content[..], [XmlContent::Text(before), XmlContent::Element(_), XmlContent::Text(after), XmlContent::Element(_)] if before == "before" && after == "after")
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn readstruct_many_distinct_and_repeated_fields_keep_first_seen_order() {
        let mut document = String::from("<root>");
        for index in 0..64 {
            document.push_str(&format!("<n{index}>first{index}</n{index}>"));
        }
        for index in (0..64).rev() {
            document.push_str(&format!("<n{index}>second{index}</n{index}>"));
        }
        document.push_str("</root>");
        let root = parse(&document).expect("valid XML with many fields");
        let mut cooperation = XmlCooperation::new(None, XmlOperation::ReadStruct);
        let projected = futures::executor::block_on(xml_node_to_struct_value(
            &root,
            &ReadStructOptions::default(),
            &mut cooperation,
        ))
        .expect("grouped projection");
        let Value::Struct(projected) = projected else {
            panic!("expected structured root with child elements");
        };
        let names: Vec<_> = projected.field_names().collect();
        let expected: Vec<_> = (0..64).map(|index| format!("n{index}")).collect();
        assert_eq!(
            names,
            expected.iter().map(String::as_str).collect::<Vec<_>>()
        );
        for index in 0..64 {
            let Some(Value::Cell(values)) = projected.fields.get(&format!("n{index}")) else {
                panic!("expected repeated element cell");
            };
            assert_eq!(
                values.data(),
                vec![
                    char_value(&format!("first{index}")),
                    char_value(&format!("second{index}")),
                ]
            );
        }
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn readstruct_parent_projection_observes_service_cancellation() {
        use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::num::NonZeroU64;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mut document = String::from("<root>");
        for index in 0..128 {
            document.push_str(&format!("<n{index}/>"));
        }
        document.push_str("</root>");
        let root = parse(&document).expect("valid XML with many empty children");
        let source = CancellationSource::new();
        let canceller = source.clone();
        let turns = Arc::new(AtomicUsize::new(0));
        let observed = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                if observed.fetch_add(1, Ordering::Relaxed) >= 64 {
                    canceller.cancel(CancellationReason::User);
                }
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::ReadStruct);
        let error = futures::executor::block_on(xml_node_to_struct_value(
            &root,
            &ReadStructOptions::default(),
            &mut cooperation,
        ))
        .expect_err("parent projection must observe a cancelled service turn");
        assert_eq!(error.identifier(), Some("RunMat:readstruct:Cancelled"));
        assert!(turns.load(Ordering::Relaxed) >= 65);
        assert!(cooperation.budget_owner().unwrap().borrow().is_some());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn malformed_tails_external_text_and_dtd_are_typed_parser_failures() {
        for source in [
            "<root attr/>",
            "<root a=\"x\" a=\"y\"/>",
            "outside<root/>",
            "<root/>outside",
            "<root/><other/>",
            "<!DOCTYPE root [<!ENTITY name 'text'>]><root>&name;</root>",
            "<![CDATA[ ]]><root/>",
            "<?xml version='1.0' encoding='UTF-8' version='1.0'?><root/>",
            "<?xml encoding='UTF-8' version='1.0'?><root/>",
            "<?xml version='1.1'?><root/>",
            "<?xml version='1.0' standalone='maybe'?><root/>",
            "<?xml version='1.0' standalone='yes' encoding='UTF-8'?><root/>",
            "<!--comment--><?xml version='1.0'?><root/>",
            " <?xml version='1.0'?><root/>",
            "<?XML?><root/>",
            "<?xml version='1.0' encoding='&#85;TF-8'?><root/>",
        ] {
            let error = parse(source).expect_err(source);
            assert_eq!(error.identifier(), Some("RunMat:xmlread:MalformedXml"));
        }
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn long_single_text_event_yields_and_cancels_before_projection() {
        use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::num::NonZeroU64;

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
        let document = format!("<root>{}</root>", "x".repeat(12 * 1024));
        let error = futures::executor::block_on(parse_xml_document_with_context(
            &document,
            &mut cooperation,
        ))
        .expect_err("reader refill must observe the cancelled service turn");
        assert_eq!(error.identifier(), Some("RunMat:xmlread:Cancelled"));
        assert!(cooperation.budget_owner().unwrap().borrow().is_some());
    }
}
