use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use super::location::primary_diagnostic;
use super::{RenderedReport, Reporter};
use crate::RunnerResult;

#[derive(Default)]
pub struct JunitReporter {
    process_output: super::process_output::ProcessOutputReports,
}

impl Reporter for JunitReporter {
    fn event(&mut self, event: &TestEvent) -> RunnerResult<()> {
        self.process_output.observe(event);
        Ok(())
    }

    fn finish(&mut self, result: &RunResult) -> RunnerResult<RenderedReport> {
        let failures = result.tests.iter().filter(|test| test.state.failed).count();
        let mut xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<testsuite name=\"runmat\" tests=\"{}\" failures=\"{}\">\n",
            result.tests.len(),
            failures
        );
        for test in &result.tests {
            let diagnostic = primary_diagnostic(test);
            let name = test
                .report_identity
                .as_ref()
                .map(|identity| identity.case_name.as_str())
                .unwrap_or_else(|| test.test_id.as_str());
            let classname = test
                .report_identity
                .as_ref()
                .map(|identity| identity.suite_name.as_str());
            let source_attributes = diagnostic
                .and_then(|diagnostic| diagnostic.source.as_ref())
                .map(|source| {
                    format!(
                        " file=\"{}\" line=\"{}\"",
                        escape_xml(&source.relative_path),
                        source.span.start_line
                    )
                })
                .unwrap_or_default();
            let source_attributes = if source_attributes.is_empty() {
                test.report_identity
                    .as_ref()
                    .map(|identity| format!(" file=\"{}\"", escape_xml(&identity.source_path)))
                    .unwrap_or_default()
            } else {
                source_attributes
            };
            let class_attribute = classname
                .map(|name| format!(" classname=\"{}\"", escape_xml(name)))
                .unwrap_or_default();
            xml.push_str(&format!(
                "  <testcase name=\"{}\"{}{}>",
                escape_xml(name),
                class_attribute,
                source_attributes,
            ));
            if test.state.failed {
                let message = diagnostic
                    .map(|diagnostic| diagnostic.message.as_str())
                    .unwrap_or("test failed");
                let identifier = diagnostic
                    .map(|diagnostic| diagnostic.identifier.as_str())
                    .unwrap_or("runmat:test:Failed");
                xml.push_str(&format!(
                    "<failure type=\"{}\" message=\"{}\">{}</failure>",
                    escape_xml(identifier),
                    escape_xml(message),
                    escape_xml(message)
                ));
            }
            xml.push_str("</testcase>\n");
        }
        let process_output = self.process_output.text();
        if !process_output.is_empty() {
            xml.push_str("  <system-out>");
            xml.push_str(&escape_xml(&process_output));
            xml.push_str("</system-out>\n");
        }
        xml.push_str("</testsuite>\n");
        Ok(RenderedReport {
            name: "test-results.xml".into(),
            media_type: "application/junit+xml".into(),
            bytes: xml.into_bytes(),
        })
    }
}

fn escape_xml(value: &str) -> String {
    let valid = value
        .chars()
        .map(|character| match character {
            '\u{9}'
            | '\u{a}'
            | '\u{d}'
            | '\u{20}'..='\u{d7ff}'
            | '\u{e000}'..='\u{fffd}'
            | '\u{10000}'..='\u{10ffff}' => character,
            _ => '\u{fffd}',
        })
        .collect::<String>();
    valid
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
