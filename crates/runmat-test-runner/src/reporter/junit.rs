use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use super::location::primary_diagnostic;
use super::{RenderedReport, Reporter};
use crate::RunnerResult;

#[derive(Default)]
pub struct JunitReporter;

impl Reporter for JunitReporter {
    fn event(&mut self, _event: &TestEvent) -> RunnerResult<()> {
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
            xml.push_str(&format!(
                "  <testcase name=\"{}\"{}>",
                escape_xml(test.test_id.as_str()),
                source_attributes
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
        xml.push_str("</testsuite>\n");
        Ok(RenderedReport {
            name: "test-results.xml".into(),
            media_type: "application/junit+xml".into(),
            bytes: xml.into_bytes(),
        })
    }
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
