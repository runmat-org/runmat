// @ts-check

export function renderReconciliationMarkdown(report) {
    const rows = Object.entries(report.summary.byStatus).map(([status, count]) => `| ${status} | ${count} |`).join("\n");
    return `# Builtin example verification\n\nStatus: **${report.status}**\n\n- Product scope: \`${report.productScope}\`\n- Source revision: \`${report.sourceRevision}\`\n- Inventory: \`${report.inventoryDigest}\`\n- Plan: \`${report.planDigest}\`\n- Shards: ${report.summary.shards}\n- Execution units: ${report.summary.executionUnits}\n\n| Result | Count |\n| --- | ---: |\n${rows}\n`;
}

export function renderReconciliationHtml(report) {
    const rows = Object.entries(report.summary.byStatus)
        .map(([status, count]) => `<tr><td>${escapeHtml(status)}</td><td>${count}</td></tr>`).join("");
    return `<!doctype html><html><head><meta charset="utf-8"><title>Builtin example verification</title></head><body><main><h1>Builtin example verification</h1><p>Status: <strong>${escapeHtml(report.status)}</strong></p><dl><dt>Product scope</dt><dd><code>${escapeHtml(report.productScope)}</code></dd><dt>Source revision</dt><dd><code>${escapeHtml(report.sourceRevision)}</code></dd><dt>Inventory</dt><dd><code>${escapeHtml(report.inventoryDigest)}</code></dd><dt>Plan</dt><dd><code>${escapeHtml(report.planDigest)}</code></dd><dt>Shards</dt><dd>${report.summary.shards}</dd><dt>Execution units</dt><dd>${report.summary.executionUnits}</dd></dl><table><thead><tr><th>Result</th><th>Count</th></tr></thead><tbody>${rows}</tbody></table></main></body></html>\n`;
}

function escapeHtml(value) {
    return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}
