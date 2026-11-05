import { copyFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { readJson, copyInput } from "./documents.mjs";
import { loadLedger, pendingView } from "./ledger.mjs";
import { COMMITTED_GRAMMAR, DESIGN_DIALECTS, committedDesignGrammar } from "./fea-grammar.mjs";
export function collectFea(source, output, register, examples, evidence, validate) {
    if (register.schema_version !== 1 || register.posture !== "discovery_only" || !Array.isArray(register.requirements))
        throw new Error("Unsupported requirement export");
    const requirements = new Map(register.requirements.map(requirement => [requirement.id, requirement]));
    if (requirements.size !== register.requirements.length)
        throw new Error("Duplicate requirement IDs");
    const manual = readJson(join(source, "docs/simulation/manual.json"));
    validate("fea-manual", manual);
    // A topic answers to the requirements of the ledger capabilities whose docs include its page.
    const { ledger } = loadLedger(source);
    validate("ledger", ledger);
    for (const topic of manual.topics) {
        const page = topic.narrative.replace(/^docs\/simulation\//, "");
        topic.requirements = [...new Set(ledger.capabilities.filter(capability => capability.docs.includes(page)).flatMap(capability => capability.requirements))];
    }
    validate("evidence-index", evidence);
    const identities = new Set();
    const routes = new Set();
    for (const topic of manual.topics) {
        if (identities.has(topic.id) || routes.has(topic.route))
            throw new Error("Duplicate manual topic/route");
        identities.add(topic.id);
        routes.add(topic.route);
        for (const id of topic.requirements)
            if (!requirements.has(id))
                throw new Error(`Unknown manual requirement: ${id}`);
        for (const id of topic.examples)
            if (!examples.some(example => example.id === id))
                throw new Error(`Unknown example: ${id}`);
        copyInput(source, output, topic.narrative, "documents/" + topic.narrative);
    }
    for (const record of evidence.records) {
        const requirement = requirements.get(record.requirement);
        const obligation = requirement?.obligations.find(obligation => obligation.id === record.obligation);
        if (!obligation?.required_envelopes.some(scope => Object.keys(scope).every(key => scope[key] === record.scope[key])))
            throw new Error("Evidence claims an undeclared scope");
    }
    // The ledger drives the site's "Not yet in RunMat" flags (fea/pending-features.json is its pending view).
    mkdirSync(join(output, "fea"), { recursive: true });
    for (const [name, value] of [["requirements", register], ["manual", manual], ["evidence-index", evidence], ["ledger", ledger], ["pending-features", pendingView(ledger)]])
        writeFileSync(join(output, `fea/${name}.json`), JSON.stringify(value, null, 2) + "\n");
    // The one `.fea` grammar (committed in runmat-design-syntax and kept current by fea-grammar.test.mjs),
    // so the site colors code exactly as the language server does. Fixture trees carry no compiler.
    if (existsSync(join(source, COMMITTED_GRAMMAR))) copyFileSync(join(source, COMMITTED_GRAMMAR), join(output, "fea/fea.tmLanguage.json"));
    // The design formats' grammars beside it (`design/part.tmLanguage.json`, …), the same copies
    // the language server and the Desktop color with.
    for (const entry of DESIGN_DIALECTS) {
        const committed = join(source, committedDesignGrammar(entry));
        if (!existsSync(committed)) continue;
        mkdirSync(join(output, "design"), { recursive: true });
        copyFileSync(committed, join(output, `design/${entry.extension}.tmLanguage.json`));
    }
}
