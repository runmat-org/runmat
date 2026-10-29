import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { readJson, copyInput } from "./documents.mjs";
export function collectFea(source, output, register, examples, evidence, validate) {
    if (register.schema_version !== 1 || register.posture !== "discovery_only" || !Array.isArray(register.requirements))
        throw new Error("Unsupported requirement export");
    const requirements = new Map(register.requirements.map(requirement => [requirement.id, requirement]));
    if (requirements.size !== register.requirements.length)
        throw new Error("Duplicate requirement IDs");
    const manual = readJson(join(source, "docs/simulation/manual.json"));
    validate("fea-manual", manual);
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
    // The pending-feature registry drives the site's "Not yet in RunMat" flags and its release-status page.
    const pending = readJson(join(source, "docs/simulation/pending-features.json"));
    validate("pending-features", pending);
    mkdirSync(join(output, "fea"), { recursive: true });
    for (const [name, value] of [["requirements", register], ["manual", manual], ["evidence-index", evidence], ["pending-features", pending]])
        writeFileSync(join(output, `fea/${name}.json`), JSON.stringify(value, null, 2) + "\n");
}
