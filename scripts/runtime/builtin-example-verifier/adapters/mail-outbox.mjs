import { mkdirSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

export function prepareMailOutbox(caseDirectory, fixture) {
    if (!fixture || typeof fixture !== "object" || !fixture.MailOutbox) {
        return { environment: {}, verify: (outcome) => outcome };
    }
    const outbox = join(caseDirectory, "mail-outbox");
    mkdirSync(outbox);
    return {
        environment: { RUNMAT_SENDMAIL_OUTBOX: outbox },
        verify(outcome) {
            if (outcome.errorText) return outcome;
            const entries = readdirSync(outbox);
            if (entries.length !== 1 || !/^runmat-mail-[0-9]+\.eml$/.test(entries[0])) {
                return { ...outcome, errorText: "mail outbox fixture expected exactly one captured message" };
            }
            const actual = readFileSync(join(outbox, entries[0]), "utf8");
            if (actual !== fixture.MailOutbox.expected_message) {
                return { ...outcome, errorText: "mail outbox fixture captured different message bytes" };
            }
            return outcome;
        }
    };
}
