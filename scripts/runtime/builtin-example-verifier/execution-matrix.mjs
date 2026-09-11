// @ts-check

import { digestObject } from "./identity.mjs";
import { isExecutionLane, laneProduct } from "./lanes.mjs";
import { validatePlan } from "./plan.mjs";
import { digest, exactKeys, integer } from "./schema.mjs";

export const EXECUTION_MATRIX_SCHEMA = "runmat.builtin-examples.execution-matrix.v1";

export function buildExecutionMatrix(plan) {
    validatePlan(plan);
    const include = plan.lanes.flatMap((lane) => lane.shards.map((shard) => ({
        lane: lane.lane,
        shardIndex: shard.index,
        assignmentDigest: shard.assignmentDigest,
        product: lane.product
    })));
    const matrix = { schema: EXECUTION_MATRIX_SCHEMA, planDigest: plan.planDigest, include, executionMatrixDigest: "" };
    matrix.executionMatrixDigest = digestObject(matrix, ["executionMatrixDigest"]);
    return matrix;
}

export function validateExecutionMatrix(matrix, plan) {
    validatePlan(plan);
    if (!matrix || matrix.schema !== EXECUTION_MATRIX_SCHEMA) throw new Error("Unsupported builtin example execution matrix schema");
    exactKeys(matrix, ["schema", "planDigest", "include", "executionMatrixDigest"], "execution matrix");
    digest(matrix.planDigest, "execution matrix plan digest");
    digest(matrix.executionMatrixDigest, "execution matrix digest");
    if (matrix.planDigest !== plan.planDigest) throw new Error("Execution matrix plan digest mismatch");
    if (!Array.isArray(matrix.include)) throw new Error("Execution matrix include must be an array");
    for (const entry of matrix.include) {
        exactKeys(entry, ["lane", "shardIndex", "assignmentDigest", "product"], "execution matrix entry");
        if (!isExecutionLane(entry.lane)) throw new Error(`Invalid execution matrix lane: ${entry.lane}`);
        if (entry.product !== laneProduct(entry.lane)) throw new Error(`Wrong product for execution matrix lane ${entry.lane}`);
        integer(entry.shardIndex, `execution matrix shard index for ${entry.lane}`);
        digest(entry.assignmentDigest, `execution matrix assignment digest for ${entry.lane}/${entry.shardIndex}`);
    }
    if (digestObject(matrix, ["executionMatrixDigest"]) !== matrix.executionMatrixDigest) throw new Error("Execution matrix digest mismatch");
    const expected = buildExecutionMatrix(plan);
    if (JSON.stringify(matrix.include) !== JSON.stringify(expected.include)) throw new Error("Execution matrix does not exactly match the frozen plan");
    return matrix;
}
