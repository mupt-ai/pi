import type { ResponseStreamEvent } from "openai/resources/responses/responses.js";
import { describe, expect, it } from "vitest";
import { processResponsesStream } from "../src/api/openai-responses-shared.ts";
import type { AssistantMessage, Model } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";

const model: Model<"openai-responses"> = {
	id: "muse-spark-1.3",
	name: "Muse Spark 1.3",
	api: "openai-responses",
	provider: "meta",
	baseUrl: "https://example.invalid",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 1048576,
	maxTokens: 131072,
};

async function* events(): AsyncIterable<ResponseStreamEvent> {
	const item = { type: "message", id: "msg_1", role: "assistant", status: "in_progress", content: [], phase: "commentary" };
	yield { type: "response.output_item.added", output_index: 0, sequence_number: 0, item } as ResponseStreamEvent;
	yield {
		type: "response.output_text.delta",
		output_index: 0,
		item_id: "msg_1",
		content_index: 0,
		sequence_number: 1,
		delta: "Looking.",
	} as ResponseStreamEvent;
}

describe("OpenAI Responses text start", () => {
	it("signs the text block with the message id and phase before text arrives", async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "pending",
			timestamp: Date.now(),
		};
		const stream = new AssistantMessageEventStream();
		const signatures: unknown[] = [];
		const read = (async () => {
			for await (const event of stream) {
				if (event.type === "text_start") {
					const block = event.partial.content[event.contentIndex];
					signatures.push(block?.type === "text" ? block.textSignature : undefined);
				}
			}
		})();
		await processResponsesStream(events(), output, stream, model).catch(() => {});
		stream.end();
		await read;
		expect(signatures).toEqual([JSON.stringify({ v: 1, id: "msg_1", phase: "commentary" })]);
	});
});
