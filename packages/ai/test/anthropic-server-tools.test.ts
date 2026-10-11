import { describe, expect, it, vi } from "vitest";
import { stream as streamAnthropic } from "../src/api/anthropic-messages.ts";
import { normalizeContext } from "../src/compat.ts";
import type { Context, Model } from "../src/types.ts";

vi.mock("@anthropic-ai/sdk", () => {
	function createSseResponse(): Response {
		const events = [
			{
				type: "message_start",
				message: { id: "msg_test", usage: { input_tokens: 1, output_tokens: 0 } },
			},
			{
				type: "content_block_start",
				index: 0,
				content_block: {
					type: "server_tool_use",
					id: "srv_1",
					name: "tool_search_tool_regex",
					input: { pattern: "read" },
				},
			},
			{ type: "content_block_stop", index: 0 },
			{
				type: "content_block_start",
				index: 1,
				content_block: {
					type: "tool_search_tool_result",
					tool_use_id: "srv_1",
					content: {
						type: "tool_search_tool_search_result",
						tool_references: [{ type: "tool_reference", tool_name: "read" }],
					},
				},
			},
			{ type: "content_block_stop", index: 1 },
			{
				type: "message_delta",
				delta: { stop_reason: "end_turn" },
				usage: { output_tokens: 1 },
			},
			{ type: "message_stop" },
		];
		const body = events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n`).join("\n");
		return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
	}

	class FakeAnthropic {
		beta = { messages: { create: () => ({ asResponse: async () => createSseResponse() }) } };
	}

	return { default: FakeAnthropic };
});

const model: Model<"anthropic-messages"> = {
	id: "claude-test",
	name: "Claude Test",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 100000,
	maxTokens: 4096,
};

const context: Context = {
	messages: [{ role: "user", content: "Find the read tool", timestamp: 1 }],
};

describe("Anthropic server tools", () => {
	it("normalizes server tool calls and tool-search results", async () => {
		const result = await streamAnthropic(model, normalizeContext(context), { apiKey: "test" }).result();

		expect(result.content).toEqual([
			{
				type: "serverToolCall",
				id: "srv_1",
				name: "tool_search_tool_regex",
				arguments: { pattern: "read" },
			},
			{
				type: "toolSearchResult",
				toolUseId: "srv_1",
				content: [{ type: "tool_reference", tool_name: "read" }],
			},
		]);
	});
});
