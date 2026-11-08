import { describe, expect, it } from "vitest";
import { streamSimple } from "../src/compat.ts";
import type { AssistantMessage, Context, Model } from "../src/types.ts";
import { convertResponsesMessages } from "../src/api/openai-responses-shared.ts";
import { transformMessages } from "../src/api/transform-messages.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

interface AnthropicPayload {
	messages: Array<{ role: string; content: string | Array<Record<string, unknown>> }>;
}

class PayloadCaptured extends Error {
	constructor() {
		super("payload captured");
		this.name = "PayloadCaptured";
	}
}

const model: Model<"anthropic-messages"> = {
	id: "claude-test",
	name: "Claude Test",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "http://127.0.0.1:9",
	reasoning: false,
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 200000,
	maxTokens: 1024,
};

const assistant: AssistantMessage = {
	role: "assistant",
	content: [{ type: "toolCall", id: "call-1", name: "read", arguments: {} }],
	api: "anthropic-messages",
	provider: "anthropic",
	model: "claude-test",
	usage: {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	},
	stopReason: "toolUse",
	timestamp: Date.now(),
};

async function capturePayload(context: Context): Promise<AnthropicPayload> {
	let capturedPayload: AnthropicPayload | undefined;
	const stream = streamSimple(model, context, {
		apiKey: "fake-key",
		cacheRetention: "none",
		onPayload: (payload) => {
			capturedPayload = payload as AnthropicPayload;
			throw new PayloadCaptured();
		},
	});
	await stream.result();
	if (!capturedPayload) throw new Error("Expected payload capture before request");
	return capturedPayload;
}

const pdf = { type: "document", source: { type: "base64", media_type: "application/pdf", data: "JVBERi0=" } };

describe("Anthropic document content", () => {
	it("sends a user PDF as a native document block", async () => {
		const payload = await capturePayload({
			messages: [
				{
					role: "user",
					content: [
						{ type: "text", text: "Summarize this" },
						{ type: "document", data: "JVBERi0=", mimeType: "application/pdf" },
					],
					timestamp: Date.now(),
				},
			],
		});
		expect(payload.messages[0].content).toEqual([{ type: "text", text: "Summarize this" }, pdf]);
	});

	it("sends a PDF inside a tool result as a native document block", async () => {
		const payload = await capturePayload({
			messages: [
				{ role: "user", content: "Read the file", timestamp: Date.now() },
				assistant,
				{
					role: "toolResult",
					toolCallId: "call-1",
					toolName: "read",
					content: [{ type: "document", data: "JVBERi0=", mimeType: "application/pdf" }],
					timestamp: Date.now(),
				},
			],
		});
		const toolResult = payload.messages[2].content as Array<Record<string, unknown>>;
		expect(toolResult[0]).toMatchObject({ type: "tool_result", tool_use_id: "call-1" });
		// Anthropic needs a text block alongside non-text blocks; the placeholder matches the image path.
		expect((toolResult[0] as { content: unknown }).content).toEqual([{ type: "text", text: "(see attached document)" }, pdf]);
	});
});

describe("document input for other APIs", () => {
	const responsesModel: Model<"openai-responses"> = { ...model, api: "openai-responses", provider: "openai", id: "gpt-test" };

	it("sends PDFs to OpenAI Responses as input_file in user messages and tool output", () => {
		const input = convertResponsesMessages(
			responsesModel,
			normalizeContext({
				messages: [
					{
						role: "user",
						content: [{ type: "document", data: "JVBERi0=", mimeType: "application/pdf" }],
						timestamp: 0,
					},
					{ ...assistant, api: "openai-responses", provider: "openai", model: "gpt-test" },
					{
						role: "toolResult",
						toolCallId: "call-1",
						toolName: "read",
						content: [
							{ type: "text", text: "Read a.pdf" },
							{ type: "document", data: "JVBERi0=", mimeType: "application/pdf" },
						],
						isError: false,
						timestamp: 0,
					},
				],
			}),
			new Set(["openai"]),
		);
		const file = { type: "input_file", filename: "document.pdf", file_data: "data:application/pdf;base64,JVBERi0=" };
		expect(input).toContainEqual({ role: "user", content: [file] });
		expect(input).toContainEqual(
			expect.objectContaining({ type: "function_call_output", output: [{ type: "input_text", text: "Read a.pdf" }, file] }),
		);
	});

	it("rejects documents explicitly for APIs without native PDF input", () => {
		const completionsModel: Model<"openai-completions"> = { ...model, api: "openai-completions", provider: "fireworks", id: "glm" };
		expect(() =>
			transformMessages(
				[{ role: "user", content: [{ type: "document", data: "JVBERi0=", mimeType: "application/pdf" }], timestamp: 0 }],
				completionsModel,
			),
		).toThrow(/not supported by the openai-completions API/);
	});
});
