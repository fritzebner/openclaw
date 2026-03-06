/**
 * Tests for the LangSmith tracer extension.
 *
 * Unit tests use a fake RunNode factory injected via the _runNodeFactory
 * constructor option — no real LangSmith account or module mocking needed.
 *
 * Live tests (describeLive block) require LANGSMITH_API_KEY + OPENCLAW_LIVE_TEST=1.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RunNode, RunNodeFactory } from "./src/tracer.js";

// ── fake RunNode ──────────────────────────────────────────────────────────────

type FakeRun = RunNode & {
  name: string;
  run_type: string;
  inputs: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  error?: string;
  children: FakeRun[];
};

function makeFakeRun(cfg: Record<string, unknown> = {}): FakeRun {
  const children: FakeRun[] = [];
  const run: FakeRun = {
    name: (cfg["name"] as string) ?? "fake-run",
    run_type: (cfg["run_type"] as string) ?? "chain",
    inputs: (cfg["inputs"] as Record<string, unknown>) ?? {},
    children,
    postRun: vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
    patchRun: vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
    end: vi
      .fn<(outputs?: Record<string, unknown>, error?: string) => Promise<void>>()
      .mockImplementation(async (outputs?: Record<string, unknown>, error?: string) => {
        run.outputs = outputs;
        run.error = error;
      }),
    createChild: vi
      .fn<(cfg: Record<string, unknown>) => RunNode>()
      .mockImplementation((childCfg: Record<string, unknown>) => {
        const child = makeFakeRun(childCfg);
        children.push(child);
        return child;
      }),
  };
  return run;
}

// ── LangSmithTracer unit tests ────────────────────────────────────────────────

describe("LangSmithTracer", () => {
  let LangSmithTracer: typeof import("./src/tracer.js").LangSmithTracer;
  const mockLogger = { warn: vi.fn(), info: vi.fn() };
  const mockClient = {} as import("langsmith").Client;

  beforeEach(async () => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    ({ LangSmithTracer } = await import("./src/tracer.js"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function makeTracer(rootRun: FakeRun) {
    return new LangSmithTracer({
      client: mockClient,
      projectName: "test",
      logger: mockLogger,
      _runNodeFactory: (() => rootRun) as RunNodeFactory,
    });
  }

  // ── happy path: normal order (llm_output before agent_end) ─────────────────
  it("happy path: full single-turn creates correct hierarchy", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-1";

    // Call with messages (prompt-build phase) — creates root run
    await tracer.onAgentStart(sessionId, { prompt: "hello", messages: [] });
    expect(rootRun.postRun).toHaveBeenCalledTimes(1);
    expect(tracer.activeSessionCount).toBe(1);

    // llm_input → child llm run under root
    await tracer.onLlmInput(sessionId, {
      runId: "run-1",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      prompt: "hello",
      historyMessages: [],
      imagesCount: 0,
    });
    expect(rootRun.createChild).toHaveBeenCalledTimes(1);
    const llmRun = rootRun.children[0];
    expect(llmRun.run_type).toBe("llm");

    // before_tool_call → tool run as child of ROOT
    await tracer.onBeforeToolCall(sessionId, { toolName: "bash", params: { cmd: "ls" } });
    expect(rootRun.createChild).toHaveBeenCalledTimes(2);
    const toolRun = rootRun.children[1];
    expect(toolRun.run_type).toBe("tool");
    expect(toolRun.name).toBe("bash");

    // after_tool_call → tool run closed
    await tracer.onAfterToolCall(sessionId, {
      toolName: "bash",
      params: { cmd: "ls" },
      result: "file.ts",
    });
    expect(toolRun.end).toHaveBeenCalledWith({ output: "file.ts" }, undefined);
    expect(toolRun.patchRun).toHaveBeenCalledTimes(1);

    // llm_output → llm run closed
    await tracer.onLlmOutput(sessionId, {
      runId: "run-1",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      assistantTexts: ["done"],
      usage: { input: 100, output: 20 },
    });
    expect(llmRun.end).toHaveBeenCalledWith(
      expect.objectContaining({
        generations: ["done"],
        prompt_tokens: 100,
        completion_tokens: 20,
        total_tokens: 0,
      }),
    );

    // agent_end → starts grace timer (pendingLlmCallCount is 0 here)
    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    expect(tracer.activeSessionCount).toBe(1); // timer hasn't fired yet
    await vi.advanceTimersByTimeAsync(3500);
    expect(rootRun.end).toHaveBeenCalledWith({ success: true }, undefined);
    expect(rootRun.patchRun).toHaveBeenCalledTimes(1);
    expect(tracer.activeSessionCount).toBe(0);
  });

  // ── before_prompt_build always creates a session ──────────────────────────
  // We now use before_prompt_build (not before_agent_start) which always fires
  // with messages. onAgentStart creates a session on every call.
  it("creates a session on onAgentStart (before_prompt_build always has messages)", async () => {
    const rootRun = makeFakeRun();
    const tracer = makeTracer(rootRun);

    await tracer.onAgentStart("sess-create", { prompt: "hello", messages: [] });
    expect(rootRun.postRun).toHaveBeenCalledTimes(1);
    expect(tracer.activeSessionCount).toBe(1);
  });

  // ── after_tool_call fallback: matches by toolName when agentId is absent ──
  it("after_tool_call finds tool by name when session key is unknown", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-tool-fallback";

    await tracer.onAgentStart(sessionId, { prompt: "p", messages: [] });
    await tracer.onBeforeToolCall(sessionId, {
      toolName: "search_troubleshooting",
      params: { query: "circuit down" },
    });
    const toolRun = rootRun.children[0];

    // after_tool_call fires with key "unknown" (no agentId in context)
    await tracer.onAfterToolCall("unknown", {
      toolName: "search_troubleshooting",
      params: { query: "circuit down" },
      result: { cases: ["#639376"] },
    });

    expect(toolRun.end).toHaveBeenCalledWith({ output: { cases: ["#639376"] } }, undefined);
    expect(toolRun.patchRun).toHaveBeenCalledTimes(1);
  });

  // ── deferred close: agent_end fires before llm_output ───────────────────────
  it("deferred close: agent_end before llm_output closes LLM run cleanly", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-deferred";

    await tracer.onAgentStart(sessionId, { prompt: "hello", messages: [] });

    await tracer.onLlmInput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      prompt: "hello",
      historyMessages: [],
      imagesCount: 0,
    });
    const llmRun = rootRun.children[0];

    // agent_end fires first (while LLM run is still open)
    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    expect(tracer.activeSessionCount).toBe(1);
    expect(rootRun.end).not.toHaveBeenCalled();
    expect(llmRun.end).not.toHaveBeenCalled();

    // llm_output arrives within the grace period
    await tracer.onLlmOutput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      assistantTexts: ["response"],
      usage: { input: 50, output: 10 },
    });

    // LLM run closed cleanly (no error)
    expect(llmRun.end).toHaveBeenCalledWith(expect.objectContaining({ generations: ["response"] }));
    expect(llmRun.error).toBeUndefined();

    // Root run not yet closed — finalization timer still running
    expect(rootRun.end).not.toHaveBeenCalled();
    expect(tracer.activeSessionCount).toBe(1);

    await vi.advanceTimersByTimeAsync(3500);

    // Root run closed with success after timer fires
    expect(rootRun.end).toHaveBeenCalledWith({ success: true }, undefined);
    expect(tracer.activeSessionCount).toBe(0);
  });

  // ── deferred close: timeout expires → force close ─────────────────────────
  it("deferred close: timeout force-closes LLM run if llm_output never arrives", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-timeout";

    await tracer.onAgentStart(sessionId, { prompt: "hello", messages: [] });

    await tracer.onLlmInput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      prompt: "hello",
      historyMessages: [],
      imagesCount: 0,
    });
    const llmRun = rootRun.children[0];

    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    expect(tracer.activeSessionCount).toBe(1);

    await vi.advanceTimersByTimeAsync(3500);

    expect(llmRun.end).toHaveBeenCalledWith(undefined, "agent ended with LLM call still pending");
    expect(rootRun.end).toHaveBeenCalled();
    expect(tracer.activeSessionCount).toBe(0);
  });

  // ── multi-turn ────────────────────────────────────────────────────────────
  it("multi-turn: two LLM calls create two child runs under root", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-multi";

    await tracer.onAgentStart(sessionId, { prompt: "multi", messages: [] });

    await tracer.onLlmInput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      prompt: "multi",
      historyMessages: [],
      imagesCount: 0,
    });
    await tracer.onLlmOutput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      assistantTexts: ["turn1"],
    });

    await tracer.onLlmInput(sessionId, {
      runId: "r2",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      prompt: "multi",
      historyMessages: [],
      imagesCount: 0,
    });
    await tracer.onLlmOutput(sessionId, {
      runId: "r2",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      assistantTexts: ["turn2"],
    });

    // agent_end fires after both LLM calls are done — grace timer then finalizes
    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    expect(tracer.activeSessionCount).toBe(1); // still open (timer not yet fired)

    await vi.advanceTimersByTimeAsync(3500);

    expect(rootRun.createChild).toHaveBeenCalledTimes(2);
    expect(rootRun.children).toHaveLength(2);
    expect(rootRun.children[0].run_type).toBe("llm");
    expect(rootRun.children[1].run_type).toBe("llm");
    expect(tracer.activeSessionCount).toBe(0);
  });

  // ── multi-turn: agent_end fires BETWEEN tool call and second LLM call ─────
  it("multi-turn: agent_end between tool call and second LLM still traces second LLM", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-mid-agent-end";

    await tracer.onAgentStart(sessionId, { prompt: "show tickets", messages: [] });

    // LLM call 1: model decides to call a tool
    await tracer.onLlmInput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "openrouter",
      model: "auto",
      prompt: "show tickets",
      historyMessages: [],
      imagesCount: 0,
    });
    await tracer.onLlmOutput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "openrouter",
      model: "auto",
      assistantTexts: ["calling tool"],
    });

    // Tool call
    await tracer.onBeforeToolCall(sessionId, {
      toolName: "search_troubleshooting",
      params: { query: "Amazon Kuiper" },
    });
    await tracer.onAfterToolCall("unknown", {
      toolName: "search_troubleshooting",
      params: { query: "Amazon Kuiper" },
      result: [{ id: 1 }],
    });

    // agent_end fires BEFORE the second LLM call (the scenario that was broken)
    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    expect(tracer.activeSessionCount).toBe(1); // still open — grace timer running

    // LLM call 2: model processes tool result and responds
    await tracer.onLlmInput(sessionId, {
      runId: "r2",
      sessionId,
      provider: "openrouter",
      model: "auto",
      prompt: "show tickets",
      historyMessages: [],
      imagesCount: 0,
    });
    // llm_input cancelled the grace timer — session still alive
    expect(tracer.activeSessionCount).toBe(1);

    await tracer.onLlmOutput(sessionId, {
      runId: "r2",
      sessionId,
      provider: "openrouter",
      model: "auto",
      assistantTexts: ["here are your tickets"],
      usage: { input: 100, output: 50, total: 150 },
    });

    // Not yet finalized — a new timer started when llm_output saw agentEnded+count=0
    expect(tracer.activeSessionCount).toBe(1);
    await vi.advanceTimersByTimeAsync(3500);

    expect(tracer.activeSessionCount).toBe(0);
    expect(rootRun.end).toHaveBeenCalledWith({ success: true }, undefined);

    // Both LLM runs and the tool run are children of root
    const llmRuns = rootRun.children.filter((c) => c.run_type === "llm");
    const toolRuns = rootRun.children.filter((c) => c.run_type === "tool");
    expect(llmRuns).toHaveLength(2);
    expect(toolRuns).toHaveLength(1);
  });

  // ── INTER-attempt: second before_agent_start reuses root run ─────────────
  // This is the real multi-step scenario: attempt.ts is called ONCE per LLM
  // iteration. Each call fires before_agent_start + llm_input + agent_end +
  // llm_output. Tool hooks fire between attempt.ts calls. The fix ensures
  // LLM1 + tool + LLM2 all appear under one openclaw-agent root trace.
  it("INTER-attempt: second before_agent_start reuses root run for LLM2", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    let callCount = 0;
    const tracer = new LangSmithTracer({
      client: mockClient,
      projectName: "test",
      logger: mockLogger,
      _runNodeFactory: (() => {
        callCount++;
        return rootRun; // always return same root — only called once if reuse works
      }) as RunNodeFactory,
    });
    const sessionId = "sess-inter";

    // ── attempt.ts call 1 ────────────────────────────────────────────────────
    await tracer.onAgentStart(sessionId, { prompt: "show tickets", messages: [] });
    expect(callCount).toBe(1); // root created

    await tracer.onLlmInput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "openrouter",
      model: "auto",
      prompt: "show tickets",
      historyMessages: [],
      imagesCount: 0,
    });
    // Tool runs happen inside attempt.ts call 1, before agent_end
    await tracer.onBeforeToolCall(sessionId, {
      toolName: "search_troubleshooting",
      params: { query: "Amazon Kuiper" },
    });
    await tracer.onAfterToolCall("unknown", {
      toolName: "search_troubleshooting",
      params: { query: "Amazon Kuiper" },
      result: [{ id: 1 }],
    });
    await tracer.onAgentEnd(sessionId, { messages: [], success: true }); // agentEnded set
    await tracer.onLlmOutput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "openrouter",
      model: "auto",
      assistantTexts: ["calling tool"],
    }); // count → 0; safety timer still running from agent_end; no NEW timer

    expect(tracer.activeSessionCount).toBe(1); // NOT finalized — timer running

    // ── attempt.ts call 2 ────────────────────────────────────────────────────
    // before_agent_start sees agentEnded is set → cancel timer, reuse session
    await tracer.onAgentStart(sessionId, { prompt: "show tickets", messages: [] });
    expect(callCount).toBe(1); // root was NOT recreated — still 1 factory call
    expect(tracer.activeSessionCount).toBe(1); // session still alive

    await tracer.onLlmInput(sessionId, {
      runId: "r2",
      sessionId,
      provider: "openrouter",
      model: "auto",
      prompt: "show tickets",
      historyMessages: [],
      imagesCount: 0,
    });
    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    await tracer.onLlmOutput(sessionId, {
      runId: "r2",
      sessionId,
      provider: "openrouter",
      model: "auto",
      assistantTexts: ["here are your tickets"],
      usage: { input: 200, output: 80, total: 280 },
    });

    expect(tracer.activeSessionCount).toBe(1); // timer running
    await vi.advanceTimersByTimeAsync(3500);
    expect(tracer.activeSessionCount).toBe(0);

    // All three runs (LLM1, tool, LLM2) are children of the SAME root
    const llmRuns = rootRun.children.filter((c) => c.run_type === "llm");
    const toolRuns = rootRun.children.filter((c) => c.run_type === "tool");
    expect(llmRuns).toHaveLength(2);
    expect(toolRuns).toHaveLength(1);
    expect(rootRun.end).toHaveBeenCalledWith({ success: true }, undefined);
  });

  // ── heartbeat filtering ─────────────────────────────────────────────────
  it("heartbeat runs are skipped (no trace created)", async () => {
    const rootRun = makeFakeRun();
    const tracer = makeTracer(rootRun);

    await tracer.onAgentStart("sess-hb", {
      prompt: "Read HEARTBEAT.md and respond",
      messages: [],
    });
    expect(rootRun.postRun).not.toHaveBeenCalled();
    expect(tracer.activeSessionCount).toBe(0);
  });

  it("heartbeat detection matches lowercase 'heartbeat'", async () => {
    const rootRun = makeFakeRun();
    const tracer = makeTracer(rootRun);

    await tracer.onAgentStart("sess-hb2", {
      prompt: "check heartbeat status",
      messages: [],
    });
    expect(rootRun.postRun).not.toHaveBeenCalled();
  });

  // ── error resilience ──────────────────────────────────────────────────────
  it("error in postRun is caught and logged — does not throw", async () => {
    const rootRun = makeFakeRun();
    rootRun.postRun.mockRejectedValue(new Error("network error"));
    const tracer = makeTracer(rootRun);

    await expect(
      tracer.onAgentStart("sess-err", { prompt: "fail", messages: [] }),
    ).resolves.toBeUndefined();
    expect(mockLogger.warn).toHaveBeenCalledWith(expect.stringContaining("onAgentStart failed"));
  });

  // ── unknown session is a no-op ──────────────────────────────────────────
  it("hooks for unknown sessionId are silently ignored", async () => {
    const rootRun = makeFakeRun();
    const tracer = makeTracer(rootRun);

    await tracer.onLlmInput("ghost-session", {
      runId: "r",
      sessionId: "ghost-session",
      provider: "x",
      model: "y",
      prompt: "z",
      historyMessages: [],
      imagesCount: 0,
    });
    expect(rootRun.createChild).not.toHaveBeenCalled();
    expect(mockLogger.warn).not.toHaveBeenCalled();
  });

  // ── after_tool_call with no matching tool ───────────────────────────────
  it("after_tool_call with no matching pending tool is a no-op", async () => {
    const rootRun = makeFakeRun();
    const tracer = makeTracer(rootRun);

    await tracer.onAgentStart("sess-notool", { prompt: "p", messages: [] });
    await tracer.onAfterToolCall("sess-notool", {
      toolName: "bash",
      params: {},
      result: "out",
    });
    expect(rootRun.patchRun).not.toHaveBeenCalled();
    expect(mockLogger.warn).not.toHaveBeenCalled();
  });

  // ── failed agent run sets error ────────────────────────────────────────
  it("agent_end with success=false sets error on root run", async () => {
    const rootRun = makeFakeRun();
    const tracer = makeTracer(rootRun);

    await tracer.onAgentStart("sess-fail", { prompt: "p", messages: [] });
    await tracer.onAgentEnd("sess-fail", {
      messages: [],
      success: false,
      error: "timeout",
    });
    expect(tracer.activeSessionCount).toBe(1); // grace timer running
    await vi.advanceTimersByTimeAsync(3500);
    expect(rootRun.end).toHaveBeenCalledWith(undefined, "timeout");
    expect(tracer.activeSessionCount).toBe(0);
  });

  // ── sdk_llm_start / sdk_llm_end: per-call LLM tracking ──────────────────

  it("sdk_llm_start/sdk_llm_end: two LLM calls in tool loop produce two child runs", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-sdk-llm";

    await tracer.onAgentStart(sessionId, { prompt: "hello", messages: [] });

    // LLM call 1
    await tracer.onSdkLlmStart(sessionId, { provider: "anthropic", model: "claude-sonnet-4-5" });
    await tracer.onSdkLlmEnd(sessionId, {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      stopReason: "tool_use",
      usage: { input: 100, output: 20 },
    });

    // Tool call
    await tracer.onBeforeToolCall(sessionId, { toolName: "bash", params: { cmd: "ls" } });
    await tracer.onAfterToolCall(sessionId, {
      toolName: "bash",
      params: { cmd: "ls" },
      result: "file.ts",
    });

    // LLM call 2
    await tracer.onSdkLlmStart(sessionId, { provider: "anthropic", model: "claude-sonnet-4-5" });
    await tracer.onSdkLlmEnd(sessionId, {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      stopReason: "end_turn",
      usage: { input: 200, output: 50, total: 250 },
    });

    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    await vi.advanceTimersByTimeAsync(3500);

    // Two LLM runs + one tool run as children of root
    const llmRuns = rootRun.children.filter((c) => c.run_type === "llm");
    const toolRuns = rootRun.children.filter((c) => c.run_type === "tool");
    expect(llmRuns).toHaveLength(2);
    expect(toolRuns).toHaveLength(1);
    expect(llmRuns[0].name).toBe("anthropic/claude-sonnet-4-5");
    expect(tracer.activeSessionCount).toBe(0);
  });

  it("sdk_llm_end carries per-call token usage", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-sdk-usage";

    await tracer.onAgentStart(sessionId, { prompt: "usage test", messages: [] });
    await tracer.onSdkLlmStart(sessionId, { provider: "anthropic", model: "claude-sonnet-4-5" });
    await tracer.onSdkLlmEnd(sessionId, {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      stopReason: "end_turn",
      usage: { input: 150, output: 30, total: 180 },
    });

    const llmRun = rootRun.children[0];
    expect(llmRun.end).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt_tokens: 150,
        completion_tokens: 30,
        total_tokens: 180,
      }),
    );

    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    await vi.advanceTimersByTimeAsync(3500);
  });

  it("sdk_llm_start suppresses llm_input fallback", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-sdk-suppress";

    await tracer.onAgentStart(sessionId, { prompt: "p", messages: [] });

    // SDK event fires first
    await tracer.onSdkLlmStart(sessionId, { provider: "anthropic", model: "claude-sonnet-4-5" });

    // Fallback llm_input should be skipped (hasSdkLlmEvents is true)
    await tracer.onLlmInput(sessionId, {
      runId: "r1",
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      prompt: "p",
      historyMessages: [],
      imagesCount: 0,
    });

    // Only one child run should exist (from sdk_llm_start, not llm_input)
    expect(rootRun.children).toHaveLength(1);

    await tracer.onSdkLlmEnd(sessionId, { usage: { input: 10, output: 5 } });
    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    await vi.advanceTimersByTimeAsync(3500);
  });

  it("sdk_llm_end after agent_end triggers deferred close", async () => {
    const rootRun = makeFakeRun({ name: "openclaw-agent", run_type: "chain" });
    const tracer = makeTracer(rootRun);
    const sessionId = "sess-sdk-deferred";

    await tracer.onAgentStart(sessionId, { prompt: "deferred", messages: [] });
    await tracer.onSdkLlmStart(sessionId, { provider: "anthropic", model: "claude-sonnet-4-5" });

    // agent_end fires while LLM call is still in progress
    await tracer.onAgentEnd(sessionId, { messages: [], success: true });
    expect(rootRun.end).not.toHaveBeenCalled();

    // sdk_llm_end arrives — should trigger finalization timer
    await tracer.onSdkLlmEnd(sessionId, {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      usage: { input: 50, output: 10 },
    });
    expect(tracer.activeSessionCount).toBe(1); // timer still running

    await vi.advanceTimersByTimeAsync(3500);
    expect(rootRun.end).toHaveBeenCalledWith({ success: true }, undefined);
    expect(tracer.activeSessionCount).toBe(0);
  });

  // ── stale session replaced ──────────────────────────────────────────────
  it("new agent start closes stale session before creating new one", async () => {
    const rootRun1 = makeFakeRun({ name: "root-1" });
    const rootRun2 = makeFakeRun({ name: "root-2" });
    let callCount = 0;
    const tracer = new LangSmithTracer({
      client: mockClient,
      projectName: "test",
      logger: mockLogger,
      _runNodeFactory: (() => {
        callCount++;
        return callCount === 1 ? rootRun1 : rootRun2;
      }) as RunNodeFactory,
    });

    await tracer.onAgentStart("sess-dup", { prompt: "first", messages: [] });
    expect(tracer.activeSessionCount).toBe(1);

    await tracer.onAgentStart("sess-dup", { prompt: "second", messages: [] });
    expect(rootRun1.end).toHaveBeenCalledWith(undefined, "replaced by new agent start");
    expect(rootRun2.postRun).toHaveBeenCalled();
    expect(tracer.activeSessionCount).toBe(1);
  });
});

// ── config unit tests ─────────────────────────────────────────────────────────

describe("config", () => {
  let savedApiKey: string | undefined;
  let savedProject: string | undefined;
  let savedEndpoint: string | undefined;

  beforeEach(() => {
    savedApiKey = process.env["LANGSMITH_API_KEY"];
    savedProject = process.env["LANGSMITH_PROJECT"];
    savedEndpoint = process.env["LANGSMITH_ENDPOINT"];
    delete process.env["LANGSMITH_API_KEY"];
    delete process.env["LANGSMITH_PROJECT"];
    delete process.env["LANGSMITH_ENDPOINT"];
  });

  afterEach(() => {
    if (savedApiKey !== undefined) process.env["LANGSMITH_API_KEY"] = savedApiKey;
    else delete process.env["LANGSMITH_API_KEY"];
    if (savedProject !== undefined) process.env["LANGSMITH_PROJECT"] = savedProject;
    else delete process.env["LANGSMITH_PROJECT"];
    if (savedEndpoint !== undefined) process.env["LANGSMITH_ENDPOINT"] = savedEndpoint;
    else delete process.env["LANGSMITH_ENDPOINT"];
  });

  it("isEnabled returns false when LANGSMITH_API_KEY is absent", async () => {
    const { isEnabled } = await import("./src/config.js");
    expect(isEnabled()).toBe(false);
  });

  it("isEnabled returns true when LANGSMITH_API_KEY is set", async () => {
    process.env["LANGSMITH_API_KEY"] = "ls__test";
    const { isEnabled } = await import("./src/config.js");
    expect(isEnabled()).toBe(true);
  });

  it("isEnabled returns true when apiKey is in pluginConfig", async () => {
    const { isEnabled } = await import("./src/config.js");
    expect(isEnabled({ apiKey: "ls__fromconfig" })).toBe(true);
  });

  it("resolveConfig uses defaults when env vars are absent", async () => {
    const { resolveConfig } = await import("./src/config.js");
    const cfg = resolveConfig();
    expect(cfg.project).toBe("openclaw-agent-runs");
    expect(cfg.endpoint).toBe("https://api.smith.langchain.com");
    expect(cfg.apiKey).toBe("");
  });

  it("resolveConfig respects LANGSMITH_PROJECT env var", async () => {
    process.env["LANGSMITH_PROJECT"] = "my-project";
    const { resolveConfig } = await import("./src/config.js");
    const cfg = resolveConfig();
    expect(cfg.project).toBe("my-project");
  });

  it("resolveConfig prefers env var over pluginConfig", async () => {
    process.env["LANGSMITH_PROJECT"] = "from-env";
    const { resolveConfig } = await import("./src/config.js");
    const cfg = resolveConfig({ project: "from-config" });
    expect(cfg.project).toBe("from-env");
  });
});

// ── plugin registration tests ─────────────────────────────────────────────────

describe("plugin registration", () => {
  let savedApiKey: string | undefined;

  beforeEach(() => {
    savedApiKey = process.env["LANGSMITH_API_KEY"];
    delete process.env["LANGSMITH_API_KEY"];
  });

  afterEach(() => {
    if (savedApiKey !== undefined) process.env["LANGSMITH_API_KEY"] = savedApiKey;
    else delete process.env["LANGSMITH_API_KEY"];
  });

  function makeMockApi(pluginConfig?: Record<string, unknown>) {
    return {
      pluginConfig,
      logger: { info: vi.fn(), warn: vi.fn() },
      on: vi.fn(),
    };
  }

  it("does not register hooks when LANGSMITH_API_KEY is absent", async () => {
    const { default: plugin } = await import("./index.js");
    const api = makeMockApi();
    plugin.register(api as unknown as import("openclaw/plugin-sdk").OpenClawPluginApi);
    expect(api.on).not.toHaveBeenCalled();
    expect(api.logger.info).toHaveBeenCalledWith(expect.stringContaining("disabled"));
  });
});

// ── live test ─────────────────────────────────────────────────────────────────

const liveEnabled =
  Boolean(process.env["LANGSMITH_API_KEY"]) && process.env["OPENCLAW_LIVE_TEST"] === "1";
const describeLive = liveEnabled ? describe : describe.skip;

describeLive("live: full round-trip to LangSmith", () => {
  it("agent turn produces a readable trace in LangSmith", async () => {
    const { buildClient, resolveConfig } = await import("./src/config.js");
    const { LangSmithTracer } = await import("./src/tracer.js");

    const cfg = resolveConfig();
    const client = buildClient(cfg);
    const tracer = new LangSmithTracer({
      client,
      projectName: cfg.project,
      logger: { warn: console.warn, info: console.info },
    });

    const sessionId = `live-test-${Date.now()}`;

    await tracer.onAgentStart(sessionId, {
      prompt: "live test prompt",
      messages: [{ role: "user", content: "live test prompt" }],
    });
    await tracer.onLlmInput(sessionId, {
      runId: sessionId,
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      prompt: "live test prompt",
      historyMessages: [{ role: "user", content: "live test prompt" }],
      imagesCount: 0,
    });
    await tracer.onBeforeToolCall(sessionId, {
      toolName: "bash",
      params: { command: "echo hello" },
    });
    // Simulate after_tool_call arriving with no agentId (as OpenClaw does)
    await tracer.onAfterToolCall("unknown", {
      toolName: "bash",
      params: { command: "echo hello" },
      result: "hello\n",
      durationMs: 42,
    });
    await tracer.onLlmOutput(sessionId, {
      runId: sessionId,
      sessionId,
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      assistantTexts: ["The command output was: hello"],
      usage: { input: 50, output: 10, total: 60 },
    });
    await tracer.onAgentEnd(sessionId, { messages: [], success: true, durationMs: 500 });

    expect(tracer.activeSessionCount).toBe(0);
    console.info(`Live test done — check project "${cfg.project}" in LangSmith`);
  });
});
