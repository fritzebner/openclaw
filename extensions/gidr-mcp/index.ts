import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi } from "../../src/plugins/types.js";
import { callGidrTool, isGidrMcpConfigured } from "./gidr-client.js";

export default function register(api: OpenClawPluginApi) {
  if (!isGidrMcpConfigured()) {
    return;
  }

  const searchTroubleshooting = {
    name: "search_troubleshooting",
    description:
      "Search Firstlight troubleshooting / NOC knowledge base. Use for network or equipment troubleshooting, outage or symptom queries. Prefer text search_mode; include account_name or location when known. Return results with citations.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query (symptoms, equipment, vendor, etc.)" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
      search_mode: Type.Optional(
        Type.String({
          description: "Search mode: text or hybrid",
          default: "text",
        }),
      ),
      account_name: Type.Optional(Type.String({ description: "Account or operator name filter" })),
      location: Type.Optional(Type.String({ description: "Location filter" })),
      equipment_vendor: Type.Optional(Type.String({ description: "Equipment vendor filter" })),
      service_type: Type.Optional(Type.String({ description: "Service type filter" })),
    }),
    async execute(
      _id: string,
      params: Record<string, unknown>,
      _signal?: AbortSignal,
    ): Promise<{ content: Array<{ type: string; text: string }> }> {
      const query = typeof params.query === "string" ? params.query.trim() : "";
      if (!query) {
        throw new Error("query is required");
      }
      const args: Record<string, unknown> = {
        query,
        limit: typeof params.limit === "number" ? params.limit : 5,
        search_mode: typeof params.search_mode === "string" ? params.search_mode : "text",
      };
      if (typeof params.account_name === "string" && params.account_name.trim()) {
        args.account_name = params.account_name.trim();
      }
      if (typeof params.location === "string" && params.location.trim()) {
        args.location = params.location.trim();
      }
      if (typeof params.equipment_vendor === "string" && params.equipment_vendor.trim()) {
        args.equipment_vendor = params.equipment_vendor.trim();
      }
      if (typeof params.service_type === "string" && params.service_type.trim()) {
        args.service_type = params.service_type.trim();
      }
      const out = await callGidrTool("search_troubleshooting", args);
      if (!out.ok) {
        throw new Error(out.error?.message ?? "search_troubleshooting failed");
      }
      const text =
        typeof out.value === "object" && out.value !== null
          ? JSON.stringify(out.value, null, 2)
          : String(out.value);
      return { content: [{ type: "text", text }] };
    },
  };

  const retrievalFirstlightNoc = {
    name: "retrieval_firstlight_noc",
    description:
      "Retrieve Firstlight NOC / runbook content. Use for NOC procedures, runbooks, or when the user asks for Firstlight NOC. Combine with search_troubleshooting when the user has a troubleshooting question.",
    parameters: Type.Object({
      query: Type.String({ description: "Query for NOC/runbook retrieval" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
      search_mode: Type.Optional(Type.String({ description: "Search mode: text or hybrid" })),
    }),
    async execute(
      _id: string,
      params: Record<string, unknown>,
      _signal?: AbortSignal,
    ): Promise<{ content: Array<{ type: string; text: string }> }> {
      const query = typeof params.query === "string" ? params.query.trim() : "";
      if (!query) {
        throw new Error("query is required");
      }
      const args: Record<string, unknown> = {
        text: query, // MCP server expects "text"; OpenClaw schema exposes it as "query"
        limit: typeof params.limit === "number" ? params.limit : 5,
        search_mode: typeof params.search_mode === "string" ? params.search_mode : "text",
      };
      const out = await callGidrTool("retrieval_firstlight_noc", args);
      if (!out.ok) {
        throw new Error(out.error?.message ?? "retrieval_firstlight_noc failed");
      }
      const text =
        typeof out.value === "object" && out.value !== null
          ? JSON.stringify(out.value, null, 2)
          : String(out.value);
      return { content: [{ type: "text", text }] };
    },
  };

  api.registerTool(() => [searchTroubleshooting, retrievalFirstlightNoc], {
    names: ["search_troubleshooting", "retrieval_firstlight_noc"],
    optional: true,
  });
}
