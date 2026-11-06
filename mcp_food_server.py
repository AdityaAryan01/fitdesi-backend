"""
mcp_food_server.py — FitDesi MCP Server (OpenFoodFacts Fallback)
=================================================================
A standalone MCP server that exposes one tool:
    search_openfoodfacts(query: str) -> str

The Calculator Agent calls this when a food item is NOT FOUND
in the local PostgreSQL food_items table. It queries the free
OpenFoodFacts REST API (no key needed) and returns macro data
from real product labels.

Transport: stdio (run as a subprocess by agent.py on startup)

Usage (standalone test):
    python mcp_food_server.py
"""

import asyncio
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json

# ── Server Init ──────────────────────────────────────────────────────────────

app = Server("fitdesi-food-server")

# ── OpenFoodFacts API Config ──────────────────────────────────────────────────

OFF_ENDPOINTS = [
    "https://in.openfoodfacts.org/cgi/search.pl",
    "https://world.openfoodfacts.org/cgi/search.pl",
]
OFF_HEADERS = {
    "User-Agent": "FitDesi-AI/1.0 (educational project; github.com/AdityaAryan01)"
}

def _extract_macros(product: dict) -> dict | None:
    """Extract and validate macro nutrients from an OFF product dict."""
    n = product.get("nutriments", {})
    kcal     = n.get("energy-kcal_100g") or n.get("energy-kcal") or n.get("energy_100g")
    protein  = n.get("proteins_100g")     or n.get("proteins")
    carbs    = n.get("carbohydrates_100g") or n.get("carbohydrates")
    fat      = n.get("fat_100g")          or n.get("fat")

    # Skip products with no meaningful macro data
    if kcal is None and protein is None:
        return None

    return {
        "kcal":    round(float(kcal   or 0), 1),
        "protein": round(float(protein or 0), 1),
        "carbs":   round(float(carbs  or 0), 1),
        "fat":     round(float(fat    or 0), 1),
    }


async def _query_openfoodfacts(query: str) -> str:
    """
    Hits OpenFoodFacts search APIs (India mirror first, then World).
    Returns a formatted macro string, or 'NOT_FOUND_IN_OFF' if nothing useful.
    """
    params = {
        "search_terms":    query,
        "search_simple":   1,
        "action":          "process",
        "json":            1,
        "page_size":       5,
        "fields":          "product_name,nutriments,serving_size,brands",
        "lc":              "en",
    }

    data = None
    async with httpx.AsyncClient(timeout=6.0) as client:
        for url in OFF_ENDPOINTS:
            try:
                resp = await client.get(url, params=params, headers=OFF_HEADERS)
                if resp.status_code == 200 and "json" in resp.headers.get("content-type", ""):
                    data = resp.json()
                    break
            except Exception:
                continue

    if not data:
        return "NOT_FOUND_IN_OFF"

    products = data.get("products", [])
    if not products:
        return "NOT_FOUND_IN_OFF"

    # Walk through top-5 results and pick the first with complete macro data
    for product in products:
        name   = product.get("product_name", "").strip()
        brand  = product.get("brands", "").strip()
        macros = _extract_macros(product)

        if not name or not macros:
            continue

        display_name = f"{brand} {name}".strip() if brand else name
        serving = product.get("serving_size") or "100g"

        return (
            f"[OpenFoodFacts] {display_name}: "
            f"{macros['kcal']} kcal | {macros['protein']}g protein | "
            f"{macros['carbs']}g carbs | {macros['fat']}g fat "
            f"per 100g (serving size on label: {serving})"
        )

    return "NOT_FOUND_IN_OFF"


# ── MCP Tool Definition ────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_openfoodfacts",
            description=(
                "Fallback food macro lookup using the OpenFoodFacts database. "
                "Use this ONLY when the local PostgreSQL food database returns NOT FOUND. "
                "Best for packaged Indian products: Amul, Mother Dairy, Haldiram's, "
                "MuscleBlaze, Epigamia, Britannia, protein bars, packaged snacks, etc. "
                "Returns per-100g macros from the actual product label."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The food or product name to search for (e.g., 'Epigamia Greek Yogurt', 'MuscleBlaze Whey')."
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "search_openfoodfacts":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="NOT_FOUND_IN_OFF")]

    print(f"[MCP Server] Querying OpenFoodFacts for: '{query}'", flush=True)
    result = await _query_openfoodfacts(query)
    print(f"[MCP Server] Result: {result[:120]}", flush=True)

    return [TextContent(type="text", text=result)]


# ── Entry Point ────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

