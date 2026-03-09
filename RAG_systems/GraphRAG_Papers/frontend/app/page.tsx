"use client";

import { useMemo, useState } from "react";

type Paper = {
  id: string;
  title: string;
  year: number;
  abstract: string;
  authors: string[];
  topics: string[];
};

type QueryResponse = {
  answer: string;
  papers: Paper[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [showGraph, setShowGraph] = useState(false);

  const canSearch = query.trim().length > 2 && !loading;

  const handleSearch = async () => {
    if (!canSearch) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error("Request failed");
      }

      const data: QueryResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError("Unable to fetch results. Check the backend service.");
    } finally {
      setLoading(false);
    }
  };

  const paperList = useMemo(() => result?.papers ?? [], [result]);

  return (
    <main className="px-6 py-12">
      <div className="mx-auto max-w-5xl">
        <header className="mb-10">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate">
            Graph-RAG Research Assistant
          </p>
          <h1 className="mt-3 text-4xl font-semibold text-ink">
            Explore arXiv with graph-first reasoning
          </h1>
          <p className="mt-4 max-w-2xl text-base text-slate">
            Ask relational questions about authors, topics, and trends in graph neural networks. The system retrieves
            evidence from a Neo4j knowledge graph before synthesizing an answer.
          </p>
        </header>

        <section className="rounded-2xl border border-mist bg-white p-6 shadow-soft">
          <div className="flex flex-col gap-4 md:flex-row">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Which authors publish most on graph neural networks?"
              className="flex-1 rounded-xl border border-mist px-4 py-3 text-base focus:border-accent focus:outline-none"
            />
            <button
              onClick={handleSearch}
              disabled={!canSearch}
              className="rounded-xl bg-accent px-6 py-3 text-white transition disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Searching…" : "Search"}
            </button>
          </div>
          {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
        </section>

        {result && (
          <section className="mt-10 grid gap-6">
            <div className="rounded-2xl border border-mist bg-white p-6 shadow-soft">
              <h2 className="text-lg font-semibold text-ink">Answer</h2>
              <p className="mt-3 text-slate whitespace-pre-line">{result.answer}</p>
            </div>

            <div className="rounded-2xl border border-mist bg-white p-6 shadow-soft">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-ink">Referenced Papers</h2>
                <button
                  className="text-sm font-medium text-accent"
                  onClick={() => setShowGraph((v) => !v)}
                >
                  {showGraph ? "Hide" : "Show"} graph insights
                </button>
              </div>

              <div className="mt-4 grid gap-4">
                {paperList.length === 0 && (
                  <p className="text-sm text-slate">No papers found for this query.</p>
                )}
                {paperList.map((paper) => (
                  <article key={paper.id} className="rounded-xl border border-mist p-4">
                    <h3 className="text-base font-semibold text-ink">{paper.title}</h3>
                    <p className="mt-2 text-sm text-slate">
                      {paper.authors.join(", ")} · {paper.year}
                    </p>
                    <p className="mt-3 text-sm text-slate line-clamp-4">{paper.abstract}</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {paper.topics.map((topic) => (
                        <span
                          key={topic}
                          className="rounded-full bg-mist px-3 py-1 text-xs font-medium text-slate"
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  </article>
                ))}
              </div>

              {showGraph && (
                <div className="mt-6 rounded-xl border border-mist bg-slate/5 p-4">
                  <p className="text-sm text-slate">
                    Graph insight: results are retrieved via Neo4j traversals across Paper, Author, and Topic nodes.
                    This view is a placeholder for future interactive graph visualization.
                  </p>
                </div>
              )}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
