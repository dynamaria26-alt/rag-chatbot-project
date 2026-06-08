import { useState, useRef, useEffect } from "react";

const CHUNK_SIZE = 400;
const CHUNK_OVERLAP = 80;
const TOP_K = 4;

function chunkText(text, source) {
  const words = text.split(/\s+/);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    const chunk = words.slice(i, i + CHUNK_SIZE).join(" ");
    if (chunk.trim()) chunks.push({ text: chunk, source });
    i += CHUNK_SIZE - CHUNK_OVERLAP;
  }
  return chunks;
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

function tfidfEmbed(text, vocab) {
  const words = text.toLowerCase().split(/\W+/);
  const freq = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;
  return vocab.map(v => (freq[v] || 0) / (words.length + 1));
}

function buildVocab(chunks) {
  const allWords = chunks.flatMap(c => c.text.toLowerCase().split(/\W+/));
  const freq = {};
  for (const w of allWords) if (w.length > 2) freq[w] = (freq[w] || 0) + 1;
  return Object.entries(freq).filter(([, v]) => v > 1).sort((a, b) => b[1] - a[1]).slice(0, 1500).map(([k]) => k);
}

function retrieve(query, chunks, embeddings, vocab, k) {
  const qEmb = tfidfEmbed(query, vocab);
  return chunks
    .map((c, i) => ({ ...c, score: cosineSim(qEmb, embeddings[i]) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

async function askClaude(messages) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "claude-sonnet-4-20250514", max_tokens: 1000, messages }),
  });
  const data = await res.json();
  return data.content?.map(b => b.text || "").join("") || "No response.";
}

const sampleDocs = [
  {
    name: "sample-knowledge-base.txt",
    content: `Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation. Instead of relying solely on pre-trained knowledge, RAG systems first retrieve relevant documents from a knowledge base, then use those documents as context to generate more accurate and grounded answers.

The main benefits of RAG include: reduced hallucinations, up-to-date information without retraining, source attribution, and domain-specific knowledge injection.

RAG systems typically consist of three components: a document store (the knowledge base), a retrieval mechanism (often using embeddings and vector similarity search), and a generation model (like GPT or Claude) that synthesizes the retrieved context with the user query.

Vector embeddings are numerical representations of text that capture semantic meaning. Similar texts have embeddings that are close together in vector space. This property enables semantic search, where we can find documents that are conceptually related to a query even if they don't share exact keywords.

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate how important a word is to a document in a collection. It increases proportionally with the number of times a word appears in the document, but is offset by the frequency of the word in the corpus.

Chunking is the process of splitting large documents into smaller pieces before indexing. Good chunking strategies balance between having enough context in each chunk (for coherent answers) and keeping chunks small enough to be specific (for precise retrieval). Common chunk sizes range from 200 to 1000 tokens.

The cosine similarity metric measures the angle between two vectors. A value of 1 means the vectors point in the same direction (identical content), while 0 means they're orthogonal (no relation). It's widely used in information retrieval because it's scale-invariant — the length of the document doesn't affect the similarity score.

Anthropic's Claude is a large language model built to be safe, helpful, and honest. It excels at complex reasoning, document analysis, and following nuanced instructions. Claude is available via API and can be integrated into RAG pipelines for enterprise document Q&A systems.`
  }
];

export default function RAGChatbot() {
  const [docs, setDocs] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [embeddings, setEmbeddings] = useState([]);
  const [vocab, setVocab] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [sources, setSources] = useState([]);
  const [tab, setTab] = useState("chat");
  const fileRef = useRef();
  const bottomRef = useRef();

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  function indexChunks(newChunks) {
    const v = buildVocab(newChunks);
    const emb = newChunks.map(c => tfidfEmbed(c.text, v));
    setVocab(v);
    setEmbeddings(emb);
    return { v, emb };
  }

  async function handleFile(file) {
    setProcessing(true);
    const text = await file.text();
    const newChunks = chunkText(text, file.name);
    const allChunks = [...chunks, ...newChunks];
    setDocs(d => [...d, { name: file.name, size: file.size, count: newChunks.length }]);
    setChunks(allChunks);
    indexChunks(allChunks);
    setProcessing(false);
  }

  function loadSample() {
    setProcessing(true);
    const doc = sampleDocs[0];
    const newChunks = chunkText(doc.content, doc.name);
    setDocs([{ name: doc.name, size: doc.content.length, count: newChunks.length }]);
    setChunks(newChunks);
    indexChunks(newChunks);
    setProcessing(false);
    setMessages([{ role: "assistant", content: "Sample knowledge base loaded! I know about RAG systems, vector embeddings, TF-IDF, and Anthropic Claude. Ask me anything about these topics." }]);
  }

  async function send() {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(m => [...m, { role: "user", content: userMsg }]);
    setLoading(true);
    setSources([]);

    let systemPrompt = "You are a helpful assistant. Answer the user's question concisely and accurately.";
    let contextBlock = "";

    if (chunks.length > 0) {
      const top = retrieve(userMsg, chunks, embeddings, vocab, TOP_K);
      setSources(top.filter(t => t.score > 0.01));
      contextBlock = top.filter(t => t.score > 0.01).map((c, i) =>
        `[Source ${i + 1} — ${c.source}]\n${c.text}`
      ).join("\n\n---\n\n");
      systemPrompt = `You are a helpful assistant that answers questions based on provided documents.

Use ONLY the context below to answer. If the answer isn't in the context, say so honestly.
Always cite which source (by number) supports your answer.

CONTEXT:
${contextBlock || "No relevant context found."}`;
    }

    const history = [...messages, { role: "user", content: userMsg }]
      .slice(-10)
      .map(m => ({ role: m.role, content: m.content }));

    try {
      const apiMessages = [{ role: "user", content: systemPrompt + "\n\nUser question: " + userMsg }];
      if (history.length > 1) {
        const chatHistory = history.slice(0, -1);
        apiMessages[0] = {
          role: "user",
          content: systemPrompt + "\n\nConversation history:\n" +
            chatHistory.map(m => `${m.role}: ${m.content}`).join("\n") +
            "\n\nCurrent question: " + userMsg
        };
      }
      const reply = await askClaude(apiMessages);
      setMessages(m => [...m, { role: "assistant", content: reply }]);
    } catch (e) {
      setMessages(m => [...m, { role: "assistant", content: "Error: " + e.message }]);
    }
    setLoading(false);
  }

  const hasIndex = chunks.length > 0;

  return (
    <div style={{ fontFamily: "Georgia, 'Times New Roman', serif", maxWidth: 780, margin: "0 auto", padding: "1rem 0" }}>
      <h2 className="sr-only">RAG Chatbot with Document Retrieval</h2>

      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: "1.25rem", borderBottom: "0.5px solid var(--color-border-tertiary)", paddingBottom: "1rem" }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--color-text-secondary)", marginBottom: 4 }}>Document Intelligence</div>
          <div style={{ fontSize: 22, fontWeight: 500, color: "var(--color-text-primary)" }}>RAG Chatbot</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["chat", "docs"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "5px 14px", fontSize: 13, borderRadius: "var(--border-radius-md)",
              background: tab === t ? "var(--color-background-secondary)" : "transparent",
              border: `0.5px solid ${tab === t ? "var(--color-border-secondary)" : "var(--color-border-tertiary)"}`,
              color: tab === t ? "var(--color-text-primary)" : "var(--color-text-secondary)", cursor: "pointer"
            }}>
              <i className={`ti ti-${t === "chat" ? "message" : "files"}`} style={{ marginRight: 5, fontSize: 13 }} aria-hidden="true" />
              {t === "chat" ? "Chat" : `Docs${docs.length ? ` (${docs.length})` : ""}`}
            </button>
          ))}
        </div>
      </div>

      {tab === "docs" && (
        <div>
          <div
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); [...e.dataTransfer.files].forEach(handleFile); }}
            onClick={() => fileRef.current.click()}
            style={{
              border: "0.5px dashed var(--color-border-secondary)", borderRadius: "var(--border-radius-lg)",
              padding: "2.5rem 1.5rem", textAlign: "center", cursor: "pointer",
              background: "var(--color-background-secondary)", marginBottom: "1rem"
            }}
          >
            <i className="ti ti-cloud-upload" style={{ fontSize: 32, color: "var(--color-text-secondary)", display: "block", marginBottom: 8 }} aria-hidden="true" />
            <div style={{ fontSize: 14, color: "var(--color-text-secondary)" }}>Drop .txt files here or click to browse</div>
            <input ref={fileRef} type="file" accept=".txt" multiple style={{ display: "none" }} onChange={e => [...e.target.files].forEach(handleFile)} />
          </div>

          <button onClick={loadSample} style={{
            width: "100%", padding: "10px", fontSize: 13, borderRadius: "var(--border-radius-md)",
            border: "0.5px solid var(--color-border-tertiary)", cursor: "pointer",
            background: "transparent", color: "var(--color-text-secondary)", marginBottom: "1.25rem"
          }}>
            <i className="ti ti-sparkles" style={{ marginRight: 6 }} aria-hidden="true" />
            Load sample knowledge base (RAG / AI concepts)
          </button>

          {processing && (
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)", textAlign: "center", padding: "0.5rem" }}>
              <i className="ti ti-loader" style={{ marginRight: 6 }} aria-hidden="true" />Indexing document…
            </div>
          )}

          {docs.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {docs.map((d, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 12, padding: "10px 14px",
                  background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)",
                  borderRadius: "var(--border-radius-md)"
                }}>
                  <i className="ti ti-file-text" style={{ fontSize: 18, color: "var(--color-text-secondary)" }} aria-hidden="true" />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>{d.name}</div>
                    <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{d.count} chunks · {(d.size / 1024).toFixed(1)} KB</div>
                  </div>
                  <div style={{
                    fontSize: 11, padding: "2px 8px", borderRadius: "var(--border-radius-md)",
                    background: "var(--color-background-success)", color: "var(--color-text-success)"
                  }}>indexed</div>
                </div>
              ))}
            </div>
          )}

          {hasIndex && (
            <div style={{
              marginTop: "1.25rem", padding: "12px 14px", borderRadius: "var(--border-radius-md)",
              background: "var(--color-background-info)", border: "0.5px solid var(--color-border-info)",
              fontSize: 13, color: "var(--color-text-info)"
            }}>
              <i className="ti ti-check" style={{ marginRight: 6 }} aria-hidden="true" />
              {chunks.length} chunks indexed across {docs.length} document{docs.length !== 1 ? "s" : ""}. Switch to Chat to ask questions.
            </div>
          )}
        </div>
      )}

      {tab === "chat" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {!hasIndex && (
            <div style={{
              padding: "1.25rem", borderRadius: "var(--border-radius-lg)", marginBottom: "1rem",
              background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)",
              fontSize: 13, color: "var(--color-text-secondary)", textAlign: "center"
            }}>
              <i className="ti ti-database-off" style={{ fontSize: 24, display: "block", margin: "0 auto 8px" }} aria-hidden="true" />
              No documents indexed yet. Go to <strong style={{ color: "var(--color-text-primary)" }}>Docs</strong> to upload files or load a sample.
              <br />You can still chat — Claude will answer from general knowledge.
            </div>
          )}

          <div style={{ minHeight: 320, maxHeight: 440, overflowY: "auto", display: "flex", flexDirection: "column", gap: "1rem", marginBottom: "0.75rem", paddingRight: 4 }}>
            {messages.length === 0 && (
              <div style={{ textAlign: "center", color: "var(--color-text-secondary)", fontSize: 13, marginTop: "2rem", fontStyle: "italic" }}>
                Ask anything — {hasIndex ? "I'll search your documents first" : "general knowledge mode"}
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", flexDirection: m.role === "user" ? "row-reverse" : "row" }}>
                <div style={{
                  width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
                  background: m.role === "user" ? "var(--color-background-secondary)" : "var(--color-background-info)",
                  display: "flex", alignItems: "center", justifyContent: "center", border: "0.5px solid var(--color-border-tertiary)"
                }}>
                  <i className={`ti ti-${m.role === "user" ? "user" : "robot"}`} style={{ fontSize: 14, color: m.role === "user" ? "var(--color-text-secondary)" : "var(--color-text-info)" }} aria-hidden="true" />
                </div>
                <div style={{
                  maxWidth: "78%", padding: "10px 14px", borderRadius: "var(--border-radius-lg)",
                  background: m.role === "user" ? "var(--color-background-secondary)" : "var(--color-background-primary)",
                  border: "0.5px solid var(--color-border-tertiary)",
                  fontSize: 14, color: "var(--color-text-primary)", lineHeight: 1.65,
                  fontFamily: m.role === "assistant" ? "Georgia, serif" : "inherit"
                }}>
                  {m.content}
                </div>
              </div>
            ))}
            {loading && (
              <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--color-background-info)", display: "flex", alignItems: "center", justifyContent: "center", border: "0.5px solid var(--color-border-tertiary)" }}>
                  <i className="ti ti-robot" style={{ fontSize: 14, color: "var(--color-text-info)" }} aria-hidden="true" />
                </div>
                <div style={{ padding: "10px 14px", borderRadius: "var(--border-radius-lg)", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", fontSize: 13, color: "var(--color-text-secondary)" }}>
                  {hasIndex ? "Retrieving context…" : "Thinking…"}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {sources.length > 0 && !loading && (
            <div style={{ marginBottom: "0.75rem" }}>
              <div style={{ fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-secondary)", marginBottom: 6 }}>Retrieved sources</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {sources.map((s, i) => (
                  <div key={i} style={{
                    padding: "8px 12px", borderRadius: "var(--border-radius-md)",
                    background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)",
                    fontSize: 12, color: "var(--color-text-secondary)", display: "flex", gap: 8, alignItems: "flex-start"
                  }}>
                    <span style={{
                      flexShrink: 0, width: 18, height: 18, borderRadius: "50%", fontSize: 10, fontWeight: 500,
                      background: "var(--color-background-info)", color: "var(--color-text-info)",
                      display: "flex", alignItems: "center", justifyContent: "center"
                    }}>{i + 1}</span>
                    <div>
                      <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{s.source}</span>
                      <span style={{ marginLeft: 6 }}>· score {s.score.toFixed(3)}</span>
                      <div style={{ marginTop: 3, color: "var(--color-text-secondary)" }}>{s.text.slice(0, 120)}…</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ display: "flex", gap: 8 }}>
            <input
              type="text" value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && send()}
              placeholder={hasIndex ? "Ask about your documents…" : "Ask anything…"}
              style={{ flex: 1, padding: "9px 12px", fontSize: 14, borderRadius: "var(--border-radius-md)", fontFamily: "inherit" }}
            />
            <button onClick={send} disabled={loading || !input.trim()} style={{
              padding: "9px 16px", borderRadius: "var(--border-radius-md)", cursor: loading ? "default" : "pointer",
              fontSize: 13, border: "0.5px solid var(--color-border-secondary)", background: "transparent",
              color: loading ? "var(--color-text-secondary)" : "var(--color-text-primary)"
            }}>
              <i className="ti ti-send" style={{ fontSize: 15 }} aria-hidden="true" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
