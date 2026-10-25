"""
app.py - LangGraph chat RAG app over MIT 6.046J material.
Graph:
    START -> router --+--> chat_direct --------------------------> END   (chit-chat, memory)
                      +--> decline -----------------------------> END   (unrelated topics)
                      +--> rewrite -> retrieve -> judge --+--> retrieve (refine, loop)
                                                          +--> generate -> END
The router (LLM) decides per message whether to trigger the knowledge base or let the
chatbot answer directly. Course answers come ONLY from the retrieved lecture passages.
Needs the GROQ_API_KEY environment variable (Space secret on Hugging Face).
Run locally:  python app.py
"""
import json
import os
import pickle
import re
from pathlib import Path
from typing import TypedDict

import faiss
import gradio as gr
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

try:
    # Free-tier Gradio Spaces run on ZeroGPU, which refuses to start an app that has
    # no @spaces.GPU function. This app runs on CPU, so this is only a placeholder
    # that satisfies the check (it is never called and uses no GPU quota).
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return True
except ImportError:  # running locally, where the 'spaces' package is not installed
    pass

EMB_MODEL = "BAAI/bge-small-en-v1.5"  # must match build_index.py
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "qwen/qwen3.8-27b")  # Groq model id
TOP_K = 4                                         # passages per search
MAX_LOOPS = int(os.getenv("MAX_LOOPS", "2"))      # search/judge rounds
MAX_CTX = 6                                       # passages kept for the final answer
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))

NOT_COVERED = ("I can't find this in the lecture material I was given, so I won't "
               "answer it. Try a question about the course topic (dynamic programming).")
OFFTOPIC_REPLY = ("That's outside what I can help with here. I only answer from the "
                  "MIT 6.046J lecture material, so ask me about dynamic programming or "
                  "the course, for example: *What is memoization?*")
SMALLTALK_REPLY = ("Hi! I'm a teaching assistant for MIT 6.046J (Design and Analysis of "
                   "Algorithms). Ask me about the lecture material, for example: "
                   "*What is memoization?* or *How do top-down and bottom-up DP differ?*")
ROUTE_LABEL = {"chat": "direct chat", "knowledge": "knowledge base", "offtopic": "out of scope"}

THINK = re.compile(r"<think>.*?</think>", re.S)

# fast paths for the router (no LLM call needed)
SMALLTALK = re.compile(
    r"^\s*(hi+|hello+|hey+|hola|namaste|sat sri akal|good (morning|afternoon|evening)|"
    r"thanks?( you)?( so much)?|thank you|ok(ay)?|cool|bye|goodbye|"
    r"who are you|what can you do|help)\W*\s*$", re.I)
KB_WORDS = re.compile(
    r"\b(dynamic programming|dp|memoi[sz]\w*|subproblems?|recurrences?|recursion|recursive|"
    r"algorithms?|complexity|runtime|running time|big[- ]?o|knapsack|bellman|dijkstra|"
    r"shortest paths?|graphs?|lectures?|recitation|transcript|6\.046|topological|dag|"
    r"bottom[- ]up|top[- ]down|fibonacci|parenthesization|text justification|"
    r"longest common subsequence|edit distance|optimal substructure)\b", re.I)

ROUTER_PROMPT = """You are the router of a teaching chatbot for the MIT course 6.046J (algorithms, dynamic programming). Classify the student's latest message with exactly ONE word:
KNOWLEDGE - a question about the course, algorithms, data structures, dynamic programming, math or computer-science concepts, or a follow-up to such a question ("explain that again", "give an example").
CHAT - greetings, thanks, introductions ("my name is Sam"), statements about the student, questions about the student or this conversation ("what's my name?", "what did I ask?"), questions about the bot itself, compliments, goodbyes.
OFFTOPIC - anything else (general knowledge, news, sports, jokes, unrelated coding or homework).
Reply with only KNOWLEDGE, CHAT or OFFTOPIC."""

CHAT_PROMPT = """You are a friendly teaching-assistant chatbot for the MIT course 6.046J (Design and Analysis of Algorithms).
Chat naturally and briefly (1-3 sentences). Remember and use anything the student told you earlier in this conversation (their name, goals).
IMPORTANT: never answer subject-matter questions about algorithms or any other topic from your own knowledge. If the student seems to want that, tell them you answer from the lecture material and invite them to ask the question. Do not invent facts about the course."""

SYSTEM_PROMPT = """You are a teaching assistant for MIT 6.046J (Design and Analysis of Algorithms).
Answer the student's question using ONLY the numbered context passages below.
Rules:
- If the passages do not contain the answer, reply exactly: "This is not covered in the lecture material."
- Do not use outside knowledge, even if you know the answer.
- Cite the passages you used like [1], [2] after the sentences they support.
- Passages from transcripts are auto-generated captions and may contain spelling errors; fix obvious ones silently.
- Passages marked AI-generated page description may contain small errors; say so if a number matters.
- Be clear and concise, and explain step by step when the question asks for reasoning."""

REWRITE_PROMPT = """Rewrite the student's latest message as ONE standalone search query for course material on dynamic programming, using the chat history to resolve words like "it", "that" or "again". Output only the query, nothing else."""

JUDGE_PROMPT = """You check whether retrieved lecture passages contain enough information to answer a student's question.
Reply with ONLY a JSON object, no other text:
{"sufficient": true or false, "next_query": "a short, differently worded search query likely to find the missing information, or an empty string if sufficient"}"""

try:  # device="cpu": avoids ZeroGPU's emulated CUDA device
    emb = SentenceTransformer(EMB_MODEL, device="cpu")
except Exception as e:  # Hub unreachable: use the copy already cached on disk
    print("Hub unreachable, loading cached model:", str(e)[:80])
    emb = SentenceTransformer(EMB_MODEL, device="cpu", local_files_only=True)
index = faiss.read_index("index.faiss")
chunks = pickle.load(open("chunks.pkl", "rb"))
sources = json.loads(Path("data/sources.json").read_text(encoding="utf-8"))


# ---------- retrieval + LLM helpers ----------
def retrieve(query):
    qv = emb.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    scores, ids = index.search(qv, TOP_K)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def llm(messages, max_tokens=800, temperature=0.2):
    """One Groq chat call. Returns text, or None if the API is unavailable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        try:  # thinking off: faster, and saves the free token budget
            resp = client.chat.completions.create(
                model=ANSWER_MODEL, messages=messages, temperature=temperature,
                max_tokens=max_tokens, extra_body={"reasoning_effort": "none"})
        except Exception as e:
            if "429" in str(e):
                raise
            resp = client.chat.completions.create(
                model=ANSWER_MODEL, messages=messages, temperature=temperature,
                max_tokens=max(max_tokens, 1200),
                extra_body={"reasoning_format": "hidden"})
        return THINK.sub("", resp.choices[0].message.content or "").strip() or None
    except Exception as e:
        print("LLM error:", str(e)[:200])
        return None


def _text(content):
    """Chat history content can be a string, dict or list depending on Gradio version."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, (list, tuple)):
        return " ".join(_text(x) for x in content)
    return ""


def history_pairs(history, keep=6):
    pairs, user = [], None
    for h in history or []:
        if isinstance(h, dict):
            if h.get("role") == "user":
                user = _text(h.get("content"))
            elif h.get("role") == "assistant" and user is not None:
                pairs.append((user, _text(h.get("content"))))
                user = None
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            pairs.append((_text(h[0]), _text(h[1])))
    # keep only the answer part of earlier replies (drop sources / steps)
    return [(u, a.split("\n\n---")[0].split("\n\n_")[0][:500]) for u, a in pairs[-keep:]]


def rewrite_query(question, pairs):
    if not pairs:
        return question
    convo = "\n".join(f"Student: {u}\nAssistant: {a}" for u, a in pairs[-3:])
    out = llm([{"role": "system", "content": REWRITE_PROMPT},
               {"role": "user", "content": f"Chat history:\n{convo}\n\nLatest message: {question}"}],
              max_tokens=80, temperature=0)
    return (out or question).strip().strip('"')[:300]


def ranked_pool(pool):
    return sorted(pool.values(), key=lambda x: x[1], reverse=True)[:MAX_CTX]


def judge(question, ranked):
    """Ask the LLM if the passages suffice. Returns a dict, or None if unavailable."""
    snippets = "\n".join(f"[{n}] {c['text'][:350]}" for n, (c, _) in enumerate(ranked, 1))
    out = llm([{"role": "system", "content": JUDGE_PROMPT},
               {"role": "user", "content": f"Question: {question}\n\nPassages:\n{snippets}"}],
              max_tokens=150, temperature=0)
    if not out:
        return None
    m = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def format_sources(used):
    """used = list of (number, chunk, score) for passages the answer cited."""
    lines = []
    for n, c, score in used:
        label = f"{c['lecture']}, {c['type']}, {c['locator']}"
        if c.get("kind") == "AI-generated page description":
            label += " (AI-generated description)"
        link = c.get("url", "")
        lines.append(f"**[{n}]** {label}" +
                     (f" - [open]({link})" if link.startswith("http") else "") +
                     f" _(match {score:.2f})_")
    return "### Sources\n" + "\n\n".join(lines)


# ---------- LangGraph: state and nodes ----------
class State(TypedDict, total=False):
    question: str      # the student's message
    pairs: list        # recent (user, assistant) turns
    route: str         # chat | knowledge | offtopic
    standalone: str    # question rewritten for search
    query: str         # current search query
    pool: dict         # retrieved passages collected across loops
    best: float        # best match score seen
    steps: list        # human-readable trace of the loop
    loop: int          # searches done
    next: str          # judge decision: retrieve | generate
    answer: str        # final reply


def router_node(state: State):
    """Decide: trigger the knowledge base, or let the chatbot answer directly."""
    q, pairs = state["question"], state.get("pairs", [])
    if SMALLTALK.match(q):
        route = "chat"
    elif KB_WORDS.search(q):
        route = "knowledge"
    else:
        convo = "\n".join(f"Student: {u}\nAssistant: {a[:200]}" for u, a in pairs[-2:])
        out = llm([{"role": "system", "content": ROUTER_PROMPT},
                   {"role": "user",
                    "content": f"Recent conversation:\n{convo or '(none)'}\n\nLatest message: {q}"}],
                  max_tokens=10, temperature=0)
        word = (out or "").upper()
        # unclear or LLM down -> knowledge base, the grounded and safe default
        route = "chat" if "CHAT" in word else "offtopic" if "OFFTOPIC" in word else "knowledge"
    print(f"router: {q!r} -> {route}")
    return {"route": route}


def chat_direct_node(state: State):
    """Chit-chat and conversation memory, answered by the chatbot itself."""
    msgs = [{"role": "system", "content": CHAT_PROMPT}]
    for u, a in state.get("pairs", []):
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": state["question"]})
    out = llm(msgs, max_tokens=300, temperature=0.5)
    return {"answer": f"{out or SMALLTALK_REPLY}\n\n_🧭 route: {ROUTE_LABEL['chat']}_"}


def decline_node(state: State):
    return {"answer": f"{OFFTOPIC_REPLY}\n\n_🧭 route: {ROUTE_LABEL['offtopic']}_"}


def rewrite_node(state: State):
    standalone = rewrite_query(state["question"], state.get("pairs", []))
    return {"standalone": standalone, "query": standalone}


def retrieve_node(state: State):
    query, pool = state["query"], dict(state.get("pool", {}))
    hits = retrieve(query)
    top = hits[0][1] if hits else 0.0
    for c, s in hits:
        if id(c) not in pool or s > pool[id(c)][1]:
            pool[id(c)] = (c, s)
    loop = state.get("loop", 0) + 1
    steps = list(state.get("steps", [])) + [f'search {loop}: "{query}" (best match {top:.2f})']
    return {"pool": pool, "best": max(state.get("best", 0.0), top),
            "steps": steps, "loop": loop}


def judge_node(state: State):
    """Loop control: are the passages enough, or should we search again?"""
    steps = list(state["steps"])
    if state["loop"] >= MAX_LOOPS:
        return {"next": "generate"}
    verdict = judge(state["standalone"], ranked_pool(state["pool"]))
    if verdict is None:
        return {"next": "generate", "steps": steps + ["judge unavailable"]}
    if verdict.get("sufficient") is True:
        return {"next": "generate", "steps": steps + ["judged sufficient"]}
    nq = str(verdict.get("next_query") or "").strip()
    if not nq or nq.lower() == state["query"].lower():
        return {"next": "generate", "steps": steps + ["judged insufficient, no better query"]}
    return {"next": "retrieve", "query": nq[:300],
            "steps": steps + ["judged insufficient, refining the search"]}


def generate_node(state: State):
    ranked, best = ranked_pool(state["pool"]), state.get("best", 0.0)
    trace = f"_🧭 route: {ROUTE_LABEL['knowledge']} → " + " → ".join(state["steps"]) + "_"
    print(f"question={state['question']!r} best_score={best:.3f} steps={state['steps']}")
    if not ranked or best < MIN_SCORE:
        return {"answer": (f"{NOT_COVERED}\n\n_(Best match score: {best:.2f}, "
                           f"threshold: {MIN_SCORE})_\n\n{trace}")}

    context = "\n\n".join(
        f"[{n}] ({c['lecture']} | {c['type']} | {c['locator']})\n{c['text']}"
        for n, (c, _) in enumerate(ranked, 1))
    text = llm([{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Context passages:\n{context}\n\nQuestion: {state['standalone']}"}])
    if text is None:  # fallback: retrieval only
        text = ("The AI answer service is unavailable right now. These are the most "
                "relevant passages from the lecture material:\n\n" +
                "\n\n".join(f"**[{n}]** {c['text'][:600]}..."
                            for n, (c, _) in enumerate(ranked, 1)))

    refused = "not covered in the lecture material" in text.lower()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    used = [(n, c, sc) for n, (c, sc) in enumerate(ranked, 1) if n in cited]
    if refused or not used:  # no citations = not grounded in the material: refuse
        return {"answer": f"{NOT_COVERED}\n\n{trace}"}
    return {"answer": f"{text}\n\n---\n{format_sources(used)}\n\n{trace}"}


# ---------- LangGraph: wiring ----------
builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("chat_direct", chat_direct_node)
builder.add_node("decline", decline_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("judge", judge_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", lambda s: s["route"],
                              {"chat": "chat_direct", "offtopic": "decline",
                               "knowledge": "rewrite"})
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "judge")
builder.add_conditional_edges("judge", lambda s: s["next"],
                              {"retrieve": "retrieve", "generate": "generate"})
for _name in ("chat_direct", "decline", "generate"):
    builder.add_edge(_name, END)
graph = builder.compile()


# ---------- Gradio chat (a generator: yields progress, then the final answer) ----------
def chat(message, history):
    question = _text(message).strip()
    if not question:
        yield "Please type a question."
        return
    state = {"question": question, "pairs": history_pairs(history),
             "pool": {}, "best": 0.0, "steps": [], "loop": 0}
    yield "🧭 Deciding how to answer..."
    answer = None
    for update in graph.stream(state, stream_mode="updates"):
        for node, out in (update or {}).items():
            out = out or {}
            if node == "router":
                yield f"🧭 Route: {ROUTE_LABEL.get(out.get('route'), '?')}..."
            elif node == "retrieve":
                yield f"🔎 {out['steps'][-1]}"
            elif node == "judge":
                yield "🧠 Checking whether these passages are enough..."
            elif node == "rewrite":
                yield "✍️ Preparing the answer..."
            if "answer" in out:
                answer = out["answer"]
    yield answer or "Sorry, something went wrong. Please try again."


# ---------- archive tab data ----------
archive_rows = [[s["lecture"], s["type"], s["file"],
                 f"[link]({s['url']})" if s["url"].startswith("http") else s["url"]]
                for s in sources]
archive_files = [str(Path("data") / s["file"]) for s in sources
                 if (Path("data") / s["file"]).exists()]

with gr.Blocks(title="DSA RAG - MIT 6.046J") as demo:
    gr.Markdown("# Ask the lecture: MIT 6.046J Dynamic Programming\n"
                "Chat with the lecture video transcript, lecture notes and recitation. "
                "Course answers come only from these sources, with citations.")
    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=chat,
            examples=["What is memoization?",
                      "What is the difference between top-down and bottom-up DP?",
                      "Explain that again with an example"],
            cache_examples=False,
        )
    with gr.Tab("Study material archive"):
        gr.Markdown("Every source in the knowledge base.")
        gr.Dataframe(value=archive_rows,
                     headers=["Lecture", "Type", "File", "Link"],
                     datatype=["str", "str", "str", "markdown"], interactive=False)
        gr.File(value=archive_files, label="Download source files", file_count="multiple")
    gr.Markdown("Source: MIT OpenCourseWare 6.046J (Spring 2015), Erik Demaine and "
                "Srinivas Devadas, licensed CC BY-NC-SA 4.0. "
                "[Course page](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/)")

if __name__ == "__main__":
    demo.launch(allowed_paths=["data"])