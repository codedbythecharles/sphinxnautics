#!/usr/bin/env python3
# pip install annoy sentence-transformers datasets

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from annoy import AnnoyIndex

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ---------------- Utilities ----------------

_WS_RE = re.compile(r"\s+")

def normalize_text(txt: str) -> str:
    txt = (txt or "").lower()
    txt = re.sub(r"[^\w\s]", " ", txt)    # drop punctuation
    txt = _WS_RE.sub(" ", txt).strip()
    return txt

def chunk_text(text: str, max_tokens: int = 128, overlap: int = 32) -> List[str]:
    tokens = text.split()
    if not tokens:
        return [text]
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

def mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return [x / len(vectors) for x in acc]

def l2norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x*x for x in vec))
    return [x/s for x in vec] if s > 0 else vec

def text_hash(txt: str) -> str:
    return hashlib.sha256(normalize_text(txt).encode("utf-8")).hexdigest()

# ---------------- ProblemIndex ----------------

class ProblemIndex:
    def __init__(
        self,
        embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        do_chunk: bool = False,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        metric: str = "angular",
    ):
        self.do_chunk = do_chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metric = metric

        if embedder is not None:
            self._embed = embedder
            self.model = None
        else:
            if SentenceTransformer is None:
                raise ImportError("Install sentence-transformers or pass a custom embedder.")
            self.model = SentenceTransformer(model_name)
            self._embed = lambda texts: self.model.encode(texts, show_progress_bar=False).tolist()

        self.docs: Dict[str, str] = {}
        self.ids: List[str] = []
        self.embeddings: Dict[str, List[float]] = {}

        self.annoy_index: Optional[AnnoyIndex] = None
        self._id2ann: Dict[int, str] = {}
        self._ann2id: Dict[str, int] = {}
        self._dim: int = 0
        self.model_name = model_name

    def add_documents(self, corpus: List[str], ids: Optional[List[str]] = None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(corpus))]
        assert len(corpus) == len(ids)
        for text, pid in zip(corpus, ids):
            normed = normalize_text(text)
            self.docs[pid] = normed
            self.ids.append(pid)

    def _embed_doc(self, txt: str) -> List[float]:
        if self.do_chunk:
            chunks = chunk_text(txt, self.chunk_size, self.chunk_overlap)
            vecs = self._embed(chunks)
            vec = mean_pool(vecs)
        else:
            vec = self._embed([txt])[0]
        return l2norm(vec)

    def build_annoy(self, n_trees: int = 50):
        # embed all docs
        for pid in self.ids:
            self.embeddings[pid] = self._embed_doc(self.docs[pid])

        any_vec = next(iter(self.embeddings.values())) if self.embeddings else [0.0]
        self._dim = len(any_vec)
        self.annoy_index = AnnoyIndex(self._dim, self.metric)
        self._id2ann.clear()
        self._ann2id.clear()

        for i, pid in enumerate(self.ids):
            self.annoy_index.add_item(i, self.embeddings[pid])
            self._id2ann[i] = pid
            self._ann2id[pid] = i

        self.annoy_index.build(n_trees)

    @staticmethod
    def _angular_to_cosine(dist: float) -> float:
        # for 'angular' metric, dist ≈ sqrt(2*(1 - cos))
        return 1.0 - (dist * dist) / 2.0

    def query_batch(self, queries: List[str], threshold: float = 0.9, top_k: int = 10):
        """Return list of tuples:
           (is_match: bool, best_source_pid: str|None, best_sim: float, best_source_pos: int)
        """
        if self.annoy_index is None:
            raise RuntimeError("Call build_annoy() or load() first.")

        if self.do_chunk:
            vecs = [ self._embed_doc(normalize_text(q or "")) for q in queries ]
        else:
            q_norm = [normalize_text(q or "") for q in queries]
            vecs = self._embed(q_norm)

        results = []
        for q_vec in vecs:
            qv = l2norm(q_vec)
            ids, dists = self.annoy_index.get_nns_by_vector(qv, top_k, include_distances=True)

            best_id, best_sim, best_pos = None, -1.0, -1
            for internal_id, dist in zip(ids, dists):
                cos_sim = self._angular_to_cosine(dist)
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    best_id = self._id2ann[internal_id]
                    best_pos = internal_id

            results.append((best_sim >= threshold, best_id, float(best_sim), int(best_pos)))
        return results

    # -------- persistence --------
    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "metric": self.metric,
            "dim": self._dim,
            "ids": self.ids,
            "docs": self.docs,
            "model_name": self.model_name,
            "do_chunk": self.do_chunk,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        if self.annoy_index is None:
            raise RuntimeError("No Annoy index to save. Call build_annoy() first.")
        self.annoy_index.save(str(out_dir / "index.ann"))

    @classmethod
    def load(cls, in_dir: Path):
        meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
        pi = cls(
            embedder=None,
            model_name=meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            do_chunk=meta.get("do_chunk", False),
            chunk_size=meta.get("chunk_size", 128),
            chunk_overlap=meta.get("chunk_overlap", 32),
            metric=meta.get("metric", "angular"),
        )
        pi.docs = meta["docs"]
        pi.ids = meta["ids"]
        pi._dim = meta["dim"]
        pi.annoy_index = AnnoyIndex(pi._dim, pi.metric)
        pi.annoy_index.load(str(in_dir / "index.ann"))
        pi._id2ann = {i: pid for i, pid in enumerate(pi.ids)}
        pi._ann2id = {pid: i for i, pid in enumerate(pi.ids)}
        return pi

# ---------------- dataset helpers ----------------

CANDIDATE_DESC_FIELDS = ["description", "statement", "problem", "text"]

def load_any_dataset(path_or_name: str, split: Optional[str] = None) -> Dataset:
    # 1) try load_from_disk
    try:
        ds_any = load_from_disk(path_or_name)
        if isinstance(ds_any, DatasetDict):
            if split is None:
                raise ValueError("DatasetDict detected; provide --split.")
            return ds_any[split]
        return ds_any
    except Exception:
        pass
    # 2) hub
    if split is None:
        raise ValueError("For hub datasets, provide --split.")
    return load_dataset(path_or_name, split=split)

def detect_field(ds: Dataset, candidates: List[str]) -> str:
    cols = set(ds.column_names)
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"None of {candidates} found in dataset columns: {ds.column_names}")

def detect_id_field(ds: Dataset) -> str:
    cols = set(ds.column_names)
    if "problem_id" in cols: return "problem_id"
    if "id" in cols: return "id"
    raise ValueError("No 'problem_id' or 'id' column in dataset.")

# ---------------- CLI ops ----------------

def build_index(args):
    ds = load_any_dataset(args.ref_ds, args.ref_split)
    id_field = args.id_field or detect_id_field(ds)
    desc_field = args.desc_field or detect_field(ds, CANDIDATE_DESC_FIELDS)
    print(f"[build] Using id_field='{id_field}', desc_field='{desc_field}', size={len(ds)}")

    ids = [str(x) for x in ds[id_field]]
    texts = [x if isinstance(x, str) else "" for x in ds[desc_field]]

    index = ProblemIndex(
        model_name=args.model_name,
        do_chunk=args.do_chunk,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        metric="angular",
    )
    index.add_documents(texts, ids)
    index.build_annoy(n_trees=args.n_trees)
    index.save(Path(args.out_index))
    print(f"[build] Saved index to {args.out_index}")

    # Augment and optionally save the source dataset with hash and annoy position
    desc_hash = [text_hash(t) for t in texts]
    annoy_pos = list(range(len(ids)))
    ds = ds.add_column("desc_hash", desc_hash)
    ds = ds.add_column("annoy_pos", annoy_pos)

    if args.ref_out:
        outp = Path(args.ref_out)
        ds.save_to_disk(outp)
        print(f"[build] Saved augmented source dataset with hash/pos to {outp}")

def compare_dataset(args):
    index = ProblemIndex.load(Path(args.index_dir))
    ds = load_any_dataset(args.query_ds, args.query_split)
    id_field = args.id_field or detect_id_field(ds)
    desc_field = args.desc_field or detect_field(ds, CANDIDATE_DESC_FIELDS)
    print(f"[compare] Using id_field='{id_field}', desc_field='{desc_field}', size={len(ds)}")

    batch = args.batch
    dup_match, dup_top_id, dup_top_score, dup_top_pos = [], [], [], []

    buf = []
    for i in range(len(ds)):
        buf.append(ds[i][desc_field])
        if len(buf) == batch or i == len(ds) - 1:
            results = index.query_batch(buf, threshold=args.threshold, top_k=args.top_k)
            for (is_dup, top_id, score, pos) in results:
                dup_match.append(bool(is_dup))
                dup_top_id.append(top_id)
                dup_top_score.append(float(score))
                dup_top_pos.append(int(pos))
            buf.clear()

    ds = ds.add_column("dup_match", dup_match)
    ds = ds.add_column("dup_top_id", dup_top_id)       # best matching source pid
    ds = ds.add_column("dup_top_score", dup_top_score) # highest cosine similarity
    ds = ds.add_column("dup_top_pos", dup_top_pos)     # 0-based Annoy/source index

    out_path = Path(args.out_ds)
    ds.save_to_disk(out_path)
    print(f"[compare] Saved annotated dataset to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Build and compare problem-description index with Annoy.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build
    b = sub.add_parser("build", help="Build an Annoy index from a source dataset.")
    b.add_argument("--ref_ds", required=True, help="path to load_from_disk dataset or HF hub name")
    b.add_argument("--ref_split", default=None, help="split if loading from hub")
    b.add_argument("--id_field", default=None)
    b.add_argument("--desc_field", default=None)
    b.add_argument("--out_index", required=True, help="output directory for the Annoy index")
    b.add_argument("--ref_out", default=None, help="optional: save augmented source dataset with hash/pos")
    b.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    b.add_argument("--do_chunk", action="store_true")
    b.add_argument("--chunk_size", type=int, default=128)
    b.add_argument("--chunk_overlap", type=int, default=32)
    b.add_argument("--n_trees", type=int, default=50)

    # compare
    c = sub.add_parser("compare", help="Compare a target dataset against a saved index.")
    c.add_argument("--index_dir", required=True, help="directory produced by 'build'")
    c.add_argument("--query_ds", required=True, help="path to load_from_disk dataset or HF hub name")
    c.add_argument("--query_split", default=None, help="split if loading from hub")
    c.add_argument("--id_field", default=None)
    c.add_argument("--desc_field", default=None)
    c.add_argument("--threshold", type=float, default=0.90)
    c.add_argument("--top_k", type=int, default=10)
    c.add_argument("--batch", type=int, default=512)
    c.add_argument("--out_ds", required=True, help="output directory for annotated dataset")

    args = ap.parse_args()
    if args.cmd == "build":
        build_index(args)
    else:
        compare_dataset(args)

if __name__ == "__main__":
    main()
