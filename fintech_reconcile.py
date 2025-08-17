from __future__ import annotations
import argparse
import itertools
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

# Optional libraries
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    HAS_RAPIDFUZZ = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# -----------------------------
# Utilities
# -----------------------------
def parse_amount_to_cents(x) -> Optional[int]:
    """Parse amounts like '1,234.56', '$99.00', '(12.34)' -> cents (int)."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x) * 100
    try:
        s = str(x).strip()
        if s == "":
            return None
        s = s.replace(",", "")
        for sym in ["$", "€", "£", "PKR", "Rs", "rs", "USD", "EUR"]:
            s = s.replace(sym, "")
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1]
        amt = float(s)
        cents = int(round(amt * 100))
        return -cents if neg else cents
    except Exception:
        return None

def cents_to_str(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{sign}{c//100}.{c%100:02d}"

def make_uid(prefix: str, idx: int) -> str:
    return f"{prefix}-{idx:06d}"


@dataclass
class MatchResult:
    target_id: str
    target_cents: int
    method: str  # 'direct','bruteforce','dp','ga','ga_approx','ml_ranked'
    transactions: List[str]
    amounts_cents: List[int]
    total_cents: int
    meta: Dict


# -----------------------------
# Load & Prepare (Part 1)
# -----------------------------
def load_and_prepare(transactions_path: str, targets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Transactions file (Customer_Ledger_Entries_FULL.xlsx)
    df1 = pd.read_excel(transactions_path, header=0, usecols=[8, 14])
    df1.columns = ["Description", "Amount"]

    # Targets file (KH_Bank.XLSX)
    df2 = pd.read_excel(targets_path, header=0, usecols=[2, 14])
    df2.columns = ["ReferenceID", "Target"]

    # Standardize & clean
    df1["Amount_cents"] = df1["Amount"].apply(parse_amount_to_cents)
    df2["Target_cents"] = df2["Target"].apply(parse_amount_to_cents)
    df1 = df1.dropna(subset=["Amount_cents"]).reset_index(drop=True)
    df2 = df2.dropna(subset=["Target_cents"]).reset_index(drop=True)

    # 🔥 For testing: only keep first 3 rows from each file
    df1 = df1.head(3).reset_index(drop=True)
    df2 = df2.head(3).reset_index(drop=True)

    # Unique IDs
    df1["TxnID"] = [make_uid("TXN", i) for i in range(len(df1))]
    df2["TargetID"] = [make_uid("TGT", i) for i in range(len(df2))]

    return df1[["TxnID", "Amount", "Amount_cents", "Description"]], \
           df2[["TargetID", "Target", "Target_cents", "ReferenceID"]]


# -----------------------------
# Direct matching (Part 2.1)
# -----------------------------
def direct_exact_matches(df1: pd.DataFrame, df2: pd.DataFrame, tolerance: int = 0) -> List[MatchResult]:
    """Direct 1:1 amount matches (with optional tolerance in cents)."""
    bucket: Dict[int, List[Tuple[str, int]]] = {}
    for row in df1.itertuples(index=False):
        bucket.setdefault(int(row.Amount_cents), []).append((row.TxnID, int(row.Amount_cents)))
    results: List[MatchResult] = []
    for t in df2.itertuples(index=False):
        tval = int(t.Target_cents)
        # quick path exact
        if tolerance == 0 and tval in bucket:
            for txn_id, amt in bucket[tval]:
                results.append(MatchResult(
                    target_id=t.TargetID, target_cents=tval, method="direct",
                    transactions=[txn_id], amounts_cents=[amt], total_cents=amt,
                    meta={"note": "exact amount match"}
                ))
        else:
            # near matches
            for a, txns in bucket.items():
                if abs(a - tval) <= tolerance:
                    for txn_id, amt in txns:
                        results.append(MatchResult(
                            target_id=t.TargetID, target_cents=tval, method="direct",
                            transactions=[txn_id], amounts_cents=[amt], total_cents=amt,
                            meta={"note": f"within tolerance {tolerance} cents", "diff": int(amt - tval)}
                        ))
    return results


# -----------------------------
# Subset Sum - Brute Force (Part 2.2)
# -----------------------------
def subset_sum_bruteforce(target_cents: int,
                          txns: List[Tuple[str, int]],
                          max_subset_size: int = 5,
                          tolerance: int = 0) -> Optional[List[Tuple[str, int]]]:
    for r in range(1, min(max_subset_size, len(txns)) + 1):
        for combo in itertools.combinations(txns, r):
            total = sum(a for _, a in combo)
            if abs(total - target_cents) <= tolerance:
                return list(combo)
    return None


# -----------------------------
# Subset Sum - Dynamic Programming (Part 3.2)
# -----------------------------
def subset_sum_dp(target_cents: int,
                  txns: List[Tuple[str, int]],
                  allow_negative: bool = False,
                  tolerance: int = 0) -> Optional[List[Tuple[str, int]]]:
    """Optimized DP subset-sum; supports tolerance by checking ±t band."""
    amounts = [a for _, a in txns]
    ids = [i for i, _ in txns]

    if not allow_negative and all(a >= 0 for a in amounts):
        max_sum = sum(a for a in amounts if a > 0)
        if target_cents < 0 or target_cents > max_sum + tolerance:
            return None

        # classical boolean DP with parent tracking
        dp = [-2] * (max_sum + 1)
        dp[0] = -1
        parent = [(-1, -1)] * (max_sum + 1)
        for idx, a in enumerate(amounts):
            for s in range(max_sum - a, -1, -1):
                if dp[s] != -2 and dp[s + a] == -2:
                    dp[s + a] = idx
                    parent[s + a] = (s, idx)

        # Check exact or tolerance window
        candidates = []
        for s in range(max(0, target_cents - tolerance), min(max_sum, target_cents + tolerance) + 1):
            if 0 <= s <= max_sum and dp[s] != -2:
                candidates.append(s)
        if not candidates:
            return None

        # pick s with smallest |s - target|
        s_star = min(candidates, key=lambda s: abs(s - target_cents))
        sol_indices = []
        s = s_star
        while s != 0:
            prev_s, idx = parent[s]
            if idx == -1:
                break
            sol_indices.append(idx)
            s = prev_s
        sol_indices.reverse()
        return [(ids[i], amounts[i]) for i in sol_indices]
    else:
        # meet negatives using set-expansion (can be heavy)
        dp: Dict[int, Optional[Tuple[int, int]]] = {0: None}
        for idx, a in enumerate(amounts):
            new_dp = dict(dp)
            for s in list(dp.keys()):
                ns = s + a
                if ns not in new_dp:
                    new_dp[ns] = (s, idx)
            dp = new_dp

        # check tolerance window
        candidates = [s for s in dp.keys() if abs(s - target_cents) <= tolerance]
        if not candidates:
            return None
        s_star = min(candidates, key=lambda s: abs(s - target_cents))

        path = []
        s = s_star
        while s != 0:
            prev_s, idx = dp[s]
            path.append(idx)
            s = prev_s
        path.reverse()
        return [(ids[i], amounts[i]) for i in path]


# -----------------------------
# Genetic Algorithm (Part 4.1)
# -----------------------------
def genetic_search(target_cents: int,
                   txns: List[Tuple[str, int]],
                   pop_size: int = 80,
                   generations: int = 200,
                   mutation_rate: float = 0.02,
                   elitism: int = 2,
                   tolerance: int = 0,
                   seed: Optional[int] = 42) -> Optional[List[Tuple[str, int]]]:
    rnd = random.Random(seed)
    n = len(txns)
    if n == 0:
        return None
    amounts = np.array([a for _, a in txns], dtype=np.int64)

    def fitness(mask: np.ndarray) -> int:
        total = int(amounts[mask == 1].sum())
        return abs(total - target_cents)

    population = np.zeros((pop_size, n), dtype=np.int8)
    for i in range(pop_size):
        bits = max(1, n // 20)
        idxs = rnd.sample(range(n), min(bits, n))
        population[i, idxs] = 1

    best_mask = None
    best_fit = 10**18

    for _ in range(generations):
        fits = np.array([fitness(population[i]) for i in range(pop_size)])
        order = np.argsort(fits)
        population = population[order]
        fits = fits[order]
        if fits[0] < best_fit:
            best_fit = int(fits[0])
            best_mask = population[0].copy()
            if best_fit <= tolerance:
                break

        # tournament selection
        def tournament():
            a, b = rnd.randrange(max(1, pop_size//2)), rnd.randrange(max(1, pop_size//2))
            return population[a] if fits[a] < fits[b] else population[b]

        # next generation
        next_pop = [population[i].copy() for i in range(min(elitism, pop_size))]
        while len(next_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            cx = rnd.randrange(1, n) if n > 1 else 0
            child = np.concatenate([p1[:cx], p2[cx:]]) if n > 0 else np.array([], dtype=np.int8)
            if n > 0:
                for j in range(n):
                    if rnd.random() < mutation_rate:
                        child[j] ^= 1
            next_pop.append(child)
        population = np.stack(next_pop[:pop_size], axis=0)

    if best_mask is None:
        return None
    chosen = np.where(best_mask == 1)[0].tolist()
    subset = [txns[i] for i in chosen]
    return subset


# -----------------------------
# Fuzzy Matching (Part 4.2)
# -----------------------------
def fuzzy_score(a: str, b: str) -> float:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(fuzz.token_set_ratio(str(a), str(b)))
    else:
        return 100.0 * difflib.SequenceMatcher(a=str(a), b=str(b)).ratio()

def fuzzy_match(df1: pd.DataFrame, df2: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    records = []
    for t in df2.itertuples(index=False):
        scores = []
        for s in df1.itertuples(index=False):
            sc = fuzzy_score(s.Description, t.ReferenceID)
            scores.append((s.TxnID, s.Description, sc, int(s.Amount_cents)))
        scores.sort(key=lambda x: x[2], reverse=True)
        for rank, (txn, desc, sc, amt) in enumerate(scores[:top_k], 1):
            records.append({
                "TargetID": t.TargetID,
                "ReferenceID": t.ReferenceID,
                "TxnID": txn,
                "TxnDescription": desc,
                "FuzzyScore": sc,
                "TxnAmount": cents_to_str(int(amt))
            })
    return pd.DataFrame.from_records(records)


# -----------------------------
# Reconciliation Orchestrator (Parts 2 & 3)
# -----------------------------
def reconcile(df1: pd.DataFrame,
              df2: pd.DataFrame,
              max_subset_size: int = 5,
              use_dp: bool = True,
              use_ga: bool = False,
              tolerance: int = 0) -> List[MatchResult]:
    results: List[MatchResult] = []

    # Direct 1:1 matches
    results.extend(direct_exact_matches(df1, df2, tolerance=tolerance))

    # Prepare list once
    txns_list = list(zip(df1["TxnID"].tolist(), df1["Amount_cents"].astype(int).tolist()))
    allow_neg = any(a < 0 for _, a in txns_list)

    for t in df2.itertuples(index=False):
        target = int(t.Target_cents)

        # Brute force (small combos)
        bf = subset_sum_bruteforce(target, txns_list, max_subset_size=max_subset_size, tolerance=tolerance)
        if bf is not None:
            results.append(MatchResult(
                target_id=t.TargetID, target_cents=target, method="bruteforce",
                transactions=[i for i, _ in bf],
                amounts_cents=[a for _, a in bf],
                total_cents=sum(a for _, a in bf),
                meta={"k": len(bf), "diff": int(sum(a for _, a in bf) - target)}
            ))
            continue

        # Dynamic programming (optimized)
        if use_dp:
            dp = subset_sum_dp(target, txns_list, allow_negative=allow_neg, tolerance=tolerance)
            if dp is not None:
                results.append(MatchResult(
                    target_id=t.TargetID, target_cents=target, method="dp",
                    transactions=[i for i, _ in dp],
                    amounts_cents=[a for _, a in dp],
                    total_cents=sum(a for _, a in dp),
                    meta={"n": len(dp), "diff": int(sum(a for _, a in dp) - target)}
                ))
                continue

        # Genetic Algorithm (approximate or exact)
        if use_ga:
            ga = genetic_search(target, txns_list, tolerance=tolerance)
            if ga is not None:
                total = sum(a for _, a in ga)
                results.append(MatchResult(
                    target_id=t.TargetID, target_cents=target,
                    method="ga" if abs(total - target) <= tolerance else "ga_approx",
                    transactions=[i for i, _ in ga],
                    amounts_cents=[a for _, a in ga],
                    total_cents=total,
                    meta={"n": len(ga), "diff": int(total - target)}
                ))
    return results


# -----------------------------
# Benchmarking & Visualization (Part 5)
# -----------------------------
def ml_rank_candidates(param, param1, model, max_k, top_n):
    pass

def generate_ml_dataset(param, param1, max_k, random_negatives):
    pass

def train_ml_model(Xy, model_type):
    pass

def benchmark(df1: pd.DataFrame, df2: pd.DataFrame,
              sizes: Iterable[int] = (25, 50, 100, 200),
              repeats: int = 1,
              tolerance: int = 0,
              include_ga: bool = True,
              include_ml: bool = True) -> pd.DataFrame:
    rows = []
    base_txns = list(zip(df1["TxnID"], df1["Amount_cents"].astype(int)))
    targets = df2["Target_cents"].astype(int).tolist() or [0]
    allow_neg = any(a < 0 for _, a in base_txns)

    for n in sizes:
        txns = base_txns[:min(n, len(base_txns))]
        for r in range(repeats):
            t = targets[r % len(targets)]

            # Brute force
            start = time.perf_counter()
            _ = subset_sum_bruteforce(t, txns, max_subset_size=5, tolerance=tolerance)
            bf_time = time.perf_counter() - start

            # Dynamic programming
            start = time.perf_counter()
            _ = subset_sum_dp(t, txns, allow_negative=allow_neg, tolerance=tolerance)
            dp_time = time.perf_counter() - start

            # Genetic Algorithm
            ga_time = None
            if include_ga:
                start = time.perf_counter()
                _ = genetic_search(t, txns, tolerance=tolerance, generations=50)
                ga_time = time.perf_counter() - start

            # Machine Learning
            ml_train_time, ml_infer_time = None, None
            if include_ml:
                try:
                    # Generate dataset
                    start = time.perf_counter()
                    Xy, _ = generate_ml_dataset(pd.DataFrame(txns, columns=["TxnID", "Amount_cents"]),
                                                df2.head(1), max_k=3, random_negatives=20)
                    if not Xy.empty:
                        model, _ = train_ml_model(Xy, model_type="rf")
                        ml_train_time = time.perf_counter() - start

                        # Inference timing
                        start = time.perf_counter()
                        _ = ml_rank_candidates(df1.head(min(n, len(df1))),
                                               df2.head(1),
                                               model,
                                               max_k=3,
                                               top_n=3)
                        ml_infer_time = time.perf_counter() - start
                except Exception:
                    pass

            rows.append({
                "TxnCount": len(txns),
                "Target": t,
                "BruteForce_s": bf_time,
                "DP_s": dp_time,
                "GA_s": ga_time,
                "ML_train_s": ml_train_time,
                "ML_infer_s": ml_infer_time
            })

    return pd.DataFrame(rows)

def save_benchmark_plots(bench_df: pd.DataFrame, out_dir: str) -> List[str]:
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    if bench_df is None or bench_df.empty:
        return paths

    # Plot runtimes
    plt.figure()
    g = bench_df.groupby("TxnCount").mean().reset_index()
    for col, label in [
        ("BruteForce_s", "Brute Force"),
        ("DP_s", "Dynamic Programming"),
        ("GA_s", "Genetic Algorithm"),
        ("ML_train_s", "ML Train"),
        ("ML_infer_s", "ML Inference"),
    ]:
        if col in g.columns and g[col].notna().any():
            plt.plot(g["TxnCount"], g[col], marker="o", label=label)

    plt.xlabel("Transaction count")
    plt.ylabel("Time (s, log scale)")
    plt.yscale("log")   # useful to show differences
    plt.title("Algorithm Runtime Comparison")
    plt.legend()
    p = os.path.join(out_dir, "bench_all.png")
    plt.savefig(p, bbox_inches="tight")
    paths.append(p)

    return paths


# -----------------------------
# ML Feature Engineering & Models (Part 3.1 & 3.3)
# -----------------------------
def _subset_stats(amounts: List[int]) -> Dict[str, float]:
    if not amounts:
        return {"n": 0, "sum": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "neg_ratio": 0.0, "abs_sum": 0}
    arr = np.array(amounts, dtype=float)
    return {
        "n": float(len(arr)),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "neg_ratio": float((arr < 0).mean()),
        "abs_sum": float(np.abs(arr).sum()),
    }

def _target_subset_features(target_row, subset: List[Tuple[str, int]], txn_map: Dict[str, Dict]) -> Dict[str, float]:
    target = int(target_row.Target_cents)
    ref = str(target_row.ReferenceID) if not pd.isna(target_row.ReferenceID) else ""
    amounts = [a for _, a in subset]
    stats = _subset_stats(amounts)
    sum_subset = int(round(stats["sum"]))
    diff = sum_subset - target
    # fuzzy to best txn desc in subset
    best_fuzzy = 0.0
    for txn_id, _ in subset:
        desc = txn_map[txn_id]["Description"]
        best_fuzzy = max(best_fuzzy, fuzzy_score(desc, ref))
    return {
        **stats,
        "target": float(target),
        "diff": float(diff),
        "abs_diff": float(abs(diff)),
        "best_fuzzy": float(best_fuzzy),
    }

def generate_ml_dataset_regression(df1: pd.DataFrame, df2: pd.DataFrame,
                                   max_k: int = 3,
                                   random_samples: int = 200,
                                   seed: int = 123) -> pd.DataFrame:
    """Generate regression dataset: label = abs(sum(subset) - target)."""
    rnd = random.Random(seed)
    txns_list = list(zip(df1["TxnID"].tolist(), df1["Amount_cents"].astype(int).tolist()))
    txn_map = {r.TxnID: {"Amount": int(r.Amount_cents), "Description": r.Description}
               for r in df1.itertuples(index=False)}

    rows = []
    for t in df2.itertuples(index=False):
        target = int(t.Target_cents)
        ids = [i for i, _ in txns_list]
        for _ in range(random_samples):
            k = rnd.randint(1, max_k)
            subset_ids = rnd.sample(ids, min(k, len(ids))) if ids else []
            subset = [(sid, txn_map[sid]["Amount"]) for sid in subset_ids]
            feat = _target_subset_features(t, subset, txn_map)
            feat["label"] = abs(int(feat["sum"]) - target)  # regression label
            rows.append(feat)
    return pd.DataFrame(rows)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_ml_model_regression(Xy: pd.DataFrame, model_type: str = "rf") -> Tuple[object, Dict[str, float]]:
    y = Xy["label"].astype(float).values
    X = Xy.drop(columns=["label"]).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return model, {"MAE": float(mae), "n_train": len(X_train), "n_test": len(X_test)}

def ml_rank_candidates_regression(df1: pd.DataFrame, df2: pd.DataFrame, model,
                                   max_k: int = 3, top_n: int = 3) -> pd.DataFrame:
    txn_map = {r.TxnID: {"Amount": int(r.Amount_cents), "Description": r.Description}
               for r in df1.itertuples(index=False)}
    txns_list = list(zip(df1["TxnID"].tolist(), df1["Amount_cents"].astype(int).tolist()))
    candidates = []
    max_combos = 2000

    for t in df2.itertuples(index=False):
        combos = []
        for k in range(1, max_k + 1):
            allk = list(itertools.combinations(txns_list, k))
            random.shuffle(allk)
            combos.extend(allk[:max(0, max_combos - len(combos))])

        if not combos:
            continue

        feat_rows, subset_texts = [], []
        for subset in combos:
            f = _target_subset_features(t, list(subset), txn_map)
            feat_rows.append(f)
            ids = [sid for sid, _ in subset]
            amts = [txn_map[sid]["Amount"] for sid in ids]
            subset_texts.append((", ".join(ids), ", ".join(cents_to_str(a) for a in amts), int(sum(amts))))

        X = pd.DataFrame(feat_rows)
        preds = model.predict(X.values)
        X["predicted_error"] = preds
        X["TargetID"] = t.TargetID
        X["Target"] = int(t.Target_cents)
        X["TxnIDs"] = [s[0] for s in subset_texts]
        X["TxnAmounts"] = [s[1] for s in subset_texts]
        X["SubsetSum"] = [s[2] for s in subset_texts]

        X = X.sort_values("predicted_error", ascending=True).head(top_n)
        candidates.append(X)

    return pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()

# -----------------------------
# Reporting (Part 5.2)
# -----------------------------
def build_report(matches: List[MatchResult],
                 fuzzy_df: Optional[pd.DataFrame],
                 bench_df: Optional[pd.DataFrame],
                 ml_info: Optional[Dict],
                 ml_candidates: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    recs = []
    for m in matches:
        recs.append({
            "TargetID": m.target_id,
            "Target": cents_to_str(m.target_cents),
            "Method": m.method,
            "TxnIDs": ", ".join(m.transactions),
            "TxnAmounts": ", ".join(cents_to_str(a) for a in m.amounts_cents),
            "Total": cents_to_str(m.total_cents),
            "Meta": str(m.meta)
        })
    match_df = pd.DataFrame(recs) if recs else pd.DataFrame(columns=[
        "TargetID", "Target", "Method", "TxnIDs", "TxnAmounts", "Total", "Meta"
    ])
    sheets = {"Matches": match_df}
    if fuzzy_df is not None and not fuzzy_df.empty:
        sheets["FuzzySuggestions"] = fuzzy_df
    if bench_df is not None and not bench_df.empty:
        sheets["Benchmark"] = bench_df
    if ml_info is not None:
        sheets["ML_Info"] = pd.DataFrame([ml_info])
    if ml_candidates is not None and not ml_candidates.empty:
        tmp = ml_candidates.copy()
        if "Target" in tmp.columns:
            tmp["Target"] = tmp["Target"].apply(lambda c: cents_to_str(int(c)))
        if "SubsetSum" in tmp.columns:
            tmp["SubsetSum"] = tmp["SubsetSum"].apply(lambda c: cents_to_str(int(c)))
        sheets["ML_Candidates"] = tmp
    return sheets

def save_report_xlsx(sheets: Dict[str, pd.DataFrame], path: str) -> None:
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


# -----------------------------
# CLI
# -----------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Excel reconciliation toolkit (two-file, ML, GA, Viz)")
    p.add_argument("--transactions", required=True, help="Excel file with transactions: Amount, Description (cols A,B)")
    p.add_argument("--targets", required=True, help="Excel file with target amounts: Target, ReferenceID (cols A,B)")

    # Core options
    p.add_argument("--max-subset-size", type=int, default=5, help="Max subset size for brute-force")
    p.add_argument("--no-dp", action="store_true", help="Disable dynamic programming subset search")
    p.add_argument("--ga", action="store_true", help="Enable genetic algorithm fallback")
    p.add_argument("--fuzzy", action="store_true", help="Compute fuzzy suggestions (top 3 per target)")
    p.add_argument("--tolerance", type=int, default=0, help="Tolerance in cents for near matches (e.g., 100 = ±1.00)")
    p.add_argument("--report", default="reconcile_report.xlsx", help="Path to save the Excel report")

    # Benchmarking & plots
    p.add_argument("--benchmark", action="store_true", help="Run brute force vs DP timing comparison")
    p.add_argument("--plots-dir", default=None, help="Directory to save benchmark plots (PNG)")

    # ML options
    p.add_argument("--ml", action="store_true", help="Enable ML pipeline (feature engineering, model training, ranking)")
    p.add_argument("--ml-model", default="logreg", choices=["logreg", "rf"], help="ML model type")
    p.add_argument("--ml-max-k", type=int, default=3, help="Max subset size for ML candidate subsets")
    p.add_argument("--ml-negatives", type=int, default=200, help="Random negatives per target for ML training")

    args = p.parse_args(argv)

    # Load & prepare
    df1, df2 = load_and_prepare(args.transactions, args.targets)

    # Reconcile (Part 2 + 3 + 4)
    matches = reconcile(df1, df2,
                        max_subset_size=args.max_subset_size,
                        use_dp=not args.no_dp,
                        use_ga=args.ga,
                        tolerance=args.tolerance)

    # Fuzzy (Part 4.2)
    fuzzy_df = fuzzy_match(df1, df2, top_k=3) if args.fuzzy else None

    # Benchmark + plots (Part 5.1 & 5.2)
    bench_df = benchmark(df1, df2, sizes=(5, 10, 20, 50, 100, 200, 500), repeats=5,
                         tolerance=args.tolerance) if args.benchmark else None
    if args.plots_dir and bench_df is not None and not bench_df.empty:
        paths = save_benchmark_plots(bench_df, args.plots_dir)
        print("Saved benchmark plots:")
        for pth in paths:
            print("  ", pth)

    # ML pipeline (Part 3.1 & 3.3)
    ml_info = None
    ml_candidates = None
    if args.ml:
        if not HAS_SKLEARN:
            raise RuntimeError("--ml requires scikit-learn. Install via: pip install scikit-learn")
        Xy = generate_ml_dataset_regression(df1, df2, max_k=args.ml_max_k, random_samples=args.ml_negatives)
        if Xy.empty:
            print("[ML] Dataset empty. Skipping ML.")
        else:
            model, info = train_ml_model_regression(Xy, model_type=args.ml_model)
            ml_info = {**info, "rows": int(len(Xy)), "features": list(Xy.drop(columns=["label"]).columns)}
            ml_candidates = ml_rank_candidates_regression(df1, df2, model, max_k=args.ml_max_k, top_n=3)

    # Build & save report (Part 5.2)
    sheets = build_report(matches, fuzzy_df, bench_df, ml_info, ml_candidates)
    save_report_xlsx(sheets, args.report)

    print(f"Saved report with {len(matches)} match rows -> {args.report}")
    if args.fuzzy:
        print("Included fuzzy suggestions sheet.")
    if args.benchmark:
        print("Included benchmark sheet and (if specified) plots.")
    if args.ml and ml_info is not None:
        print("Included ML info and ranked candidate subsets.")
    print("Done.")


if __name__ == "__main__":
    main()

