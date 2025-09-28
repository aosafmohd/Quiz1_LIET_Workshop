
"""
string_similarity.py
Simple string similarity + alignment report for two strings (6-10 chars).
No external dependencies.
"""

from difflib import SequenceMatcher

def validate_string(s):
    if not (6 <= len(s) <= 10):
        raise ValueError("String length must be between 6 and 10 characters.")
    return s

# -----------------------------
# Levenshtein (edit) distance
# -----------------------------
def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[n][m]

def similarity_from_levenshtein(a: str, b: str) -> float:
    dist = levenshtein_distance(a, b)
    denom = max(len(a), len(b))
    if denom == 0:
        return 100.0
    return (1 - dist / denom) * 100.0

# -----------------------------
# Needleman-Wunsch style global alignment
# (we'll use: match=+1, mismatch=0, gap=-1)
# -----------------------------
def global_alignment(a: str, b: str, match=1, mismatch=0, gap=-1):
    n, m = len(a), len(b)
    # score matrix
    score = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        score[i][0] = score[i-1][0] + gap
    for j in range(1, m+1):
        score[0][j] = score[0][j-1] + gap
    # fill
    for i in range(1, n+1):
        for j in range(1, m+1):
            diag = score[i-1][j-1] + (match if a[i-1]==b[j-1] else mismatch)
            up = score[i-1][j] + gap
            left = score[i][j-1] + gap
            score[i][j] = max(diag, up, left)

    # traceback (build aligned strings)
    i, j = n, m
    aligned_a = []
    aligned_b = []
    while i > 0 or j > 0:
        cur = score[i][j]
        # options (large negative when out of range)
        diag = score[i-1][j-1] + (match if i>0 and j>0 and a[i-1]==b[j-1] else mismatch) if i>0 and j>0 else -10**9
        up = score[i-1][j] + gap if i>0 else -10**9
        left = score[i][j-1] + gap if j>0 else -10**9

        # Prefer diagonal when it matches the current score (deterministic tie-breaking)
        if i>0 and j>0 and cur == score[i-1][j-1] + (match if a[i-1]==b[j-1] else mismatch):
            aligned_a.append(a[i-1]); aligned_b.append(b[j-1])
            i -= 1; j -= 1
        elif i > 0 and cur == up:
            aligned_a.append(a[i-1]); aligned_b.append('-')
            i -= 1
        else:
            aligned_a.append('-'); aligned_b.append(b[j-1])
            j -= 1

    aligned_a = ''.join(reversed(aligned_a))
    aligned_b = ''.join(reversed(aligned_b))

    # build match visualization and detailed report
    match_line = ''.join('|' if aligned_a[k] == aligned_b[k] and aligned_a[k] != '-' else ' '
                         for k in range(len(aligned_a)))

    # Count matches and produce per-position report with original indices
    matches = []
    mismatches = []
    gaps = []
    idx_a = 0
    idx_b = 0
    for k in range(len(aligned_a)):
        ca = aligned_a[k]; cb = aligned_b[k]
        pos_a = idx_a if ca != '-' else None
        pos_b = idx_b if cb != '-' else None

        if ca != '-':
            idx_a += 1
        if cb != '-':
            idx_b += 1

        if ca != '-' and cb != '-' and ca == cb:
            matches.append({'pos_a': pos_a, 'char_a': ca, 'pos_b': pos_b, 'char_b': cb})
        elif ca != '-' and cb != '-' and ca != cb:
            mismatches.append({'pos_a': pos_a, 'char_a': ca, 'pos_b': pos_b, 'char_b': cb})
        else:
            # one side has a gap
            if ca == '-':
                gaps.append({'side': 'b_has_extra', 'pos_b': pos_b, 'char_b': cb})
            else:
                gaps.append({'side': 'a_has_extra', 'pos_a': pos_a, 'char_a': ca})

    match_count = len(matches)
    alignment_length = sum(1 for k in range(len(aligned_a)) if not (aligned_a[k]=='-' and aligned_b[k]=='-'))
    similarity_pct = match_count / max(len(a), len(b)) * 100.0

    return {
        'aligned_a': aligned_a,
        'aligned_b': aligned_b,
        'match_line': match_line,
        'matches': matches,
        'mismatches': mismatches,
        'gaps': gaps,
        'match_count': match_count,
        'alignment_length': alignment_length,
        'similarity_pct': similarity_pct,
        'score': score[n][m]
    }

# -----------------------------
# Utility: quick difflib ratio
# -----------------------------
def difflib_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100.0

# -----------------------------
# Example interactive main
# -----------------------------
def main():
    print("Enter two strings (6-10 chars each). Comparison is case-insensitive by default.")
    s1 = validate_string(input("String A: ").strip())
    s2 = validate_string(input("String B: ").strip())

    # choose case sensitivity
    case_sensitive = False
    if not case_sensitive:
        a = s1.lower(); b = s2.lower()
    else:
        a = s1; b = s2

    print("\n--- Levenshtein-based similarity ---")
    lev_sim = similarity_from_levenshtein(a, b)
    print(f"Levenshtein similarity: {lev_sim:.2f}% (distance = {levenshtein_distance(a,b)})")

    print("\n--- Alignment-based similarity & match report ---")
    report = global_alignment(a, b)
    print(report['aligned_a'])
    print(report['match_line'])
    print(report['aligned_b'])
    print(f"Matches: {report['match_count']}, Alignment length: {report['alignment_length']}")
    print(f"Similarity (matches / max_len): {report['similarity_pct']:.2f}%")
    print("\nMatched characters (with indices):")
    for m in report['matches']:
        print(f" A[{m['pos_a']}]='{m['char_a']}'  <==>  B[{m['pos_b']}]='{m['char_b']}'")
    if report['mismatches']:
        print("\nMismatches:")
        for mm in report['mismatches']:
            print(f" A[{mm['pos_a']}]='{mm['char_a']}'  vs  B[{mm['pos_b']}]='{mm['char_b']}'")
    if report['gaps']:
        print("\nGaps / insertions:")
        for g in report['gaps']:
            if g['side'] == 'b_has_extra':
                print(f" B has extra char B[{g['pos_b']}]='{g['char_b']}'")
            else:
                print(f" A has extra char A[{g['pos_a']}]='{g['char_a']}'")

    print("\n--- difflib quick ratio (for reference) ---")
    print(f"difflib ratio: {difflib_ratio(a,b):.2f}%")

if __name__ == "__main__":
    main()
