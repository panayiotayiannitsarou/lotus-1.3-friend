# -*- coding: utf-8 -*-
from typing import List, Dict, Set, Optional
import pandas as pd, re, ast

# ✅ Βασικοί τίτλοι που κρατάμε σε κάθε minimal export
CORE_COLUMNS_DEFAULT = [
    "ΟΝΟΜΑ", "ΦΥΛΟ", "ΖΩΗΡΟΣ", "ΙΔΙΑΙΤΕΡΟΤΗΤΑ",
    "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ", "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ",
    "ΦΙΛΟΙ", "ΣΥΓΚΡΟΥΣΗ"
]

COLUMN_FIXES = {
    "ΖΩΗΡΟΙ": "ΖΩΗΡΟΣ",
    "ΙΔΙΑΙΤΕΡΟΤΗΤΕΣ": "ΙΔΙΑΙΤΕΡΟΤΗΤΑ",
    "ΠΑΙΔΙ ΕΚΠΑΙΔΕΥΤΙΚΟΥ": "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ",
    "ΟΝΟΜΑΤΕΠΩΝΥΜΟ": "ΟΝΟΜΑ",
    "ΣΥΓΚΡΟΥΣΗ/CONFLICT": "ΣΥΓΚΡΟΥΣΗ",
    "ΓΝΩΣΗ ΕΛΛ.": "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ",
    "ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ": "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ",
}

SAFE_SEP = re.compile(r"[,\|\;/·\n]+")

def norm_yesno(val: object) -> str:
    s = str(val).strip().upper()
    return "Ν" if s in {"Ν","ΝΑΙ","YES","TRUE","1","Y","Τ","ΑΙΣ","NAI"} else "Ο"

def norm_behavior(val: object) -> str:
    """
    Κανονικοποίηση της στήλης ΖΩΗΡΟΣ.

    Ο  = χωρίς αυξημένη συμπεριφορική ανάγκη
    Ν1 = ηπιότερη ανάγκη υποστήριξης χωρίς συνοδό
    Ν  = εντονότερη ανάγκη υποστήριξης χωρίς συνοδό
    """
    import unicodedata

    if val is None:
        return "Ο"

    try:
        if pd.isna(val):
            return "Ο"
    except Exception:
        pass

    s = unicodedata.normalize("NFKC", str(val))
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = s.strip().upper()

    aliases = {
        "": "Ο",
        "NAN": "Ο",
        "NONE": "Ο",
        "Ο": "Ο",
        "O": "Ο",
        "0": "Ο",
        "ΟΧΙ": "Ο",
        "NO": "Ο",
        "FALSE": "Ο",
        "Ν1": "Ν1",
        "N1": "Ν1",
        "Ν": "Ν",
        "N": "Ν",
        "ΝΑΙ": "Ν",
        "NAI": "Ν",
        "YES": "Ν",
        "TRUE": "Ν",
        "1": "Ν",
        "Y": "Ν",
        "Τ": "Ν",
        "ΑΙΣ": "Ν",
    }

    if s not in aliases:
        raise ValueError(
            f"Μη έγκυρη τιμή στη στήλη ΖΩΗΡΟΣ: {val!r}. "
            "Επιτρέπονται μόνο Ο, Ν1 ή Ν."
        )

    return aliases[s]

def behavior_weight(val: object) -> int:
    """Εσωτερικό βάρος κατανομής: Ο=0, Ν1=1, Ν=2."""
    return {"Ο": 0, "Ν1": 1, "Ν": 2}[norm_behavior(val)]

def behavior_positive(val: object) -> bool:
    """True για Ν1 ή Ν. Χρησιμοποιείται σε συγκεντρωτικά στατιστικά/ελέγχους."""
    return norm_behavior(val) in {"Ν1", "Ν"}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for c in df.columns:
        cc = str(c).strip()
        if cc in COLUMN_FIXES:
            rename[c] = COLUMN_FIXES[cc]
        elif cc.upper() == "GENDER":
            rename[c] = "ΦΥΛΟ"
        elif "ΦΙΛ" in cc.upper() and cc != "ΦΙΛΟΙ":
            rename[c] = "ΦΙΛΟΙ"
    if rename:
        df = df.rename(columns=rename)
    # Value normalization
    if "ΖΩΗΡΟΣ" in df.columns:
        df["ΖΩΗΡΟΣ"] = df["ΖΩΗΡΟΣ"].map(norm_behavior)
    for col in ["ΙΔΙΑΙΤΕΡΟΤΗΤΑ","ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ","ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"]:
        if col in df.columns:
            df[col] = df[col].map(norm_yesno)
    if "ΟΝΟΜΑ" in df.columns:
        df["ΟΝΟΜΑ"] = df["ΟΝΟΜΑ"].astype(str).str.strip()
    return df

def parse_friends_cell(x) -> List[str]:
    if isinstance(x, list):
        return [str(s).strip() for s in x if str(s).strip()]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(t).strip() for t in v if str(t).strip()]
    except Exception:
        pass
    parts = SAFE_SEP.split(s)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() != "nan"]

def are_mutual_friends(df: pd.DataFrame, a: str, b: str) -> bool:
    ra = df[df["ΟΝΟΜΑ"].astype(str) == str(a)]
    rb = df[df["ΟΝΟΜΑ"].astype(str) == str(b)]
    if ra.empty or rb.empty: return False
    fa = set(parse_friends_cell(ra.iloc[0].get("ΦΙΛΟΙ","")))
    fb = set(parse_friends_cell(rb.iloc[0].get("ΦΙΛΟΙ","")))
    return (str(b).strip() in fa) and (str(a).strip() in fb)

def scope_step2(df: pd.DataFrame, step1_col: str) -> Set[str]:
    s = set()
    for _, r in df.iterrows():
        placed = pd.notna(r.get(step1_col))
        z = behavior_positive(r.get("ΖΩΗΡΟΣ",""))
        i = str(r.get("ΙΔΙΑΙΤΕΡΟΤΗΤΑ","")).strip()=="Ν"
        pk = str(r.get("ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ","")).strip()=="Ν"
        if (not placed and (z or i)) or (placed and pk):
            s.add(str(r.get("ΟΝΟΜΑ","")).strip())
    return s

def mutual_pairs_in_scope(df: pd.DataFrame, scope: Set[str]):
    scope = {str(x).strip() for x in scope if str(x).strip()}
    pairs = []
    names = sorted(scope)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if are_mutual_friends(df, a, b):
                pairs.append((a,b))
    return pairs

# --------- ΝΕΑ βοηθητικά για το minimal export ---------
def extract_step1_id(step1_col_name: str) -> int:
    m = re.search(r'(?:ΒΗΜΑ1_|V1_)ΣΕΝΑΡΙΟ[_\s]*(\d+)', str(step1_col_name))
    return int(m.group(1)) if m else 1

def find_step1_scenario_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).strip().upper().startswith("ΒΗΜΑ1_ΣΕΝΑΡΙΟ_")]

def pick_core_columns(df: pd.DataFrame, core_list: Optional[List[str]] = None) -> List[str]:
    base = core_list or CORE_COLUMNS_DEFAULT
    return [c for c in base if c in df.columns]
