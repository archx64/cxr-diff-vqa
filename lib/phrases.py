# phrases.py
import json
from collections import defaultdict

# --- tiny helper to normalize strings
def _norm(s):
    return (s or "").strip().lower()

def load_keyinfo(path):
    """
    Reads all_diseases.json into a dict: study_id -> {'entity': {...}, 'no_entity': [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for it in data:
        sid = str(it.get("study_id"))
        idx[sid] = {
            "entity": it.get("entity", {}),
            "no_entity": it.get("no_entity", []),
        }
    return idx

def _entity_descriptors(ent_dict):
    """
    ent_dict: like {'pneumothorax': {'entity_name': 'pneumothorax', 'location': [...], 'level': [...], ...}, ...}
    Returns set of canonical descriptors like:
      "pneumothorax|left|apical|minimal"
    """
    desc = set()
    for name, info in ent_dict.items():
        name = _norm(info.get("entity_name") or name)
        locs  = info.get("location") or []
        lvl   = info.get("level") or []
        # Build a few canonical combos (location x level; fallback to just name)
        if not locs and not lvl:
            desc.add(f"{name}")
        else:
            if not locs: locs = [""]
            if not lvl:  lvl  = [""]
            for L in locs:
                for V in lvl:
                    segs = [name]
                    if _norm(L): segs.append(_norm(L))
                    if _norm(V): segs.append(_norm(V))
                    desc.add("|".join(segs))
    return desc

def diff_keyinfo(cur_info, ref_info):
    """
    Returns (added, removed, changed) sets of descriptors between current and reference.
    We treat 'changed' as same disease appearing with different attributes.
    """
    cur_e = _entity_descriptors(cur_info.get("entity", {}))
    ref_e = _entity_descriptors(ref_info.get("entity", {}))
    added  = cur_e - ref_e
    removed= ref_e - cur_e

    # changed: same disease name appears on both sides but attributes differ
    # crude heuristic: match by prefix (disease name before first '|')
    cur_by_name = defaultdict(set); ref_by_name = defaultdict(set)
    for d in cur_e:
        nm = d.split("|",1)[0]
        cur_by_name[nm].add(d)
    for d in ref_e:
        nm = d.split("|",1)[0]
        ref_by_name[nm].add(d)
    changed = set()
    for nm in cur_by_name.keys() & ref_by_name.keys():
        if cur_by_name[nm] != ref_by_name[nm]:
            # flag as changed but remove from added/removed to avoid double counting
            changed.update(cur_by_name[nm] ^ ref_by_name[nm])
            added   -= cur_by_name[nm]
            removed -= ref_by_name[nm]

    return added, removed, changed

def phrase_from_delta(added, removed, changed, max_items=3):
    """
    Turn deltas into a short natural phrase (<= ~20 tokens).
    Examples:
      "new left apical minimal pneumothorax; resolved basal atelectasis"
    """
    def nice(d):
        parts = d.split("|")
        name = parts[0]
        attrs = [p for p in parts[1:] if p]
        if attrs:
            return f"{' '.join(attrs)} {name}"
        return name

    segs = []
    if added:
        a = "; ".join(nice(x) for x in list(added)[:max_items])
        segs.append(f"new {a}")
    if removed:
        r = "; ".join(nice(x) for x in list(removed)[:max_items])
        segs.append(f"resolved {r}")
    if changed:
        c = "; ".join(nice(x) for x in list(changed)[:max_items])
        segs.append(f"changed {c}")

    if not segs:
        return "no significant change"
    return "; ".join(segs)

def build_diff_phrase(study_id_cur, study_id_ref, keyinfo_index):
    """
    Public API: given current/ref study_ids (as str), and preloaded keyinfo_index, return phrase.
    """
    cur = keyinfo_index.get(str(study_id_cur))
    ref = keyinfo_index.get(str(study_id_ref))
    if not cur or not ref:
        return "no significant change"
    added, removed, changed = diff_keyinfo(cur, ref)
    return phrase_from_delta(added, removed, changed)