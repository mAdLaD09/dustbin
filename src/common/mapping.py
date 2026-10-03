"""Conservative source-label to SegreSmart class mapping."""
WET = {"biodegradable", "organic"}
DRY = {"paper", "plastic", "glass", "metal", "cardboard", "cloth", "recyclable", "non-recyclable"}
REJECT = {"e-waste", "ewaste", "hazardous"}

def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").split())

def map_label(label: str, enable_reject: bool = True) -> str | None:
    key = normalize_label(label)
    if key in WET: return "wet"
    if key in DRY: return "dry"
    if key in REJECT: return "reject" if enable_reject else None
    return None
