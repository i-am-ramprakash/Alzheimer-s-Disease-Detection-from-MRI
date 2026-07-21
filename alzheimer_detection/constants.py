"""Shared constants for the four-class educational classifier."""

CLASS_KEYS = (
    "mild_demented",
    "moderate_demented",
    "non_demented",
    "very_mild_demented",
)

CLASS_LABELS = {
    "mild_demented": "Mild Demented",
    "moderate_demented": "Moderate Demented",
    "non_demented": "Non Demented",
    "very_mild_demented": "Very Mild Demented",
}

CLASS_ALIASES = {
    "milddemented": "mild_demented",
    "milddementia": "mild_demented",
    "moderatedemented": "moderate_demented",
    "moderatedementia": "moderate_demented",
    "nondemented": "non_demented",
    "nondementia": "non_demented",
    "verymilddemented": "very_mild_demented",
    "verymilddementia": "very_mild_demented",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
