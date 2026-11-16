"""Compatibility module alias for callers using the old path."""
import sys
from .manuals import search
sys.modules[__name__] = search
