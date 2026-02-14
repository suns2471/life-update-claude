import os
import json
import re
import random
import datetime
import functools
from io import StringIO

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

from db import (
    init_db,
    register_user,
    login_user,
    get_user_by_id,
    get_all_contacts,
    get_existing_contact_phones,
    get_all_categories,
    insert_contact,
    update_contact_category,
    get_contact_by_id,
    update_contact,
    delete_contact,
    clear_all_contacts,
    get_interactions,
    add_interaction,
    delete_interaction,
    get_journal_entries,
    get_journal_entry_by_date,
    save_journal_entry,
)

# ── Config ───────────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__, static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", "dev-key-change-in-production")
CORS(app, supports_credentials=True)

# Initialize database tables on startup (works with both gunicorn and flask run)
init_db()


# ── Auth helpers ─────────────────────────────────────────────────
def require_login(func):
    """Decorator: return 401 if the user isn't logged in."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Not logged in"}), 401
        return func(*args, **kwargs)
    return wrapper


# ── Gemini AI helper ─────────────────────────────────────────────
def get_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")


# ── Serve the frontend ───────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    """Lightweight ping so the frontend can check if the server is alive."""
    return jsonify({"status": "ok"})


# ── API: Authentication ──────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def api_register():
    """Create a new account with email, name, and password."""
    try:
        body = request.get_json() or {}
        email = body.get("email", "").strip().lower()
        name = body.get("name", "").strip()
        password = body.get("password", "")
        if not email or not name or not password:
            return jsonify({"error": "All fields are required"}), 400
        user_id = register_user(email, name, password)
        session["user_id"] = user_id
        user = get_user_by_id(user_id)
        return jsonify({"message": "Account created", "user": user})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/login", methods=["POST"])
def api_login():
    """Log in with email and password."""
    try:
        body = request.get_json() or {}
        email = body.get("email", "").strip().lower()
        password = body.get("password", "")
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        user_id = login_user(email, password)
        if user_id is None:
            return jsonify({"error": "Invalid email or password"}), 401
        session["user_id"] = user_id
        user = get_user_by_id(user_id)
        return jsonify({"message": "Logged in", "user": user})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/me", methods=["GET"])
def api_me():
    """Return the current logged-in user, or 401."""
    user_id = session.get("user_id")
    if user_id is None:
        return jsonify({"user": None}), 401
    user = get_user_by_id(user_id)
    if user is None:
        session.clear()
        return jsonify({"user": None}), 401
    return jsonify({"user": user})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Clear the session."""
    session.clear()
    return jsonify({"message": "Logged out"})


# ── API: Contacts ────────────────────────────────────────────────
@app.route("/api/contacts", methods=["GET"])
@require_login
def api_get_contacts():
    user_id = session["user_id"]
    try:
        contacts = get_all_contacts(user_id)
        categories = get_all_categories(user_id)
        return jsonify({"contacts": contacts, "categories": categories})
    except Exception as e:
        return jsonify({"contacts": [], "categories": [], "error": str(e)})


@app.route("/api/contacts/upload", methods=["POST"])
@require_login
def api_upload_contacts():
    user_id = session["user_id"]
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        content = file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        if "Category" not in df.columns:
            df["Category"] = "Uncategorized"

        # Figure out which columns are "extra" (not Name or Category)
        base_cols = {"Name", "Category"}
        extra_cols = [c for c in df.columns if c not in base_cols]

        # Replace only THIS user's contacts with the new CSV data
        clear_all_contacts(user_id)
        for _, row in df.iterrows():
            name = str(row.get("Name", ""))
            category = str(row.get("Category", "Uncategorized"))
            extra = {col: str(row[col]) for col in extra_cols if pd.notna(row[col])}
            insert_contact(user_id, name, category, extra if extra else None)

        return jsonify({"message": "Contacts uploaded successfully", "count": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Phone formatting helper ─────────────────────────────────────
def format_phone(raw):
    """
    Normalize a phone number string.

    US numbers (10 digits, or 11 starting with 1) → +1 (XXX) XXX-XXXX
    International numbers → +CC (rest grouped)
    Anything else → cleaned up but returned as-is.
    """
    # Strip everything except digits and leading +
    has_plus = raw.strip().startswith("+")
    digits = re.sub(r"[^\d]", "", raw)

    if not digits:
        return raw  # nothing to work with

    # US: 10 digits or 11 starting with 1
    if len(digits) == 10:
        return f"+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits[0] == "1":
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"

    # International: has a + prefix or more than 11 digits
    if has_plus or len(digits) > 11:
        # Group digits in chunks for readability: +CC XXXX XXXX XX…
        # Insert spaces every 4 digits after the first group
        spaced = ""
        for i, d in enumerate(digits):
            if i > 0 and i % 4 == 0:
                spaced += " "
            spaced += d
        return f"+{spaced}"

    # Fallback: return cleaned-up original
    return raw.strip()


# ── vCard helpers ───────────────────────────────────────────────

# Map common vCard type parameters to friendly labels
_TYPE_LABELS = {
    "CELL": "Cell", "MOBILE": "Cell",
    "HOME": "Home", "WORK": "Work",
    "MAIN": "Main", "FAX": "Fax",
    "IPHONE": "iPhone", "OTHER": "Other",
    "PREF": None,  # "preferred" isn't a useful label — skip it
}


def _extract_type(prop_part):
    """Pull a human-friendly type label from a vCard property line like
    TEL;type=CELL;type=VOICE  →  'Cell'
    """
    params = prop_part.upper().split(";")[1:]  # everything after the property name
    for p in params:
        p = p.replace("TYPE=", "").strip()
        label = _TYPE_LABELS.get(p)
        if label:
            return label
        # If it's not in our map but looks like a plain word, title-case it
        if p.isalpha() and p not in ("VOICE", "INTERNET", "PREF", "CHARSET=UTF-8"):
            return p.title()
    return None


def _add_field(contact, base_key, value, type_label=None):
    """Add a field to a contact dict, auto-numbering duplicates.

    First occurrence  → "Phone" (or "Phone (Cell)" if type_label given)
    Second occurrence → "Phone 2" / "Phone 2 (Work)"
    etc.
    """
    # Build the display key
    suffix = f" ({type_label})" if type_label else ""

    # Try the base key first, then "base 2", "base 3", …
    key = f"{base_key}{suffix}"
    if key not in contact:
        contact[key] = value
        return

    # Key taken — find next available number
    for n in range(2, 20):
        key = f"{base_key} {n}{suffix}" if suffix else f"{base_key} {n}"
        if key not in contact:
            contact[key] = value
            return


def parse_vcf(text):
    """
    Parse a vCard (.vcf) file and return a list of contact dicts.

    Handles the standard fields that iOS/macOS Contacts exports:
    FN (full name), TEL (phone), EMAIL, ORG (company), TITLE,
    NOTE, ADR (address), URL, and X-SOCIALPROFILE.
    Multiple phones, emails, addresses, and social profiles are
    all captured with type labels.
    """
    # Unfold continuation lines (lines starting with a space or tab)
    text = re.sub(r"\r\n[ \t]", "", text)
    text = re.sub(r"\r?\n[ \t]", "", text)

    contacts = []
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.upper() == "BEGIN:VCARD":
            current = {}
            continue
        if line.upper() == "END:VCARD":
            if current and current.get("Name"):
                contacts.append(current)
            current = None
            continue
        if current is None:
            continue

        # Split "PROPERTY;params:value"
        if ":" not in line:
            continue
        prop_part, value = line.split(":", 1)
        prop_name = prop_part.split(";")[0].upper()

        # Decode quoted-printable if flagged
        if "ENCODING=QUOTED-PRINTABLE" in prop_part.upper():
            value = re.sub(
                r"=([0-9A-Fa-f]{2})",
                lambda m: chr(int(m.group(1), 16)),
                value,
            )

        # Unescape common vCard escapes
        value = value.replace("\\n", " ").replace("\\,", ",").replace("\\;", ";").strip()
        if not value:
            continue

        type_label = _extract_type(prop_part)

        if prop_name == "FN":
            current["Name"] = value
        elif prop_name == "TEL":
            _add_field(current, "Phone", format_phone(value), type_label)
        elif prop_name == "EMAIL":
            _add_field(current, "Email", value, type_label)
        elif prop_name == "ORG":
            current["Company"] = value.replace(";", " ").strip()
        elif prop_name == "TITLE":
            current["Title"] = value
        elif prop_name == "NOTE":
            current["Notes"] = value
        elif prop_name == "ADR":
            parts = [p.strip() for p in value.split(";") if p.strip()]
            if parts:
                _add_field(current, "Address", ", ".join(parts), type_label)
        elif prop_name == "URL":
            _add_field(current, "Website", value, type_label)
        elif prop_name == "X-SOCIALPROFILE":
            # iOS stores social profiles with x-user param for the username
            # e.g. X-SOCIALPROFILE;type=twitter;x-user=johndoe:https://…
            username = ""
            for param in prop_part.split(";"):
                if param.upper().startswith("X-USER="):
                    username = param.split("=", 1)[1]
            display = username if username else value
            platform = type_label or "Social"
            current[platform] = display

    return contacts


@app.route("/api/contacts/parse-vcf", methods=["POST"])
@require_login
def api_parse_vcf():
    """Parse a vCard file, flag duplicates, and return for preview."""
    user_id = session["user_id"]
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        content = file.read().decode("utf-8", errors="replace")
        contacts = parse_vcf(content)

        if not contacts:
            return jsonify({"error": "No contacts found in the file"}), 400

        # Flag contacts that already exist (phone-number match)
        existing_phones = get_existing_contact_phones(user_id)
        for c in contacts:
            # Collect all phone digits from this parsed contact
            contact_phones = set()
            for k, v in c.items():
                if k.startswith("Phone") and v:
                    digits = re.sub(r"[^\d]", "", v)
                    if digits:
                        contact_phones.add(digits)
            # Mark as duplicate if ANY phone number already exists
            c["_duplicate"] = bool(contact_phones & existing_phones)

        return jsonify({"contacts": contacts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/import-selected", methods=["POST"])
@require_login
def api_import_selected():
    """Append only the user-selected contacts (no wiping existing data)."""
    user_id = session["user_id"]
    try:
        body = request.get_json()
        contacts = body.get("contacts", [])

        if not contacts:
            return jsonify({"error": "No contacts selected"}), 400

        for c in contacts:
            c.pop("_duplicate", None)  # strip internal flag
            name = c.pop("Name", "")
            if not name:
                continue
            category = c.pop("Category", "Uncategorized")
            extra = c if c else None
            insert_contact(user_id, name, category, extra)

        return jsonify({"message": "Contacts imported successfully", "count": len(contacts)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/category", methods=["POST"])
@require_login
def api_update_category():
    """Update a single contact's Category by its database id."""
    try:
        body = request.get_json()
        contact_id = body.get("id")
        new_category = body.get("category")
        if contact_id is None or new_category is None:
            return jsonify({"error": "id and category are required"}), 400

        update_contact_category(int(contact_id), new_category)
        return jsonify({"message": "Category updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/add", methods=["POST"])
@require_login
def api_add_contact():
    """Manually add a single contact with any fields."""
    user_id = session["user_id"]
    try:
        body = request.get_json()
        name = body.get("Name", "").strip()
        if not name:
            return jsonify({"error": "Name is required"}), 400
        category = body.get("Category", "Uncategorized")
        extra = {k: v for k, v in body.items()
                 if k not in ("Name", "Category") and v and str(v).strip()}
        contact_id = insert_contact(user_id, name, category, extra if extra else None)
        return jsonify({"message": "Contact added", "contact_id": contact_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/clear", methods=["POST"])
@require_login
def api_clear_contacts():
    """Delete all contacts for the current user."""
    user_id = session["user_id"]
    try:
        clear_all_contacts(user_id)
        return jsonify({"message": "All contacts cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/<int:contact_id>", methods=["GET"])
@require_login
def api_get_contact(contact_id):
    """Return a single contact with all its fields."""
    user_id = session["user_id"]
    try:
        contact = get_contact_by_id(contact_id, user_id)
        if not contact:
            return jsonify({"error": "Contact not found"}), 404
        return jsonify({"contact": contact})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/<int:contact_id>", methods=["PUT"])
@require_login
def api_update_contact(contact_id):
    """Update a contact's Name, Category, and all extra fields."""
    user_id = session["user_id"]
    try:
        body = request.get_json()
        name = body.get("Name", "").strip()
        if not name:
            return jsonify({"error": "Name is required"}), 400
        category = body.get("Category", "Uncategorized")
        # Everything except Name and Category goes into extra_fields
        extra = {k: v for k, v in body.items()
                 if k not in ("Name", "Category") and v and str(v).strip()}
        update_contact(contact_id, user_id, name, category, extra if extra else None)
        return jsonify({"message": "Contact updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/<int:contact_id>", methods=["DELETE"])
@require_login
def api_delete_contact(contact_id):
    """Delete a single contact by its database id."""
    user_id = session["user_id"]
    try:
        delete_contact(contact_id, user_id)
        return jsonify({"message": "Contact deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Contact Interactions ────────────────────────────────────
@app.route("/api/contacts/<int:contact_id>/interactions", methods=["GET"])
@require_login
def api_get_interactions(contact_id):
    """Return all interactions for a contact, newest first."""
    user_id = session["user_id"]
    try:
        interactions = get_interactions(contact_id, user_id)
        return jsonify({"interactions": interactions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/<int:contact_id>/interactions", methods=["POST"])
@require_login
def api_add_interaction(contact_id):
    """Add a new interaction note for a contact."""
    user_id = session["user_id"]
    try:
        body = request.get_json()
        date = body.get("date", "")
        note = body.get("note", "").strip()
        if not date or not note:
            return jsonify({"error": "Date and note are required"}), 400
        add_interaction(contact_id, user_id, date, note)
        return jsonify({"message": "Interaction added"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interactions/<int:interaction_id>", methods=["DELETE"])
@require_login
def api_delete_interaction(interaction_id):
    """Delete a single interaction."""
    user_id = session["user_id"]
    try:
        delete_interaction(interaction_id, user_id)
        return jsonify({"message": "Interaction deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contacts/random", methods=["GET"])
@require_login
def api_random_contact():
    user_id = session["user_id"]
    try:
        contacts = get_all_contacts(user_id)
        category = request.args.get("category")
        if category and category != "All Contacts":
            contacts = [c for c in contacts if c.get("Category") == category]
        if not contacts:
            return jsonify({"error": "No contacts found"}), 404
        person = random.choice(contacts)
        return jsonify({"contact": person})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Journals ────────────────────────────────────────────────
TAB_MAP = {"life": "life_journal", "work": "work_journal"}


@app.route("/api/journal/<journal_type>", methods=["GET"])
@require_login
def api_get_journal(journal_type):
    user_id = session["user_id"]
    table = TAB_MAP.get(journal_type)
    if not table:
        return jsonify({"error": "Invalid journal type"}), 400
    try:
        entries = get_journal_entries(user_id, table)
        return jsonify({"entries": entries})
    except Exception as e:
        return jsonify({"entries": [], "error": str(e)})


@app.route("/api/journal/<journal_type>/<date>", methods=["GET"])
@require_login
def api_get_journal_entry(journal_type, date):
    """Return the single journal entry for this user + date, or null."""
    user_id = session["user_id"]
    table = TAB_MAP.get(journal_type)
    if not table:
        return jsonify({"error": "Invalid journal type"}), 400
    try:
        entry = get_journal_entry_by_date(user_id, table, date)
        return jsonify({"entry": entry})
    except Exception as e:
        return jsonify({"entry": None, "error": str(e)})


@app.route("/api/journal/<journal_type>", methods=["POST"])
@require_login
def api_save_journal(journal_type):
    user_id = session["user_id"]
    table = TAB_MAP.get(journal_type)
    if not table:
        return jsonify({"error": "Invalid journal type"}), 400
    try:
        body = request.get_json()
        date_str = body.get("date", datetime.date.today().strftime("%Y-%m-%d"))
        entry1 = body.get("entry1", "")
        entry2 = body.get("entry2", "")
        entry3 = body.get("entry3", "")
        save_journal_entry(user_id, table, date_str, entry1, entry2, entry3)
        return jsonify({"message": "Entry saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: AI Summaries ────────────────────────────────────────────
@app.route("/api/summary/<journal_type>", methods=["POST"])
@require_login
def api_generate_summary(journal_type):
    user_id = session["user_id"]
    label_map = {"life": "LIFE", "work": "WORK"}
    table = TAB_MAP.get(journal_type)
    label = label_map.get(journal_type)
    if not table:
        return jsonify({"error": "Invalid journal type"}), 400
    try:
        entries = get_journal_entries(user_id, table)
        if not entries:
            return jsonify({"error": "No entries found"}), 404

        body = request.get_json() or {}
        timeframe = body.get("timeframe", "all")

        # Convert to DataFrame for easy date filtering
        df = pd.DataFrame(entries)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            now = pd.Timestamp.now()
            # timeframe is a number of days (e.g. "7", "30", "90") or "all"
            if timeframe != "all":
                try:
                    days = int(timeframe)
                    df = df[df["Date"] >= now - pd.Timedelta(days=days)]
                except (ValueError, TypeError):
                    pass  # fall through to "all" if not a valid number

        if df.empty:
            return jsonify({"error": "No entries in the selected timeframe"}), 404

        model = get_gemini_model()
        prompt = f"Summarize these {label} entries concisely:\n\n{df.tail(10).to_string()}"
        res = model.generate_content(prompt)
        return jsonify({"summary": res.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5001)
