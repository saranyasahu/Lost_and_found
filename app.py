from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from PIL import Image
from datetime import datetime
import imagehash
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "lostfound-secret-key")
app.config["UPLOAD_FOLDER"]                  = "static/uploads"
app.config["SQLALCHEMY_DATABASE_URI"]        = os.environ.get("DATABASE_URL", "sqlite:///lostfound.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

CATEGORIES      = ["Electronics", "Clothing", "Accessories", "Documents", "Bags", "Keys", "Other"]
MATCH_THRESHOLD = 0.35


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class LostItem(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100))
    description = db.Column(db.String(500))
    location    = db.Column(db.String(100))
    reporter    = db.Column(db.String(100))
    phone       = db.Column(db.String(20))
    category    = db.Column(db.String(50), default="Other")
    found       = db.Column(db.Boolean, default=False)
    image       = db.Column(db.String(200))
    reported_at = db.Column(db.DateTime, default=datetime.utcnow)
    matches     = db.relationship("Match", backref="lost_item", lazy=True,
                                  foreign_keys="Match.lost_item_id")


class FoundItem(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100))
    description = db.Column(db.String(500))
    location    = db.Column(db.String(100))
    finder      = db.Column(db.String(100))
    phone       = db.Column(db.String(20))
    category    = db.Column(db.String(50), default="Other")
    image       = db.Column(db.String(200))
    reported_at = db.Column(db.DateTime, default=datetime.utcnow)
    handed_over = db.Column(db.Boolean, default=False)
    matches     = db.relationship("Match", backref="found_item", lazy=True,
                                  foreign_keys="Match.found_item_id")


class Match(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    lost_item_id  = db.Column(db.Integer, db.ForeignKey("lost_item.id"), nullable=False)
    found_item_id = db.Column(db.Integer, db.ForeignKey("found_item.id"), nullable=False)
    score         = db.Column(db.Float, default=0.0)
    status        = db.Column(db.String(20), default="pending")
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    messages      = db.relationship("ChatMessage", backref="match", lazy=True)
    notifications = db.relationship("Notification", backref="match", lazy=True)


class ChatMessage(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    match_id    = db.Column(db.Integer, db.ForeignKey("match.id"), nullable=False)
    sender      = db.Column(db.String(100))
    sender_name = db.Column(db.String(100))
    body        = db.Column(db.Text)
    sent_at     = db.Column(db.DateTime, default=datetime.utcnow)


class Notification(db.Model):
    """Stores in-app notifications for lost item owners."""
    id         = db.Column(db.Integer, primary_key=True)
    match_id   = db.Column(db.Integer, db.ForeignKey("match.id"), nullable=False)
    phone      = db.Column(db.String(20))   # owner's phone — used to look up their notifications
    message    = db.Column(db.String(500))
    is_read    = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ── Auto-migrate existing database before SQLAlchemy touches it ──────────────
import sqlite3 as _sqlite3

_possible = [
    os.path.join("instance", "lostfound.db"),
    "lostfound.db",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "lostfound.db"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "lostfound.db"),
]
_db_path = next((p for p in _possible if os.path.exists(p)), None)
if _db_path is None:
    os.makedirs("instance", exist_ok=True)
    _db_path = os.path.join("instance", "lostfound.db")

_conn = _sqlite3.connect(_db_path)
_cur  = _conn.cursor()

# Add new columns to existing tables if they don't exist yet
_migrations = [
    ("lost_item",  "phone",       "VARCHAR(20) DEFAULT ''"),
    ("found_item", "phone",       "VARCHAR(20) DEFAULT ''"),
]
for _table, _col, _typedef in _migrations:
    try:
        _cur.execute(f"ALTER TABLE {_table} ADD COLUMN {_col} {_typedef}")
        print(f"[MIGRATE] Added {_table}.{_col}")
    except _sqlite3.OperationalError:
        pass  # column already exists

# Create notification table if it doesn't exist
_cur.execute("""
    CREATE TABLE IF NOT EXISTS notification (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id   INTEGER NOT NULL REFERENCES match(id),
        phone      VARCHAR(20),
        message    VARCHAR(500),
        is_read    BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")

_conn.commit()
_conn.close()
print(f"[MIGRATE] Done — using {_db_path}")

# Point SQLAlchemy at the exact file we just migrated
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.abspath(_db_path)}"

with app.app_context():
    db.create_all()
    os.makedirs("static/uploads", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def create_notification(lost_item, found_item, match):
    """Store an in-app notification for the lost item owner."""
    msg = (
        f"🔔 Match found! Someone reported a '{found_item.name}' "
        f"found near {found_item.location} — "
        f"{int(match.score * 100)}% match with your lost '{lost_item.name}'."
    )
    notif = Notification(
        match_id   = match.id,
        phone      = lost_item.phone,
        message    = msg,
        is_read    = False,
    )
    db.session.add(notif)
    db.session.commit()
    app.logger.info(f"[NOTIF] Created notification for phone {lost_item.phone}")


def get_notifications(phone):
    """Get all notifications for a phone number, newest first."""
    if not phone:
        return []
    return Notification.query.filter_by(phone=phone)\
                             .order_by(Notification.created_at.desc()).all()


def unread_count(phone):
    """Count unread notifications for a phone number."""
    if not phone:
        return 0
    return Notification.query.filter_by(phone=phone, is_read=False).count()


def img_sim(path_a, path_b):
    """Image similarity using perceptual hashing. Returns 0.0 to 1.0."""
    try:
        with Image.open(path_a) as img_a:
            small_a = img_a.convert("RGB").resize((64, 64), Image.LANCZOS)
        with Image.open(path_b) as img_b:
            small_b = img_b.convert("RGB").resize((64, 64), Image.LANCZOS)
        hash_a = imagehash.phash(small_a)
        hash_b = imagehash.phash(small_b)
        return max(0.0, 1.0 - (hash_a - hash_b) / 64.0)
    except Exception as e:
        app.logger.warning(f"[IMG] img_sim failed: {e}")
        return 0.0


def text_score(query, items):
    """TF-IDF cosine similarity between query string and list of items."""
    if not items:
        return []
    texts = [f"{i.name} {i.description} {i.location} {i.category}" for i in items]
    if len(items) == 1:
        text        = texts[0].lower()
        query_words = [w for w in query.lower().split() if len(w) > 2]
        matches     = sum(1 for w in query_words if w in text)
        return [min(1.0, matches / max(len(query_words), 1))]
    texts.append(query)
    vec  = TfidfVectorizer(stop_words="english")
    mat  = vec.fit_transform(texts)
    sims = sk_cosine_similarity(mat[-1], mat[:-1])
    return sims[0].tolist()


def match_score(found_item, lost_items):
    """Returns {lost_item.id: score} comparing a found item against all lost items."""
    if not lost_items:
        return {}
    query    = f"{found_item.name} {found_item.description} {found_item.location} {found_item.category}"
    t_scores = text_score(query, lost_items)
    scores   = {item.id: float(s) for item, s in zip(lost_items, t_scores)}
    if found_item.image:
        found_path = os.path.join(app.config["UPLOAD_FOLDER"], found_item.image)
        if os.path.exists(found_path):
            for item in lost_items:
                if item.image:
                    lost_path = os.path.join(app.config["UPLOAD_FOLDER"], item.image)
                    if os.path.exists(lost_path):
                        iscore = img_sim(found_path, lost_path)
                        scores[item.id] = scores[item.id] * 0.4 + iscore * 0.6
    return scores


def save_image(field):
    f = request.files.get(field)
    if f and f.filename:
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))
        return f.filename
    return None


def time_ago(dt):
    if not dt:
        return ""
    seconds = int((datetime.utcnow() - dt).total_seconds())
    if seconds < 60:     return "just now"
    if seconds < 3600:   return f"{seconds//60}m ago"
    if seconds < 86400:  return f"{seconds//3600}h ago"
    if seconds < 604800: return f"{seconds//86400}d ago"
    return dt.strftime("%b %d, %Y")


app.jinja_env.globals["time_ago"]     = time_ago
app.jinja_env.filters["time_ago"]     = time_ago
app.jinja_env.globals["unread_count"] = unread_count


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — SESSION (who is viewing)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/set-phone", methods=["POST"])
def set_phone():
    """Store the user's phone in session so we can show their notifications."""
    phone = request.form.get("phone", "").strip()
    if phone:
        session["phone"] = phone
    next_url = request.form.get("next", "/")
    return redirect(next_url)


@app.route("/clear-phone")
def clear_phone():
    session.pop("phone", None)
    return redirect("/")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/notifications")
def notifications():
    phone = session.get("phone", "")
    if not phone:
        flash("Enter your phone number first to see your notifications.", "warning")
        return redirect("/")
    notifs = get_notifications(phone)
    # Mark all as read
    for n in notifs:
        if not n.is_read:
            n.is_read = True
    db.session.commit()
    return render_template("notifications.html", notifications=notifs, phone=phone)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — LOST ITEMS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    phone    = session.get("phone", "")
    category = request.args.get("category", "")
    q = LostItem.query.filter_by(found=False)
    if category and category in CATEGORIES:
        q = q.filter_by(category=category)
    items = q.order_by(LostItem.reported_at.desc()).all()
    return render_template("home.html",
                           items_with_scores=[(i, None) for i in items],
                           categories=CATEGORIES,
                           active_category=category,
                           query="",
                           session_phone=phone,
                           notif_count=unread_count(phone))


@app.route("/search", methods=["GET", "POST"])
def search():
    phone    = session.get("phone", "")
    query    = (request.args.get("q") or request.args.get("query") or
                request.form.get("q") or request.form.get("query") or "").strip()
    category = (request.args.get("category") or request.form.get("category") or "").strip()
    uploaded = request.files.get("image") if request.files else None
    has_image= bool(uploaded and uploaded.filename)

    if not query and not has_image and not category:
        return redirect("/")

    base_q = LostItem.query.filter_by(found=False)
    if category and category in CATEGORIES:
        base_q = base_q.filter_by(category=category)
    items = base_q.all()

    if not items:
        return render_template("home.html", items_with_scores=[],
                               categories=CATEGORIES, active_category=category,
                               query=query, session_phone=phone,
                               notif_count=unread_count(phone))

    score_map = {item.id: 0.0 for item in items}
    if query:
        ilike = LostItem.query.filter(
            LostItem.found == False,
            db.or_(LostItem.name.ilike(f"%{query}%"),
                   LostItem.description.ilike(f"%{query}%"),
                   LostItem.location.ilike(f"%{query}%"),
                   LostItem.category.ilike(f"%{query}%"))
        ).all()
        for item in ilike:
            score_map[item.id] = max(score_map[item.id], 0.5)
        t_scores = text_score(query, items)
        for item, s in zip(items, t_scores):
            score_map[item.id] = max(score_map[item.id], s)

    if has_image:
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        qpath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded.filename)
        uploaded.save(qpath)
        for item in items:
            if item.image:
                ipath  = os.path.join(app.config["UPLOAD_FOLDER"], item.image)
                iscore = img_sim(qpath, ipath)
                score_map[item.id] = (score_map[item.id]*0.4 + iscore*0.6) if query else iscore

    threshold = 0.0 if (not query and not has_image) else 0.1
    results   = sorted([(i, score_map[i.id]) for i in items if score_map[i.id] >= threshold],
                        key=lambda x: x[1], reverse=True)

    return render_template("home.html", items_with_scores=results,
                           categories=CATEGORIES, active_category=category,
                           query=query, session_phone=phone,
                           notif_count=unread_count(phone))


@app.route("/report-lost", methods=["POST"])
def report_lost():
    filename = save_image("image")
    phone    = request.form.get("phone", "").strip()
    item = LostItem(
        name        = request.form["item_name"],
        description = request.form["description"],
        location    = request.form["location"],
        reporter    = request.form["reporter"],
        phone       = phone,
        category    = request.form.get("category", "Other"),
        image       = filename,
    )
    db.session.add(item)
    db.session.commit()
    # Save phone in session so notifications show immediately
    if phone:
        session["phone"] = phone
    flash("Your lost item has been reported! You'll see notifications here when a match is found.", "success")
    return redirect("/")


@app.route("/mark-found/<int:item_id>")
def mark_found(item_id):
    item = LostItem.query.get(item_id)
    if item:
        item.found = True
        db.session.commit()
    return redirect("/")


@app.route("/found-items")
def found_items():
    items = LostItem.query.filter_by(found=True).order_by(LostItem.reported_at.desc()).all()
    return render_template("found.html", items=items)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — FOUND ITEMS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/report-found", methods=["GET", "POST"])
def report_found_page():
    if request.method == "GET":
        return render_template("report_found.html", categories=CATEGORIES)

    try:
        filename   = save_image("image")
        found_item = FoundItem(
            name        = request.form["item_name"],
            description = request.form["description"],
            location    = request.form["location"],
            finder      = request.form["finder"],
            phone       = request.form.get("phone", "").strip(),
            category    = request.form.get("category", "Other"),
            image       = filename,
        )
        db.session.add(found_item)
        db.session.commit()
        app.logger.info(f"[FOUND] Saved: {found_item.name}")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"[FOUND] Save failed: {e}")
        flash("Something went wrong. Please try again.", "danger")
        return redirect("/report-found")

    matches_created = 0
    try:
        lost_items = LostItem.query.filter_by(found=False).all()
        app.logger.info(f"[MATCH] Comparing against {len(lost_items)} lost items")
        scores = match_score(found_item, lost_items)

        for lost_item in lost_items:
            s = scores.get(lost_item.id, 0.0)
            app.logger.info(f"[MATCH] Score vs '{lost_item.name}': {round(s, 3)}")
            if s >= MATCH_THRESHOLD:
                existing = Match.query.filter_by(lost_item_id=lost_item.id,
                                                 found_item_id=found_item.id).first()
                if not existing:
                    m = Match(lost_item_id=lost_item.id,
                              found_item_id=found_item.id, score=s)
                    db.session.add(m)
                    db.session.commit()
                    app.logger.info(f"[MATCH] Match created! Score={round(s,3)}")
                    create_notification(lost_item, found_item, m)
                    matches_created += 1
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"[MATCH] Error: {e}")

    if matches_created:
        flash(f"Found item reported! {matches_created} potential owner(s) have been notified.", "success")
    else:
        flash("Found item reported! No strong matches yet.", "info")

    return redirect("/found-reports")


@app.route("/found-reports")
def found_reports():
    phone = session.get("phone", "")
    items = FoundItem.query.filter_by(handed_over=False).order_by(FoundItem.reported_at.desc()).all()
    return render_template("found_reports.html", items=items, categories=CATEGORIES,
                           session_phone=phone, notif_count=unread_count(phone))


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — MATCHES & VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/verify/<int:match_id>")
def verify_match(match_id):
    match = Match.query.get_or_404(match_id)
    return render_template("verify.html", match=match)


@app.route("/verify/<int:match_id>/approve", methods=["POST"])
def approve_match(match_id):
    match = Match.query.get_or_404(match_id)
    match.status = "approved"
    db.session.commit()
    flash("Match approved! You can now chat with the finder.", "success")
    return redirect(url_for("chat", match_id=match_id))


@app.route("/verify/<int:match_id>/reject", methods=["POST"])
def reject_match(match_id):
    match = Match.query.get_or_404(match_id)
    match.status = "rejected"
    db.session.commit()
    flash("Match rejected. We'll keep looking.", "info")
    return redirect("/")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — CHAT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/chat/<int:match_id>", methods=["GET", "POST"])
def chat(match_id):
    match = Match.query.get_or_404(match_id)
    if match.status != "approved":
        flash("Chat is not available until the owner approves the match.", "warning")
        return redirect(url_for("verify_match", match_id=match_id))

    if request.method == "POST":
        body = request.form.get("body", "").strip()
        if body:
            msg = ChatMessage(match_id=match_id,
                              sender=request.form.get("sender"),
                              sender_name=request.form.get("sender_name"),
                              body=body)
            db.session.add(msg)
            if request.form.get("mark_recovered"):
                match.lost_item.found        = True
                match.found_item.handed_over = True
            db.session.commit()

    messages = ChatMessage.query.filter_by(match_id=match_id).order_by(ChatMessage.sent_at).all()
    return render_template("chat.html", match=match, messages=messages)


@app.route("/matches")
def all_matches():
    phone   = session.get("phone", "")
    matches = Match.query.order_by(Match.created_at.desc()).all()
    return render_template("matches.html", matches=matches,
                           session_phone=phone, notif_count=unread_count(phone))


if __name__ == "__main__":
    app.run(debug=True)