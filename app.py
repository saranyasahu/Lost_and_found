from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message as MailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from PIL import Image
from datetime import datetime
import imagehash
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "lostfound-secret-key")
app.config["UPLOAD_FOLDER"]                  = "static/uploads"
app.config["SQLALCHEMY_DATABASE_URI"]        = os.environ.get("DATABASE_URL", "sqlite:///lostfound.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ── Email config — set these as environment variables on Render ───────────────
app.config["MAIL_SERVER"]        = "smtp.gmail.com"
app.config["MAIL_PORT"]          = 587
app.config["MAIL_USE_TLS"]       = True
app.config["MAIL_USERNAME"]      = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"]      = os.environ.get("MAIL_PASSWORD", "")
app.config["MAIL_DEFAULT_SENDER"]= os.environ.get("MAIL_USERNAME", "")

db   = SQLAlchemy(app)
mail = Mail(app)

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
    email       = db.Column(db.String(150))
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
    email       = db.Column(db.String(150))
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


class ChatMessage(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    match_id    = db.Column(db.Integer, db.ForeignKey("match.id"), nullable=False)
    sender      = db.Column(db.String(100))
    sender_name = db.Column(db.String(100))
    body        = db.Column(db.Text)
    sent_at     = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()
    # Ensure upload folder exists on server startup
    os.makedirs("static/uploads", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def img_sim(path_a, path_b):
    """
    Image similarity using perceptual hashing (imagehash).
    Images are resized to 64x64 before hashing to minimise memory usage.
    Returns a 0-1 score: 1.0 = identical, 0.0 = completely different.
    """
    try:
        # Open, resize small immediately, then close — minimises RAM usage
        with Image.open(path_a) as img_a:
            small_a = img_a.convert("RGB").resize((64, 64), Image.LANCZOS)
        with Image.open(path_b) as img_b:
            small_b = img_b.convert("RGB").resize((64, 64), Image.LANCZOS)
        hash_a = imagehash.phash(small_a)
        hash_b = imagehash.phash(small_b)
        diff = hash_a - hash_b
        return max(0.0, 1.0 - diff / 64.0)
    except Exception as e:
        app.logger.warning(f"[IMG] img_sim failed: {e}")
        return 0.0


def text_score(query, items):
    """TF-IDF cosine similarity between query string and list of items."""
    if not items:
        return []
    if len(items) == 1:
        text = f"{items[0].name} {items[0].description} {items[0].location}".lower()
        return [0.6 if any(w in text for w in query.lower().split()) else 0.0]
    texts = [f"{i.name} {i.description} {i.location} {i.category}" for i in items]
    texts.append(query)
    vec  = TfidfVectorizer(stop_words="english")
    mat  = vec.fit_transform(texts)
    sims = sk_cosine_similarity(mat[-1], mat[:-1])
    return sims[0].tolist()


def match_score(found_item, lost_items):
    """Returns {lost_item.id: score} for all lost items vs a found item."""
    if not lost_items:
        return {}
    query    = f"{found_item.name} {found_item.description} {found_item.location} {found_item.category}"
    t_scores = text_score(query, lost_items)
    scores   = {item.id: float(s) for item, s in zip(lost_items, t_scores)}

    # Only do image matching if the found item has an image
    # Process one image at a time to keep memory low
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


def send_notification(lost_item, found_item, score):
    """Email the lost item owner when a match is detected."""
    # Skip silently if email credentials are not configured
    if not lost_item.email:
        app.logger.info("[EMAIL] Skipped — lost item has no email")
        return
    if not app.config.get("MAIL_USERNAME"):
        app.logger.info("[EMAIL] Skipped — MAIL_USERNAME not set")
        return
    if not app.config.get("MAIL_PASSWORD"):
        app.logger.info("[EMAIL] Skipped — MAIL_PASSWORD not set")
        return

    try:
        match      = Match.query.filter_by(lost_item_id=lost_item.id,
                                           found_item_id=found_item.id).first()
        verify_url = url_for("verify_match", match_id=match.id, _external=True)
        msg = MailMessage(
            subject=f"Possible match found for your lost {lost_item.name}!",
            recipients=[lost_item.email],
            html=f"""
            <h2>Good news, {lost_item.reporter}!</h2>
            <p>Someone reported finding an item that matches your lost
            <strong>{lost_item.name}</strong> with a
            <strong>{int(score*100)}% confidence score</strong>.</p>
            <p><b>Found item:</b> {found_item.name}<br>
               <b>Description:</b> {found_item.description}<br>
               <b>Found at:</b> {found_item.location}<br>
               <b>Finder:</b> {found_item.finder}</p>
            <p>Click below to review the match and contact the finder:</p>
            <a href="{verify_url}" style="background:#198754;color:white;padding:10px 20px;
               border-radius:6px;text-decoration:none;font-weight:bold;">
               Review Match & Contact Finder
            </a>
            <p style="color:#888;font-size:12px;margin-top:20px;">
            If this is not your item, you can reject the match on that page.
            </p>"""
        )
        mail.send(msg)
        app.logger.info(f"[EMAIL] Sent to {lost_item.email}")
    except Exception as e:
        app.logger.warning(f"[EMAIL] Failed to send: {e}")


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


app.jinja_env.globals["time_ago"] = time_ago


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — LOST ITEMS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    category = request.args.get("category", "")
    q = LostItem.query.filter_by(found=False)
    if category and category in CATEGORIES:
        q = q.filter_by(category=category)
    items = q.order_by(LostItem.reported_at.desc()).all()
    return render_template("home.html",
                           items_with_scores=[(i, None) for i in items],
                           categories=CATEGORIES,
                           active_category=category,
                           query="")


@app.route("/search", methods=["GET", "POST"])
def search():
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
                               categories=CATEGORIES, active_category=category, query=query)

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
                           categories=CATEGORIES, active_category=category, query=query)


@app.route("/report-lost", methods=["POST"])
def report_lost():
    filename = save_image("image")
    item = LostItem(
        name=request.form["item_name"], description=request.form["description"],
        location=request.form["location"], reporter=request.form["reporter"],
        email=request.form.get("email", ""), category=request.form.get("category", "Other"),
        image=filename,
    )
    db.session.add(item)
    db.session.commit()
    flash("Your lost item has been reported. We'll notify you if someone finds it!", "success")
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
            name=request.form["item_name"], description=request.form["description"],
            location=request.form["location"], finder=request.form["finder"],
            email=request.form.get("email", ""), category=request.form.get("category", "Other"),
            image=filename,
        )
        db.session.add(found_item)
        db.session.commit()
        app.logger.info(f"[FOUND] Saved found item: {found_item.name}")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"[FOUND] Failed to save found item: {e}")
        flash("Something went wrong saving the item. Please try again.", "danger")
        return redirect("/report-found")

    # Run auto-matching separately — errors here should NOT break the page
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
                    m = Match(lost_item_id=lost_item.id, found_item_id=found_item.id, score=s)
                    db.session.add(m)
                    db.session.commit()
                    app.logger.info(f"[MATCH] Match created! Score={round(s,3)}")
                    send_notification(lost_item, found_item, s)
                    matches_created += 1
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"[MATCH] Matching error: {e}")
        # Don't crash — item is already saved, just skip matching

    if matches_created:
        flash(f"Found item reported! {matches_created} potential owner(s) notified by email.", "success")
    else:
        flash("Found item reported! No strong matches yet — owners will be notified as new lost reports come in.", "info")

    return redirect("/found-reports")


@app.route("/found-reports")
def found_reports():
    items = FoundItem.query.filter_by(handed_over=False).order_by(FoundItem.reported_at.desc()).all()
    return render_template("found_reports.html", items=items, categories=CATEGORIES)


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
                match.lost_item.found      = True
                match.found_item.handed_over = True
            db.session.commit()

    messages = ChatMessage.query.filter_by(match_id=match_id).order_by(ChatMessage.sent_at).all()
    return render_template("chat.html", match=match, messages=messages)


@app.route("/matches")
def all_matches():
    matches = Match.query.order_by(Match.created_at.desc()).all()
    return render_template("matches.html", matches=matches)


if __name__ == "__main__":
    app.run(debug=True)