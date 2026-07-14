from flask import Flask, render_template, request
import os
from utils.voice_auth import enroll_user, authenticate_user, get_registered_users

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    uploaded_files = request.files.getlist("voice_samples")

    if not username:
        return render_template(
            "register.html", message="Please enter a username.", message_type="error"
        )

    if not uploaded_files or uploaded_files[0].filename == "":
        return render_template(
            "register.html",
            message="Please upload at least one voice sample.",
            message_type="error",
        )

    allowed_extensions = {".wav", ".mp3"}
    for file in uploaded_files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            return render_template(
                "register.html",
                message=f"Unsupported file type: '{file.filename}'. Please upload .wav or .mp3 files only.",
                message_type="error",
            )

    saved_paths = []
    for i, file in enumerate(uploaded_files):
        filename = f"{username}_{i}.wav"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        saved_paths.append(filepath)

    try:
        enroll_user(username, saved_paths)
    except ValueError as e:
        return render_template("register.html", message=str(e), message_type="error")

    return render_template(
        "register.html",
        message=f"'{username}' registered successfully with {len(saved_paths)} sample(s)!",
        message_type="success",
    )


@app.route("/authenticate", methods=["GET", "POST"])
def authenticate():
    users = get_registered_users()

    if request.method == "GET":
        return render_template("authenticate.html", users=users)

    username = request.form.get("username", "").strip()
    uploaded_file = request.files.get("voice_sample")

    if not uploaded_file or uploaded_file.filename == "":
        return render_template(
            "authenticate.html",
            users=users,
            error="Please upload a voice sample.",
            selected_user=username,
        )

    ext = os.path.splitext(uploaded_file.filename)[1].lower()
    if ext not in {".wav", ".mp3"}:
        return render_template(
            "authenticate.html",
            users=users,
            error=f"Unsupported file type: '{uploaded_file.filename}'. Please upload a .wav or .mp3 file.",
            selected_user=username,
        )

    filepath = os.path.join(UPLOAD_FOLDER, "auth_temp.wav")
    uploaded_file.save(filepath)

    try:
        result = authenticate_user(username, filepath)
    except ValueError as e:
        return render_template(
            "authenticate.html", users=users, error=str(e), selected_user=username
        )

    if "error" in result:
        return render_template(
            "authenticate.html",
            users=users,
            error=result["error"],
            selected_user=username,
        )

    return render_template(
        "authenticate.html", users=users, result=result, selected_user=username
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
